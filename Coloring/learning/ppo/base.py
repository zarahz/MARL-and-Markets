from abc import ABC, abstractmethod
import torch
import numpy as np

# from learning.ppo.utils import
from learning.ppo.utils import DictList, ParallelEnv
import learning.utils


class BaseAlgo(ABC):
    """The base class for RL algorithms."""

    def __init__(self, envs, agents, models, device, num_frames_per_proc, gamma, gae_lambda, preprocess_obss):

        # Store parameters

        self.env = ParallelEnv(envs)
        self.agents = agents
        self.models = models
        self.device = device
        self.num_frames_per_proc = num_frames_per_proc
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.preprocess_obss = preprocess_obss or learning.utils.default_preprocess_obss

        # Configure all models
        for agent in range(agents):
            self.models[agent].to(self.device)
            self.models[agent].train()

        # Store helpers values

        self.num_procs = len(envs)
        self.num_frames = self.num_frames_per_proc * self.num_procs

        # Initialize experience values

        shape = (self.num_frames_per_proc, self.num_procs)
        multi_shape = (self.num_frames_per_proc, agents, self.num_procs)
        self.obs = self.env.reset()
        self.obss = [None]*(shape[0])
        self.mask = torch.ones(shape[1], device=self.device)
        self.masks = torch.zeros(*shape, device=self.device)
        self.actions = torch.zeros(
            *multi_shape, device=self.device, dtype=torch.int)  # multi_shape, device=self.device, dtype=torch.int)
        self.values = torch.zeros(*multi_shape, device=self.device)
        self.rewards = torch.zeros(
            (self.num_frames_per_proc, self.num_procs, agents), device=self.device)
        self.advantages = torch.zeros(*multi_shape, device=self.device)
        self.log_probs = torch.zeros(
            *multi_shape, device=self.device, dtype=torch.int)  # multi_shape, device=self.device, dtype=torch.int)

        # Initialize log values

        # variable to sum up all reset fields
        self.log_num_reset_fields = torch.zeros(shape[0], dtype=torch.int)
        # variable to save the reset fields for each episode
        self.log_episode_reset_fields = torch.zeros(
            shape, dtype=torch.int)

        # variable to sum up all executed trades
        self.log_trades = torch.zeros(shape[0], dtype=torch.int)
        # variable to save the trades for each episode
        self.log_episode_trades = torch.zeros(
            shape, dtype=torch.int)

        self.log_episode_num_frames = torch.zeros(
            self.num_procs, device=self.device)

        self.log_done_counter = 0
        self.log_return = []  # [[0]*agents] * self.num_procs
        self.log_num_frames = []  # [0] * self.num_procs
        self.log_fully_colored = 0
        self.log_coloration_percentage = []

    def prepare_experiences(self, frames_to_capture):
        """Collects rollouts and computes advantages.
        Runs several environments concurrently. The next actions are computed
        in a batch mode for all environments at the same time. The rollouts
        and advantages from all environments are concatenated together.
        Returns
        -------
        exps : Array of DictList
            Contains actions, rewards, advantages etc as attributes.
            Each attribute, e.g. `exps.reward` has a shape
            (self.num_frames_per_proc * num_envs, ...). k-th block
            of consecutive `self.num_frames_per_proc` frames contains
            data obtained from the k-th environment. Be careful not to mix
            data from different environments!
        logs : dict
            Useful stats about the training process, including the average
            reward, policy loss, value loss, etc.
        """
        # save rgb frames for gif creation
        capture_frames = []

        # frames-per-proc = 128 that means 128 times the (16) parallel envs are played through and logged.
        # If worst case and all environments are played until max_steps (25) are reached it can at least finish
        # 5 times and log its rewards (that means there are at least 5*16=80 rewards in log_return)
        for i in range(self.num_frames_per_proc):
            # agent variables
            dists = []
            values = []
            joint_actions = []
            for agent in range(self.agents):
                agent_obs = [None]*len(self.obs)
                # Do one agent-environment interaction
                for index in range(len(self.obs)):
                    agent_obs[index] = self.obs[index][agent]

                preprocessed_obs = self.preprocess_obss(
                    agent_obs, device=self.device)

                # reduce memory consumption for computations
                with torch.no_grad():
                    dist, value = self.models[agent](preprocessed_obs)
                    dists.append(dist)
                    values.append(value)

                # create joint actions
                action = dist.sample()  # shape torch.Size([16])
                joint_actions.append(action)

            # convert agentList(actionTensor) into tensor of
            # tensor_actions: torch.Size([1, 16]) (agents, 16): 16 consecutive actions of each agent
            tensor_actions = torch.stack(joint_actions[:])
            obs, reward, done, info = self.env.step(
                [action.cpu().numpy() for action in joint_actions])

            # capture the last n enviroment steps
            if(i > self.num_frames_per_proc-frames_to_capture):
                capture_frames.append(np.moveaxis(
                    self.env.envs[0].render("rgb_array"), 2, 0))

            self.obss[i] = self.obs  # old obs
            self.obs = obs  # set old obs to new experience obs
            self.masks[i] = self.mask
            self.mask = 1 - \
                torch.tensor(done, device=self.device, dtype=torch.float)

            self.actions[i] = tensor_actions
            self.values[i] = torch.stack(values)
            self.rewards[i] = torch.tensor(reward, device=self.device)
            self.log_probs[i] = torch.stack([dists[agent].log_prob(
                tensor_actions[agent]) for agent in range(self.agents)])

            # Update log values
            self.log_episode_reset_fields[i] = torch.tensor(
                [env_info['reset_fields'] for env_info in info], device=self.device)
            if any('trades' in env_info for env_info in info):
                self.log_episode_trades[i] = torch.tensor(
                    [env_info['trades'] for env_info in info], device=self.device)
            self.log_episode_num_frames += torch.ones(
                self.num_procs, device=self.device)

            for done_index, done_ in enumerate(done):
                if done_:
                    self.log_done_counter += 1
                    self.log_return.append(reward[done_index])
                    self.log_num_frames.append(
                        self.log_episode_num_frames[done_index].item())
                    self.log_coloration_percentage.append(
                        info[done_index]['coloration_percentage'])
                    self.log_fully_colored += info[done_index]['fully_colored']
            self.log_episode_num_frames *= self.mask

        # --- all environment actions are now done -> Add advantage and return to experiences
        for agent in range(self.agents):
            agent_obs = [None]*len(self.obs)
            for index in range(len(self.obs)):
                agent_obs[index] = self.obs[index][agent]
            preprocessed_obs = self.preprocess_obss(
                agent_obs, device=self.device)
            with torch.no_grad():
                _, next_value = self.models[agent](preprocessed_obs)

            for i in reversed(range(self.num_frames_per_proc)):
                next_mask = self.masks[i +
                                       1] if i < self.num_frames_per_proc - 1 else self.mask
                next_value = self.values[i +
                                         1][agent] if i < self.num_frames_per_proc - 1 else next_value
                next_advantage = self.advantages[i +
                                                 1][agent] if i < self.num_frames_per_proc - 1 else 0

                delta = self.rewards[i][:, agent] + self.gamma * \
                    next_value * next_mask - self.values[i][agent]
                # advantage function is calculated here!
                self.advantages[i][agent] = delta + self.gamma * \
                    self.gae_lambda * next_advantage * next_mask

        # logs for all agents
        self.log_num_reset_fields = torch.sum(
            self.log_episode_reset_fields, dim=1)
        self.log_trades = torch.sum(
            self.log_episode_trades, dim=1)
        keep = max(self.log_done_counter, self.num_procs)

        logs = {
            "capture_frames": capture_frames,
            "trades": self.log_trades[-keep:].tolist(),
            "num_reset_fields": self.log_num_reset_fields[-keep:].tolist(),
            "grid_coloration_percentage": self.log_coloration_percentage,
            "fully_colored": self.log_fully_colored,
            "num_frames_per_episode": self.log_num_frames,
            "num_frames": self.num_frames
        }

        for agent in range(self.agents):
            logs.update({
                "reward_agent_"+str(agent): [episode_log_return[agent] for episode_log_return in self.log_return]
            })

        # reset values
        self.log_done_counter = 0
        self.log_num_frames = []
        self.log_fully_colored = 0
        self.log_coloration_percentage = []
        self.log_return = []

        return logs

    def collect_experience(self, agent):
        # Define experience:
        #   the whole experience is the concatenation of the experience
        #   of each process.
        # In comments below:
        #   - T is self.num_frames_per_proc, 128
        #   - P is self.num_procs, 16
        #   - D is the dimensionality.

        exps = DictList()
        # obs length is 2048 = 16*128
        exps.obs = [self.obss[i][j][agent]
                    for j in range(self.num_procs)
                    for i in range(self.num_frames_per_proc)]
        if self.models[0].recurrent:
            # T x P -> P x T -> (P * T) x 1
            exps.mask = self.masks.transpose(0, 1).reshape(-1).unsqueeze(1)
        # for all tensors below, T x P -> P x T -> P * T
        # self.agents, self.num_frames_per_proc*self.num_procs)
        exps.actions = self.actions[:, agent, :].transpose(0, 1).reshape(-1)
        exps.value = self.values[:, agent, :].transpose(0, 1).reshape(-1)
        exps.reward = self.rewards[:, :, agent].transpose(0, 1).reshape(-1)
        exps.advantage = self.advantages[:,
                                         agent, :].transpose(0, 1).reshape(-1)
        exps.returnn = exps.value + exps.advantage
        exps.log_probs = self.log_probs[:,
                                        agent, :].transpose(0, 1).reshape(-1)

        # Preprocess experiences

        exps.obs = self.preprocess_obss(exps.obs, device=self.device)

        return exps

    @abstractmethod
    def update_parameters(self):
        pass