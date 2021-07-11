from abc import ABC, abstractmethod
import torch
import numpy as np

# from learning.utils import
from learning.utils import DictList, ParallelEnv, default_preprocess_obss


class BaseAlgo(ABC):
    """The base class for RL algorithms."""

    def __init__(self, envs, agents, models, device, num_frames_per_proc, discount, gae_lambda, preprocess_obss):

        # Store parameters

        self.env = ParallelEnv(envs)
        self.agents = agents
        self.models = models
        self.device = device
        self.num_frames_per_proc = num_frames_per_proc
        self.discount = discount
        self.gae_lambda = gae_lambda
        self.preprocess_obss = preprocess_obss or default_preprocess_obss

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

        self.log_episode_return = torch.zeros(
            (self.num_procs, agents), device=self.device)
        self.log_episode_num_frames = torch.zeros(
            self.num_procs, device=self.device)

        self.log_done_counter = 0
        self.log_return = []  # [[0]*agents] * self.num_procs
        self.log_num_frames = []  # [0] * self.num_procs

    def prepare_experiences(self):
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
            obs, reward, done, _ = self.env.step(
                [action.cpu().numpy() for action in joint_actions])

            # Update experiences values

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

            self.log_episode_return += torch.tensor(
                reward, device=self.device, dtype=torch.float)
            self.log_episode_num_frames += torch.ones(
                self.num_procs, device=self.device)

            for i, done_ in enumerate(done):
                if done_:
                    self.log_done_counter += 1
                    self.log_return.append(self.log_episode_return[i].tolist())
                    self.log_num_frames.append(
                        self.log_episode_num_frames[i].item())

            # transpose rewards to [agent, processes] to multiplicate a mask of [processes] with it
            log_episode_return_transposed = self.log_episode_return.transpose(
                0, 1) * self.mask
            # then transpose back to tensor shape (processes, reward_of_agent)
            self.log_episode_return *= log_episode_return_transposed.transpose(
                0, 1)
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

                delta = self.rewards[i][:, agent] + self.discount * \
                    next_value * next_mask - self.values[i][agent]
                # advantage function is calculated here!
                self.advantages[i][agent] = delta + self.discount * \
                    self.gae_lambda * next_advantage * next_mask

    def collect_experience(self, agent):
        # Define experience:
        #   the whole experience is the concatenation of the experience
        #   of each process.
        # In comments below:
        #   - T is self.num_frames_per_proc,
        #   - P is self.num_procs,
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
        # self.agents, self.num_frames_per_proc*self.num_procs)
        exps.log_probs = self.log_probs[:,
                                        agent, :].transpose(0, 1).reshape(-1)

        # Preprocess experiences

        exps.obs = self.preprocess_obss(exps.obs, device=self.device)

        # Log some values

        keep = max(self.log_done_counter, self.num_procs)

        logs = {
            "return_per_episode_agent_"+str(agent): [episode_log_return[agent] for episode_log_return in self.log_return[-keep:]],
            "num_frames_per_episode": self.log_num_frames,
            "num_frames": self.num_frames
        }
        
        if self.agents == agent+1:
            self.log_done_counter = 0
            self.log_return = self.log_return[-self.num_procs:]
            self.log_num_frames = self.log_num_frames[-self.num_procs:]

        return exps, logs

    @abstractmethod
    def update_parameters(self):
        pass
