from abc import ABC, abstractmethod
import torch
import numpy as np


# from learning.ppo.utils import
from learning.utils import DictList
from learning.utils.penv import ParallelEnv
from learning.utils.format import default_preprocess_obss
from torch._C import dtype


class BaseAlgo(ABC):
    """The base class for RL algorithms."""

    def __init__(self, envs, agents, device, num_frames_per_proc, preprocess_obss):

        # Store parameters

        self.env = ParallelEnv(envs)
        self.agents = agents
        self.device = device
        self.num_frames_per_proc = num_frames_per_proc
        self.preprocess_obss = preprocess_obss or default_preprocess_obss

        # Store helpers values

        self.num_procs = len(envs)
        self.num_frames = self.num_frames_per_proc * self.num_procs

        # Initialize experience values
        self.obs = self.env.reset()
        self.on_after_reset()

        shape = (self.num_frames_per_proc, self.num_procs)
        multi_shape = (self.num_frames_per_proc, agents, self.num_procs)
        self.obss = [None]*(shape[0])
        self.mask = torch.ones(shape[1], device=self.device)
        self.masks = torch.zeros(*shape, device=self.device)
        self.actions = torch.zeros(
            *multi_shape, device=self.device, dtype=torch.int)  # multi_shape, device=self.device, dtype=torch.int)
        self.rewards = torch.zeros(
            (self.num_frames_per_proc, self.num_procs, agents), device=self.device, dtype=torch.float)
        self.advantages = torch.zeros(*multi_shape, device=self.device)

        # Initialize log values

        # variable to save the reset fields for each process
        self.log_episode_reset_fields = torch.zeros(
            self.num_procs, device=self.device)
        # variable to save the trades for each process
        self.log_episode_trades = torch.zeros(
            self.num_procs, device=self.device)
        # variable to save the steps for each process
        self.log_episode_num_frames = torch.zeros(
            self.num_procs, device=self.device)

        self.steps_done = 0
        self.log_done_counter = 0
        self.log_return = []  # [[0]*agents] * self.num_procs
        self.log_num_frames = []  # [0] * self.num_procs
        self.log_trades = []
        self.log_reset_fields = []
        self.log_fully_colored = 0
        self.log_not_fully_colored = 0
        self.log_coloration_percentage = []

    def run_and_log_parallel_envs(self, frames_to_capture):
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
        self.before_frame_starts()
        # frames-per-proc = 128 that means 128 times the (16) parallel envs are played through and logged.
        # If worst case and all environments are played until max_steps (25) are reached it can at least finish
        # 5 times and log its rewards (that means there are at least 5*16=80 rewards in log_return)
        for i in range(self.num_frames_per_proc):
            # can be used to reset/change agent variables
            self.start_of_frame()

            joint_actions = []
            for agent in range(self.agents):
                agent_obs = [None]*len(self.obs)
                # Do one agent-environment interaction
                for index in range(len(self.obs)):
                    agent_obs[index] = self.obs[index][agent]

                preprocessed_obs = self.preprocess_obss(
                    agent_obs, device=self.device)

                action = self.select_action(agent, preprocessed_obs)

                joint_actions.append(action)

            # convert agentList(actionTensor) into tensor of
            # tensor_actions: torch.Size([1, 16]) (agents, 16): 16 consecutive actions of each agent
            tensor_actions = torch.stack(joint_actions[:])
            obs, reward, done, info = self.env.step(
                [action.cpu().numpy() for action in joint_actions])

            # update steps to decay epsilon
            self.steps_done += 16  # in each env a step is made!

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
            self.rewards[i] = torch.tensor(reward, device=self.device)

            self.mid_frame_updates(i, tensor_actions, done, reward)

            # Update log values
            self.log_episode_reset_fields += torch.tensor(
                [env_info['reset_fields'] for env_info in info], device=self.device)
            if any('trades' in env_info for env_info in info):
                self.log_episode_trades += torch.tensor(
                    [env_info['trades'] for env_info in info], device=self.device)
            self.log_episode_num_frames += torch.ones(
                self.num_procs, device=self.device)

            for done_index, done_ in enumerate(done):
                if done_:
                    self.log_done_counter += 1
                    self.log_return.append(reward[done_index])
                    self.log_reset_fields.append(
                        self.log_episode_reset_fields[done_index].item())
                    self.log_trades.append(
                        self.log_episode_trades[done_index].item())
                    self.log_num_frames.append(
                        self.log_episode_num_frames[done_index].item())
                    self.log_coloration_percentage.append(
                        info[done_index]['coloration_percentage'])
                    self.log_fully_colored += info[done_index]['fully_colored']

            self.log_episode_num_frames *= self.mask
            self.log_episode_reset_fields *= self.mask
            self.log_episode_trades *= self.mask

        # logs for all agents
        logs = {
            "capture_frames": capture_frames,
            "trades": self.log_trades,
            "num_reset_fields": self.log_reset_fields,
            "grid_coloration_percentage": self.log_coloration_percentage,
            "fully_colored": self.log_fully_colored,
            "episodes": self.log_done_counter,
            "num_frames_per_episode": self.log_num_frames,
            "num_frames": self.num_frames
        }

        additional_logs = self.get_additional_logs()

        if additional_logs:
            logs.update(additional_logs)

        # agent specific logs
        for agent in range(self.agents):
            logs.update({
                "reward_agent_"+str(agent): [episode_log_return[agent] for episode_log_return in self.log_return]
            })

        # reset values
        self.reset_values()

        return logs

    def reset_values(self):
        self.log_done_counter = 0
        self.log_fully_colored = 0
        self.log_num_frames = []
        self.log_return = []
        self.log_trades = []
        self.log_reset_fields = []
        self.log_coloration_percentage = []

    @abstractmethod
    def optimize_model(self):
        pass

    @abstractmethod
    def select_action(self, agent, obs):
        pass

    @abstractmethod
    def mid_frame_updates(self, index, tensor_actions, done=None, reward=None):
        pass

    @abstractmethod
    def start_of_frame(self):
        pass

    @abstractmethod
    def before_frame_starts(self):
        pass

    @abstractmethod
    def get_additional_logs(self):
        pass

    @abstractmethod
    def on_after_reset(self):
        pass
