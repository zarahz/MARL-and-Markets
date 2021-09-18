import random
import numpy as np
import torch
import torch.nn.functional as F
import math

from learning.utils.format import default_preprocess_obss
from learning.dqn.utils import Transition
from learning.ppo.utils.penv import ParallelEnv


class DQN():
    """The Proximal Policy Optimization algorithm
    ([Schulman et al., 2015](https://arxiv.org/abs/1707.06347))."""

    def __init__(self, envs, agents, memory, policy_nets, target_nets, device=None, num_frames_per_proc=None, gamma=0.99,  lr=0.001, batch_size=256, epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=10000, target_update=1000, adam_eps=1e-8, action_space=1, preprocess_obss=None):
        """
        Parameters:
        ----------
        envs : list
            a list of environments that will be run in parallel
        models : list 
            model list of length agents, containing torch.Modules
        num_frames_per_proc : int
            the number of frames collected by every process for an update
        gamma : float
            the discount for future rewards
        lr : float
            the learning rate for optimizers
        recurrence : int
            the number of steps the gradient is propagated back in time
        preprocess_obss : function
            a function that takes observations returned by the environment
            and converts them into the format that the model can handle
        """
        self.num_frames_per_proc = num_frames_per_proc or 128
        assert target_update <= self.num_frames_per_proc, "target update needs to be smaller or equal to frames-per-proc!"

        self.action_space = action_space
        self.memory = memory

        self.policy_nets = policy_nets
        self.target_nets = target_nets
        self.device = device

        self.batch_size = batch_size
        self.agents = agents

        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.steps_done = 0
        self.target_update = target_update

        self.gamma = gamma
        self.preprocess_obss = preprocess_obss or default_preprocess_obss

        self.num_procs = len(envs)
        self.num_frames = self.num_frames_per_proc * self.num_procs

        self.env = ParallelEnv(envs)

        # Control parameters

        self.optimizers = []
        for agent in range(self.agents):
            self.optimizers.append(torch.optim.Adam(
                self.policy_nets[agent].parameters(), lr, eps=adam_eps))

        # Initialize experience values

        shape = (self.num_frames_per_proc, self.num_procs)
        multi_shape = (self.num_frames_per_proc, agents, self.num_procs)
        self.obs = self.env.reset()
        self.obss = [None]*(shape[0])
        self.mask = torch.ones(shape[1], device=self.device)
        self.masks = torch.zeros(*shape, device=self.device)
        self.actions = torch.zeros(
            *multi_shape, device=self.device, dtype=torch.int)
        self.rewards = torch.zeros(
            (self.num_frames_per_proc, self.num_procs, agents), device=self.device)

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

        self.log_episode_return = torch.zeros(
            (self.num_procs, agents), device=self.device)

        self.log_episode_num_frames = torch.zeros(
            self.num_procs, device=self.device)

        self.log_done_counter = 0
        self.log_return = []  # [[0]*agents] * self.num_procs
        self.log_num_frames = []  # [0] * self.num_procs
        self.log_coloration_percentage = []

    def train(self, frames_to_capture):
        # save rgb frames for gif creation
        capture_frames = []

        # count all episodes that finish with a fully colored grid
        log_fully_colored = 0

        # frames-per-proc = 128 that means 128 times the (16) parallel envs are played through and logged.
        # If worst case and all environments are played until max_steps (25) are reached it can at least finish
        # 5 times and log its rewards (that means there are at least 5*16=80 rewards in log_return)
        for i in range(self.num_frames_per_proc):
            # agent variables
            joint_actions = []

            for agent in range(self.agents):
                agent_obs = [None]*len(self.obs)
                # Do one agent-environment interaction
                for index in range(len(self.obs)):
                    agent_obs[index] = self.obs[index][agent]

                preprocessed_obs = self.preprocess_obss(
                    agent_obs, device=self.device)

                joint_actions.append(
                    self.select_action(agent, preprocessed_obs))

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
            self.rewards[i] = torch.tensor(reward, device=self.device)

            # Update log values
            self.log_episode_reset_fields[i] = torch.tensor(
                [env_info['reset_fields'] for env_info in info], device=self.device)

            if any('trades' in env_info for env_info in info):
                self.log_episode_trades[i] = torch.tensor(
                    [env_info['trades'] for env_info in info], device=self.device)
            self.log_episode_return += torch.tensor(
                reward, device=self.device, dtype=torch.float)
            self.log_episode_num_frames += torch.ones(
                self.num_procs, device=self.device)

            # Store the transition in memory (old obs, actions, new obs, reward)
            next_state = []
            for index, env_done in enumerate(done):
                next_state.append(None if env_done else self.obs[index])

            self.memory.push(
                self.obss[i], joint_actions, next_state, reward)

            huber_loss = []
            for agent in range(self.agents):
                huber_loss.append(self.optimize_model(agent))

            for done_index, done_ in enumerate(done):
                if done_:
                    self.log_done_counter += 1
                    self.log_return.append(
                        self.log_episode_return[done_index].tolist())
                    self.log_num_frames.append(
                        self.log_episode_num_frames[done_index].item())
                    self.log_coloration_percentage.extend(
                        [env_info['coloration_percentage'] for env_info in info])
                    log_fully_colored += sum([env_info['fully_colored']
                                             for env_info in info])

            # transpose rewards to [agent, processes] to multiplicate a mask of [processes] with it
            log_episode_return_transposed = self.log_episode_return.transpose(
                0, 1) * self.mask
            # then transpose back to tensor shape (processes, reward_of_agent)
            self.log_episode_return *= log_episode_return_transposed.transpose(
                0, 1)
            self.log_episode_num_frames *= self.mask

            # Update the target networks
            # TARGET_UPDATE = giving your network more time to consider many
            # actions that have taken place recently instead of updating all the time
            # episode % args.target_update == 0:
            if i > 0 and i % self.target_update == 0:
                for agent in range(self.agents):
                    self.target_nets[agent].load_state_dict(
                        self.policy_nets[agent].state_dict())

        # logs for all agents
        self.log_num_reset_fields = torch.sum(
            self.log_episode_reset_fields, dim=1)
        self.log_trades = torch.sum(
            self.log_episode_trades, dim=1)
        keep = max(self.log_done_counter, self.num_procs)

        logs = {
            "num_frames": self.num_frames,
            "capture_frames": capture_frames,
            "reward_agent_"+str(agent): [episode_log_return[agent] for episode_log_return in self.log_return[-keep:]],
            "trades": self.log_trades[-keep:].tolist(),
            "num_reset_fields": self.log_num_reset_fields[-keep:].tolist(),
            "grid_coloration_percentage": self.log_coloration_percentage,
            "fully_colored": log_fully_colored}
        return logs

    def select_action(self, agent, state):
        """
        select_action will select an action accordingly to an epsilon greedy policy.
        Simply put, we’ll sometimes use our networks for choosing the action, and sometimes we’ll just sample one uniformly.
        The probability of choosing a random action will start at EPS_START
        and will decay exponentially towards EPS_END. EPS_DECAY controls the rate of the decay.
        """
        self.steps_done += 1
        sample = random.random()
        eps_threshold = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
            math.exp(-1. * self.steps_done / self.epsilon_decay)
        with torch.no_grad():
            if sample > eps_threshold:
                result = self.policy_nets[agent](
                    state.image)  # .unsqueeze(0)
                action = [env_res.max(0)[1]
                          for env_res in result]
            else:
                action = [random.randrange(self.action_space)
                          for _ in range(len(self.env.envs))]

            return torch.tensor(action)

    def optimize_model(self, agent):
        """
        Here, you can find an optimize_model function that performs a single step of the optimization. It first samples a
        batch, concatenates all the tensors into a single one, computes Q(st,at) and V(st+1)=maxaQ(st+1,a), and combines
        them into our loss. By defition we set V(s)=0 if s is a terminal state. We also use a target networks to compute
        V(st+1) for added stability. The target networks has its weights kept frozen most of the time, but is updated with
        the policy networks’s weights every so often. This is usually a set number of steps but we shall use episodes for
        simplicity.
        """
        if len(self.memory) < self.batch_size:
            return 0
        transitions = self.memory.sample(self.batch_size)
        # Transpose the batch (see http://stackoverflow.com/a/19343/3343043 for detailed explanation).
        batch = Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        next_states = []  # length = batch size
        next_states_images = []  # length = batch size
        for next_state in batch.next_state:
            boolean_vals = tuple(map(lambda env_next_state: env_next_state is not None,
                                     next_state))
            next_states.append(boolean_vals)
            next_states_images.append([env_next_state[agent]["image"]
                                      for env_next_state in next_state if env_next_state is not None])

        states = []  # length = batch size
        for state in batch.state:
            states.append([env_state[agent]["image"]
                           for env_state in state if env_state is not None])

        agent_actions = []
        for actions in batch.action:
            agent_actions.append(actions[agent])

        agent_rewards = []
        for rewards in batch.reward:
            agent_reward = [item for sublist in rewards[agent:]
                            for item in sublist]
            agent_rewards.append(agent_reward)

        # flatten lists
        next_states = [item for sublist in next_states for item in sublist]
        next_states_images = [
            item for sublist in next_states_images for item in sublist]
        states = [item for sublist in states for item in sublist]
        agent_actions = [item for sublist in agent_actions for item in sublist]
        agent_rewards = [item for sublist in agent_rewards for item in sublist]

        non_final_mask = torch.tensor(
            next_states, device=self.device, dtype=torch.bool)
        non_final_next_states = torch.tensor(
            next_states_images, device=self.device, dtype=torch.float)

        state_batch = torch.tensor(
            states, device=self.device, dtype=torch.float)

        action_batch = torch.as_tensor(
            agent_actions).unsqueeze(0).T
        reward_batch = torch.as_tensor(
            agent_rewards).unsqueeze(0).T

        # Compute Q(s_t, a) - the networks computes Q(s_t), then we select the
        # columns of actions taken
        state_action_values = self.policy_nets[agent](
            state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        next_state_values = torch.zeros(
            self.batch_size*self.num_procs, device=self.device)
        next_state_values[non_final_mask] = self.target_nets[agent](
            non_final_next_states).max(1)[0].detach()

        # Compute the expected Q values
        expected_state_action_values = (
            next_state_values * self.gamma) + reward_batch[0:].T

        # calculate mean squared error -> Q learning is lowering the estimated from actual value!
        # squared, so that large errors have more relevance!
        # downside -> large values would change target drastically on big error!
        # mse = (next_state_values - expected_state_action_values)**2
        # alternative is absolute mean error -> calculates big and small errors
        # downside -> large errors don't have much relevance!
        # mae = abs(next_state_values - expected_state_action_values)

        # Compute Huber loss
        # huber loss combines errors with condition to value big errors while preventing drastic changes
        loss = F.smooth_l1_loss(state_action_values,
                                expected_state_action_values.transpose(0, 1))  # .unsqueeze(1)

        # Optimize the model
        self.optimizers[agent].zero_grad()
        loss.backward()
        for param in self.policy_nets[agent].parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizers[agent].step()

        return loss
