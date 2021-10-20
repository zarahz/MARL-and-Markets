import random
import numpy as np
import torch
import torch.nn.functional as F
import math

from learning.base import BaseAlgo

from learning.utils.format import default_preprocess_obss
from learning.dqn.utils import Transition
from learning.utils.penv import ParallelEnv


class DQN(BaseAlgo):
    """The Proximal Policy Optimization algorithm
    ([Schulman et al., 2015](https://arxiv.org/abs/1707.06347))."""

    def __init__(self, envs, agents, memory, policy_nets, target_nets, device=None, num_frames_per_proc=None, gamma=0.99, lr=0.001, batch_size=256, initial_target_update=10000, epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=10000, target_update=1000, adam_eps=1e-8, action_space=1, preprocess_obss=None):
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
        num_frames_per_proc = num_frames_per_proc or 128

        # init Base algo
        super().__init__(envs, agents, device, num_frames_per_proc, preprocess_obss)

        self.action_space = action_space
        self.memory = memory

        self.policy_nets = policy_nets
        self.target_nets = target_nets

        self.agents = agents
        self.batch_size = batch_size

        self.initial_target_update = initial_target_update
        self.target_update = target_update
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay

        self.gamma = gamma

        # Log values
        self.log_huber_loss = torch.zeros(
            (self.num_frames_per_proc, agents), device=self.device, dtype=torch.float)

        # set optimizer for each agent

        self.optimizers = []
        for agent in range(self.agents):
            self.policy_nets[agent].to(self.device)
            self.policy_nets[agent].train()
            self.target_nets[agent].to(self.device)
            self.target_nets[agent].train()
            self.optimizers.append(torch.optim.Adam(
                self.policy_nets[agent].parameters(), lr, eps=adam_eps))

    def select_action(self, agent, obs):
        """
        select_action will select an action accordingly to an epsilon greedy policy.
        Simply put, we’ll sometimes use our networks for choosing the action, and sometimes we’ll just sample one uniformly.
        The probability of choosing a random action will start at EPS_START
        and will decay exponentially towards EPS_END. EPS_DECAY controls the rate of the decay.
        """
        sample = random.random()
        eps_threshold = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
            math.exp(-1. * self.steps_done / self.epsilon_decay)
        with torch.no_grad():
            if sample > eps_threshold:
                result = self.policy_nets[agent](
                    obs.image)  # .unsqueeze(0)
                action = [env_res.max(0)[1]
                          for env_res in result]
            else:
                action = [random.randrange(self.action_space)
                          for _ in range(len(self.env.envs))]

            return torch.tensor(action)

    def mid_frame_updates(self, i, tensor_actions, done, reward):
        # Store the transition in memory (old obs, actions, new obs, reward)
        next_state = []
        for index, env_done in enumerate(done):
            next_state.append(None if env_done else self.obs[index])

        self.memory.push(self.obss[i], tensor_actions, next_state, reward)

        agents_huber_loss = []
        if self.steps_done > self.initial_target_update:
            for agent in range(self.agents):
                agents_huber_loss.append(self.optimize_model(agent))

                if self.steps_done % self.target_update == 0:
                    # Update the target networks
                    # TARGET_UPDATE = giving your network more time to consider many
                    # actions that have taken place recently instead of updating all the time
                    # episode % args.target_update == 0:
                    self.target_nets[agent].load_state_dict(
                        self.policy_nets[agent].state_dict())
        if agents_huber_loss:
            self.log_huber_loss[i] = torch.tensor(agents_huber_loss)

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
            next_states.extend(boolean_vals)
            next_states_images.extend([env_next_state[agent]["image"]
                                      for env_next_state in next_state if env_next_state is not None])

        states = []  # length = batch size
        for state in batch.state:
            states.extend([env_state[agent]["image"]
                           for env_state in state if env_state is not None])

        agent_actions = []
        for actions in batch.action:
            agent_actions.extend(actions[agent])

        agent_rewards = []
        for rewards in batch.reward:
            sublist = [sublist[agent] for sublist in rewards]
            agent_rewards.extend(sublist)

        non_final_mask = torch.tensor(
            next_states, device=self.device, dtype=torch.bool)
        non_final_next_states = torch.tensor(
            next_states_images, device=self.device, dtype=torch.float)

        state_batch = torch.tensor(
            states, device=self.device, dtype=torch.float)

        action_batch = torch.as_tensor(
            agent_actions).unsqueeze(0).T
        reward_batch = torch.as_tensor(
            agent_rewards, dtype=torch.float)

        # Compute Q(s_t, a) - the networks computes Q(s_t), then we select the
        # columns of actions taken
        state_action_values = (self.policy_nets[agent](
            state_batch).gather(1, action_batch)).float()

        # Compute V(s_{t+1}) for all next states.
        next_state_values = torch.zeros(
            self.batch_size*self.num_procs, device=self.device)
        next_state_values[non_final_mask] = self.target_nets[agent](
            non_final_next_states).max(1)[0].detach()

        # Compute the expected Q values
        expected_state_action_values = ((
            next_state_values * self.gamma) + reward_batch).unsqueeze(1).float()

        # Compute Huber loss
        # huber loss combines errors with condition to value big errors while preventing drastic changes
        loss = F.smooth_l1_loss(state_action_values.float(),
                                expected_state_action_values)  # .unsqueeze(1)

        # Optimize the model
        self.optimizers[agent].zero_grad()
        loss.backward()
        for param in self.policy_nets[agent].parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizers[agent].step()

        return loss

    def get_additional_logs(self):
        huber_loss_values = []
        for agent in range(self.agents):
            huber_loss_values.append(
                np.mean(np.array([huber_loss[agent] for huber_loss in self.log_huber_loss])))
        return {"huber_loss": huber_loss_values}

    def start_of_frame(self):
        pass

    def before_frame_starts(self):
        pass

    def on_after_reset(self):
        pass
