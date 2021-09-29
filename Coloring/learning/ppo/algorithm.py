from learning.utils.dictlist import DictList
import numpy
import torch

from learning.base import BaseAlgo


class PPO(BaseAlgo):
    """The Proximal Policy Optimization algorithm
    ([Schulman et al., 2015](https://arxiv.org/abs/1707.06347))."""

    def __init__(self, envs, agents, models, device=None, num_frames_per_proc=None, gamma=0.99, lr=0.001, gae_lambda=0.95,
                 entropy_coef=0.01, value_loss_coef=0.5, max_grad_norm=0.5, recurrence=4,
                 adam_eps=1e-8, clip_eps=0.2, epochs=4, batch_size=256, preprocess_obss=None):
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
        gae_lambda : float
            the lambda coefficient in the GAE formula
            ([Schulman et al., 2015](https://arxiv.org/abs/1506.02438))
        entropy_coef : float
            the weight of the entropy cost in the final objective
        value_loss_coef : float
            the weight of the value loss in the final objective
        max_grad_norm : float
            gradient will be clipped to be at most this value
        recurrence : int
            the number of steps the gradient is propagated back in time
        preprocess_obss : function
            a function that takes observations returned by the environment
            and converts them into the format that the model can handle
        """

        num_frames_per_proc = num_frames_per_proc or 128

        # init Base algo
        super().__init__(envs, agents, device, num_frames_per_proc, preprocess_obss)

        self.clip_eps = clip_eps
        self.epochs = epochs
        self.batch_size = batch_size
        self.agents = agents
        self.models = models

        self.entropy_coef = entropy_coef
        self.lr = lr
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.value_loss_coef = value_loss_coef
        self.max_grad_norm = max_grad_norm
        self.recurrence = recurrence

        # Log values

        self.agent_dists = []
        self.agent_values = []

        multi_shape = (self.num_frames_per_proc, agents, self.num_procs)
        self.values = torch.zeros(*multi_shape, device=self.device)
        self.log_probs = torch.zeros(
            *multi_shape, device=self.device, dtype=torch.int)  # multi_shape, device=self.device, dtype=torch.int)

        # Configure models and optimizers
        self.optimizers = []
        for agent in range(self.agents):
            self.models[agent].to(self.device)
            self.models[agent].train()
            self.optimizers.append(torch.optim.Adam(
                self.models[agent].parameters(), lr, eps=adam_eps))
        self.batch_num = 0

    def select_action(self, agent, obs):
        # reduce memory consumption for computations
        with torch.no_grad():
            dist, value = self.models[agent](obs)
            self.agent_dists.append(dist)
            self.agent_values.append(value)

        action = dist.sample()
        return action

    def algo_specific_updates(self, index, actions, done=None, reward=None):
        self.values[index] = torch.stack(self.agent_values)
        self.log_probs[index] = torch.stack([self.agent_dists[agent].log_prob(
            actions[agent]) for agent in range(self.agents)])

    def reset_algo_logs_each_frame(self):
        self.agent_dists = []
        self.agent_values = []

    def optimize_model(self, exps, agent, logs):
        # Collect experiences

        for _ in range(self.epochs):
            # Initialize log values

            log_entropies = []
            log_values = []
            log_policy_losses = []
            log_value_losses = []
            log_grad_norms = []

            for inds in self._get_batches_starting_indexes():
                # Initialize batch values

                batch_entropy = 0
                batch_value = 0
                batch_policy_loss = 0
                batch_value_loss = 0
                batch_loss = 0

                for i in range(self.recurrence):
                    sb = exps[inds + i]

                    # Compute loss

                    # if self.models[0].recurrent:
                    dist, value = self.models[agent](sb.obs)
                    # else:
                    #     dist, value = self.model(sb.obs)

                    entropy = dist.entropy().mean()
                    # PPO Formulas are calculated here (clip and loss)
                    ratio = torch.exp(dist.log_prob(
                        sb.actions) - sb.log_probs)
                    surr1 = ratio * sb.advantage
                    surr2 = torch.clamp(
                        ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * sb.advantage
                    # ??? Warum negativ und warum im durchschnitt? durchschnitt als erwartungswert?
                    policy_loss = -torch.min(surr1, surr2).mean()

                    value_clipped = sb.value + \
                        torch.clamp(value - sb.value, -
                                    self.clip_eps, self.clip_eps)
                    surr1 = (value - sb.returnn).pow(2)
                    surr2 = (value_clipped - sb.returnn).pow(2)
                    value_loss = torch.max(surr1, surr2).mean()

                    loss = policy_loss - self.entropy_coef * \
                        entropy + self.value_loss_coef * value_loss

                    # Update batch values

                    batch_entropy += entropy.item()
                    batch_value += value.mean().item()
                    batch_policy_loss += policy_loss.item()
                    batch_value_loss += value_loss.item()
                    batch_loss += loss

                # Update batch values

                batch_entropy /= self.recurrence
                batch_value /= self.recurrence
                batch_policy_loss /= self.recurrence
                batch_value_loss /= self.recurrence
                batch_loss /= self.recurrence

                # Update actor-critic

                self.optimizers[agent].zero_grad()
                batch_loss.backward()
                grad_norm = sum(p.grad.data.norm(2).item() **
                                2 for p in self.models[agent].parameters()) ** 0.5
                torch.nn.utils.clip_grad_norm_(
                    self.models[agent].parameters(), self.max_grad_norm)
                self.optimizers[agent].step()

                # Update log values

                log_entropies.append(batch_entropy)
                log_values.append(batch_value)
                log_policy_losses.append(batch_policy_loss)
                log_value_losses.append(batch_value_loss)
                log_grad_norms.append(grad_norm)

        # Log some values

        logs["entropy"].append(numpy.mean(log_entropies))
        logs["value"].append(numpy.mean(log_values))
        logs["policy_loss"].append(numpy.mean(log_policy_losses))
        logs["value_loss"].append(numpy.mean(log_value_losses))
        logs["grad_norm"].append(numpy.mean(log_grad_norms))

        return logs

    def fill_and_reshape_experiences(self, agent):
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
        exps.actions = self.actions[:,
                                    agent, :].transpose(0, 1).reshape(-1)
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

    def _get_batches_starting_indexes(self):
        """Gives, for each batch, the indexes of the observations given to
        the model and the experiences used to compute the loss at first.
        First, the indexes are the integers from 0 to `self.num_frames` with a step of
        `self.recurrence`, shifted by `self.recurrence//2` one time in two for having
        more diverse batches. Then, the indexes are splited into the different batches.
        Returns
        -------
        batches_starting_indexes : list of list of int
            the indexes of the experiences to be used at first for each batch
        """

        indexes = numpy.arange(0, self.num_frames, self.recurrence)
        indexes = numpy.random.permutation(indexes)

        # Shift starting indexes by self.recurrence//2 half the time
        if self.batch_num % 2 == 1:
            indexes = indexes[(indexes + self.recurrence) %
                              self.num_frames_per_proc != 0]
            indexes += self.recurrence // 2
        self.batch_num += 1

        num_indexes = self.batch_size // self.recurrence
        batches_starting_indexes = [indexes[i:i+num_indexes]
                                    for i in range(0, len(indexes), num_indexes)]

        return batches_starting_indexes

    def get_additional_logs(self):
        pass
