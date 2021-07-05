import torch

import learning.utils
from learning.model import ACModel


class Agent:
    """An agent.

    It is able:
    - to choose an action given an observation,
    - to analyze the feedback (i.e. reward and done state) of its action."""

    def __init__(self, obs_space, action_space, model_dir,
                 device=None, argmax=False, num_envs=1, agent_index=0):
        obs_space, self.preprocess_obss = learning.utils.get_obss_preprocessor(
            obs_space)
        self.acmodel = ACModel(obs_space, action_space)
        self.device = device
        self.argmax = argmax
        self.num_envs = num_envs

        if self.acmodel.recurrent:
            self.memories = torch.zeros(
                self.num_envs, self.acmodel.memory_size, device=self.device)

        try:
            state = learning.utils.get_model_state(model_dir)[agent_index]
        except IndexError:
            state_len = len(learning.utils.get_model_state(model_dir))
            state = learning.utils.get_model_state(
                model_dir)[agent_index % state_len]
        self.acmodel.load_state_dict(state)
        self.acmodel.to(self.device)
        self.acmodel.eval()
        if hasattr(self.preprocess_obss, "vocab"):
            self.preprocess_obss.vocab.load_vocab(
                learning.utils.get_vocab(model_dir))

    def get_actions(self, obss, agent):
        agent_obs = [None]*len(obss)
        for index in range(len(obss)):
            agent_obs[index] = obss[index][agent]
        preprocessed_obss = self.preprocess_obss(agent_obs, device=self.device)

        with torch.no_grad():
            if self.acmodel.recurrent:
                dist, _ = self.acmodel(
                    preprocessed_obss)
            else:
                dist, _ = self.acmodel(preprocessed_obss)

        if self.argmax:
            actions = dist.probs.max(1, keepdim=True)[1]
        else:
            actions = dist.sample()

        return actions.cpu().numpy()

    def get_action(self, obs, agent):
        return self.get_actions([obs], agent)

    def analyze_feedbacks(self, rewards, dones):
        if self.acmodel.recurrent:
            masks = 1 - torch.tensor(dones, dtype=torch.float,
                                     device=self.device).unsqueeze(1)
            self.memories *= masks

    def analyze_feedback(self, reward, done):
        return self.analyze_feedbacks([reward], [done])
