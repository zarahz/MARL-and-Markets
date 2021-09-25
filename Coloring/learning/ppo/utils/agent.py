import torch

from Coloring.learning.ppo.model import ACModel
from Coloring.learning.utils.storage import get_model_state
from Coloring.learning.utils.format import get_obss_preprocessor


class Agent:
    """An agent.

    It is able:
    - to choose an action given an observation,
    - to analyze the feedback (i.e. reward and done state) of its action."""

    def __init__(self, agent_index, obs_space, action_space, model_dir,
                 device=None, argmax=False, num_envs=1):
        obs_space, self.preprocess_obss = get_obss_preprocessor(
            obs_space)
        # TODO action-space differ bet
        self.acmodel = ACModel(obs_space, action_space)
        self.device = device
        self.argmax = argmax
        self.num_envs = num_envs

        try:
            all_states = get_model_state(model_dir)
            state = all_states[agent_index]
        except IndexError:
            state_len = len(get_model_state(model_dir))
            state = get_model_state(
                model_dir)[agent_index % state_len]
        self.acmodel.load_state_dict(state)
        self.acmodel.to(self.device)
        self.acmodel.eval()

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

    def analyze_feedback(self, reward, done):
        return self.analyze_feedbacks([reward], [done])
