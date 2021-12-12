from learning.dqn.model import DQNModel
from learning.ppo.model import ACModel

import torch

from learning.utils.storage import get_model_state
from learning.utils.format import get_obss_preprocessor


class Agent:
    """An agent - It is able to choose an action given an observation for visualization"""

    def __init__(self, algo, agent_index, obs_space, action_space, model_dir,
                 device=None):
        obs_space, self.preprocess_obss = get_obss_preprocessor(
            obs_space)
        
        self.algo = algo
        self.device = device

        if algo == "ppo":
            self.model = ACModel(obs_space, action_space)
            all_states = get_model_state(model_dir, "model_state")
        else:
            self.model = DQNModel(obs_space, action_space)
            all_states = get_model_state(model_dir, "target_state")

        try:
            state = all_states[agent_index]
        except IndexError:
            if algo == "ppo":
                all_states = get_model_state(model_dir, "model_state")
            else:
                all_states = get_model_state(model_dir, "target_state")

            state_len = len(all_states)
            state = all_states[agent_index % state_len]
        self.model.load_state_dict(state)
        self.model.to(self.device)
        self.model.eval()

    def get_ppo_actions(self, obss, agent):
        agent_obs = [None]*len(obss)
        for index in range(len(obss)):
            agent_obs[index] = obss[index][agent]
        preprocessed_obss = self.preprocess_obss(agent_obs, device=self.device)

        with torch.no_grad():
            if self.model.recurrent:
                dist, _ = self.model(
                    preprocessed_obss)
            else:
                dist, _ = self.model(preprocessed_obss)

        actions = dist.sample()

        return actions.cpu().numpy()

    def get_dqn_actions(self, obss, agent):
        agent_obs = [None]*len(obss)
        for index in range(len(obss)):
            agent_obs[index] = obss[index][agent]
        preprocessed_obss = self.preprocess_obss(agent_obs, device=self.device)

        with torch.no_grad():
            result = self.model(preprocessed_obss.image)  # .unsqueeze(0)
            action = [env_res.max(0)[1] for env_res in result][0]

        return action.cpu().numpy()

    def get_action(self, obs, agent):
        if self.algo == "ppo":
            return self.get_ppo_actions([obs], agent)
        else:
            return self.get_dqn_actions([obs], agent)

