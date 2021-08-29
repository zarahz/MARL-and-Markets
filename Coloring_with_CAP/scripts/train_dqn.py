import datetime
import math
import random
from itertools import count

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torch.optim as optim

from learning.dqn.config import *
# Training
from learning.dqn.utils import *
from learning.dqn.utils.envs import device, env
from learning.dqn.utils.envs import get_screen
from learning.dqn.utils.replay import ReplayMemory
from learning.dqn.model import DQN

from learning.ppo.utils.storage import get_model_dir, save_status


date = datetime.datetime.now().strftime("%y-%m-%d-%H-%M-%S")
default_model_name = f"EmptyGrid_DQN_{date}"
model_name = MODEL_NAME or default_model_name
model_dir = get_model_dir(model_name)

policy_net = DQN(
    env.observation_space["image"].shape, env.action_space.n).to(device)
target_net = DQN(
    env.observation_space["image"].shape, env.action_space.n).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.RMSprop(policy_net.parameters())
memory = ReplayMemory(10000)

steps_done = 0


def select_action(state):
    """
    select_action will select an action accordingly to an epsilon greedy policy.
    Simply put, we’ll sometimes use our networks for choosing the action, and sometimes we’ll just sample one uniformly.
    The probability of choosing a random action will start at EPS_START
    and will decay exponentially towards EPS_END. EPS_DECAY controls the rate of the decay.
    """
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            obs = torch.tensor(state[0]["image"], device=device, dtype=torch.float).transpose(
                2, 1).transpose(0, 1)
            result = policy_net(obs.unsqueeze(0))
            return result.max(1)[1].view(1, 1)
    else:
        return torch.tensor([[random.randrange(2)]], device=device, dtype=torch.long)


episode_durations = []


def plot_durations():
    """
    a helper for plotting the durations of episodes, along with an average over the last 100 episodes
    (the measure used in the official evaluations).
    The plot will be underneath the cell containing the main training loop,
    and will update after every episode.
    """
    plt.figure(2)
    plt.clf()
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())
    # Take 100 episode averages and plot them too
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated


def optimize_model():
    """
    Here, you can find an optimize_model function that performs a single step of the optimization. It first samples a
    batch, concatenates all the tensors into a single one, computes Q(st,at) and V(st+1)=maxaQ(st+1,a), and combines
    them into our loss. By defition we set V(s)=0 if s is a terminal state. We also use a target networks to compute
    V(st+1) for added stability. The target networks has its weights kept frozen most of the time, but is updated with
    the policy networks’s weights every so often. This is usually a set number of steps but we shall use episodes for
    simplicity.
    """
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see http://stackoverflow.com/a/19343/3343043 for detailed explanation).
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                            batch.next_state)), device=device, dtype=torch.uint8)
    non_final_next_states = torch.tensor([s[0]["image"] for s in batch.next_state
                                          if s is not None], device=device, dtype=torch.float)
    non_final_next_states = non_final_next_states.transpose(
        3, 2).transpose(1, 2)
    state_batch = torch.tensor([state[0]["image"]
                                for state in batch.state if state is not None], device=device, dtype=torch.float)
    # reshape into (batch_size, 3, 7 (agent_view), 7 (agent_view))
    state_batch = state_batch.transpose(3, 2).transpose(1, 2)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a) - the networks computes Q(s_t), then we select the
    # columns of actions taken
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    next_state_values[non_final_mask] = target_net(
        non_final_next_states).max(1)[0].detach()
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    loss = F.smooth_l1_loss(state_action_values,
                            expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()


# Main Training Loop
if __name__ == '__main__':
    num_episodes = 50
    update = 0
    for i_episode in range(num_episodes):
        print(i_episode)
        # Initialize the environment and state
        state = env.reset()
        for t in count():
            # env.render('human')

            # Select and perform an action
            action = select_action(state)

            print("step: ", env.env.step_count, ", action: ", action)

            obs, reward, done, _ = env.step([action.item()])
            reward = torch.tensor([reward], device=device)

            # Observe new state
            if not done:
                next_state = obs
            else:
                next_state = None

            # Store the transition in memory
            memory.push(state, action, next_state, reward)

            # Move to the next state
            state = next_state

            # Perform one step of the optimization (on the target networks)
            optimize_model()
            if done:
                episode_durations.append(t + 1)
                # plot_durations()
                break
        # Update the target networks
        if i_episode % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())

        if i_episode % 10 == 0:
            update += 1
            status = {"episode": i_episode, "update": update,
                      "model_state": [policy_net.state_dict()],
                      "optimizer_state": [optimizer.state_dict()]}
            save_status(status, model_dir)
    print('Complete')
    # plt.ioff()
    # plt.show()
