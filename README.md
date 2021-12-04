# Exploring the impact of Markets on Multiagent Reinforcement Learning

A current research question addresses the topic of how to bring mixed-motive agents to work together, in order to achieve a common goal. Mixed-motive describes agents that work independently and whose actions do not affect others directly. However, the agents are not able to communicate. One approach is to introduce markets, namely a shareholder market (SM) and an action market (AM). By using markets, agents gain incentives, when they act cooperatively. Shares of the SM let agents participate in the reward of others and AM enable agents to reward others for certain actions.

This thesis introduces the coloring environment and uses it to compare the application of the two markets in various agent compositions. The coloring environment lets agents move around and color the cells they visit. Visiting a cell that is already colored removes its color. The goal is to color the whole environment.

The rewards of competitive and mixed-motive agents are calculated with the amount of color presence in the environment. Cooperative agents however get one shared reward based on the overall coloration, which can lead to the credit assignment problem (CAP). Additionally, all three compositions face organizational challenges of getting in each other's way. The effectiveness of markets on these problems are analyzed in this research.

# Installation

First clone this repository and navigate into the project Folder

```
git clone https://github.com/zarahz/CMARL-CAP-and-Markets.git
cd CMARL-CAP-and-Markets/Coloring
pip install -e .
```

you can also try to install the requirements.txt

# Execution

Now you can run the project with the following algorithms

### Training Parameters

To list all training parameters execute

```
> python -m scripts.train -h
```

### DQN Learning

For training with DQN in a small 5x5 grid you can execute

```
> python -m scripts.train --algo dqn --agents 2 --max_steps 10 --size 5
```

the only required argument is `--algo`

### PPO Learning

For training with PPO in a small 5x5 grid you can execute

```
> python -m scripts.train --algo ppo --agents 2 --max_steps 10 --size 5
```

again, the only required argument is `--algo`

## Examples of the coloring environment:

![2 mixed-motive DQN agents with a SM and the "goal" addition](./assets/2-dqn-mixed-sm-goal.gif)

2 mixed-motive DQN agents with a SM and the "goal" addition, can be recreated with the following command:

```
python -m scripts.train --algo dqn --agents 2 --setting mixed-motive --market sm-goal --max-steps 8
```

![3 cooperative PPO agents and difference reward calculations](./assets/3-ppo-dr.gif)

3 cooperative PPO agents and difference reward calculations, can be recreated with the following command:

```
python -m scripts.train --algo ppo --agents 3 --setting difference-reward --grid-size 7 --max-steps 20 --frames-per-proc 256 --frames 200000 --capture-interval 15
```

![3 competitive DQN agents with a SM and the additions "goal-no-reset"](./assets/rooms-3-dqn-comp-sm-goal-no-reset.gif)

3 competitive DQN agents with a SM and the additions "goal-no-reset", can be recreated with the following command:

```
python -m scripts.train --algo dqn --agents 3 3-dqn-competitive-sm-goal-no-reset --env FourRooms-Grid-v0 --setting mixed-motive-competitive --market sm-goal-no-reset --initial-target-update 1000 --target-update 10000 --replay-size 700000 --epsilon-decay 20000 --grid-size 9 --max-steps 30 --frames-per-proc 256 --frames 200000 --capture-interval 15
```

Further command examples can be found in the powershell files located in Coloring/scripts/powershell/.
To run all in sequence simply execute the run.ps1 script in a powershell terminal (can be found in Coloring/scripts/).
