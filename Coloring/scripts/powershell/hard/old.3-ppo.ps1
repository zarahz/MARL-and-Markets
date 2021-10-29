
pipenv run python -m Coloring.scripts.train --algo ppo --agents 3 --model 3-ppo-dr `
    --max-steps 350 `
    --capture-interval 20 `
    --setting difference-reward

#----------------------------------------------------------------------------------------
# 3 PPO Settings
#----------------------------------------------------------------------------------------
pipenv run python -m Coloring.scripts.train --algo ppo --agents 3 --model 3-ppo `
    --max-steps 350 `
    --capture-interval 20 `

#----------------------------------------------------------------------------------------
# 3 PPO Settings with SHAREHOLDER Market
#----------------------------------------------------------------------------------------
pipenv run python -m Coloring.scripts.train --algo ppo --agents 3 --model 3-ppo-sm `
    --market sm `
    --max-steps 350 `
    --capture-interval 20 `

pipenv run python -m Coloring.scripts.train --algo ppo --agents 3 --model 3-ppo-sm-goal `
    --market sm-goal `
    --max-steps 350 `
    --capture-interval 20

pipenv run python -m Coloring.scripts.train --algo ppo --agents 3 --model 3-ppo-sm-no-reset `
    --market sm-no-reset `
    --max-steps 350 `
    --capture-interval 20

pipenv run python -m Coloring.scripts.train --algo ppo --agents 3 --model 3-ppo-sm-no-debt `
    --market sm-no-debt `
    --max-steps 350 `
    --capture-interval 20

pipenv run python -m Coloring.scripts.train --algo ppo --agents 3 --model 3-ppo-sm-goal-no-reset-no-debt `
    --market sm-goal-no-reset-no-debt `
    --max-steps 350 `
    --capture-interval 20

# pipenv run python -m Coloring.scripts.train --algo ppo --agents 3 --model 3-ppo-sm-goal-no-reset `
#     --market sm-goal-no-reset `
#     --max-steps 350 `
#     --capture-interval 20

#----------------------------------------------------------------------------------------
# 3 PPO Settings with ACTION Market
#----------------------------------------------------------------------------------------
pipenv run python -m Coloring.scripts.train --algo ppo --agents 3 --model 3-ppo-am `
    --market am `
    --max-steps 350 `
    --capture-interval 20 

pipenv run python -m Coloring.scripts.train --algo ppo --agents 3 --model 3-ppo-am-goal `
    --market am-goal `
    --max-steps 350 `
    --capture-interval 20
    
pipenv run python -m Coloring.scripts.train --algo ppo --agents 3 --model 3-ppo-am-no-reset `
    --market am-no-reset `
    --max-steps 350 `
    --capture-interval 20

pipenv run python -m Coloring.scripts.train --algo ppo --agents 3 --model 3-ppo-am-no-debt `
    --market am-no-debt `
    --max-steps 350 `
    --capture-interval 20

pipenv run python -m Coloring.scripts.train --algo ppo --agents 3 --model 3-ppo-am-goal-no-reset-no-debt `
    --market am-goal-no-reset-no-debt `
    --max-steps 350 `
    --capture-interval 20

# pipenv run python -m Coloring.scripts.train --algo ppo --agents 3 --model 3-ppo-am-goal-no-reset `
#     --market am-goal-no-reset `
#     --max-steps 350 `
#     --capture-interval 20