#----------------------------------------------------------------------------------------
# 3 PPO Percentage Settings
#----------------------------------------------------------------------------------------
pipenv run python -m Coloring.scripts.train --agents 3 --model 3-ppo-percentage `
    --setting percentage-reward `
    --max-steps 350 `
    --capture-interval 20

#----------------------------------------------------------------------------------------
# 3 PPO Percentage Settings with SHAREHOLDER Market
#----------------------------------------------------------------------------------------
pipenv run python -m Coloring.scripts.train --agents 3 --model 3-ppo-percentage-sm `
    --setting percentage-reward `
    --market sm `
    --max-steps 350 `
    --capture-interval 20

pipenv run python -m Coloring.scripts.train --agents 3 --model 3-ppo-percentage-sm-goal `
    --setting percentage-reward `
    --market sm-goal `
    --max-steps 350 `
    --capture-interval 20

pipenv run python -m Coloring.scripts.train --agents 3 --model 3-ppo-percentage-sm-no-reset `
    --setting percentage-reward `
    --market sm-no-reset `
    --max-steps 350 `
    --capture-interval 20

# pipenv run python -m Coloring.scripts.train --agents 3 --model 3-ppo-percentage-sm-goal-no-reset `
#     --setting percentage-reward `
#     --market sm-goal-no-reset `
#     --max-steps 350 `
#     --capture-interval 20

#----------------------------------------------------------------------------------------
# 3 PPO Percentage Settings with ACTION Market
#----------------------------------------------------------------------------------------
pipenv run python -m Coloring.scripts.train --agents 3 --model 3-ppo-percentage-am `
    --setting percentage-reward `
    --market am `
    --max-steps 350 `
    --capture-interval 20 

pipenv run python -m Coloring.scripts.train --agents 3 --model 3-ppo-percentage-am-goal `
    --setting percentage-reward `
    --market am-goal `
    --max-steps 350 `
    --capture-interval 20 
       
pipenv run python -m Coloring.scripts.train --agents 3 --model 3-ppo-percentage-am-no-reset `
    --setting percentage-reward `
    --market am-no-reset `
    --max-steps 350 `
    --capture-interval 20

# pipenv run python -m Coloring.scripts.train --agents 3 --model 3-ppo-percentage-am-goal-no-reset `
#     --setting percentage-reward `
#     --market am-goal-no-reset `
#     --max-steps 350 `
#     --capture-interval 20 