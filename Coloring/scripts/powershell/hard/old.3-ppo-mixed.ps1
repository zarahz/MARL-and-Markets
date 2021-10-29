#----------------------------------------------------------------------------------------
# 3 PPO Mixed Motive Settings
#----------------------------------------------------------------------------------------
pipenv run python -m Coloring.scripts.train --algo ppo --agents 3 --model 3-ppo-mixed `
    --setting mixed-motive `
    --max-steps 350 `
    --capture-interval 20

#----------------------------------------------------------------------------------------
# 3 PPO Mixed Motive Settings with SHAREHOLDER Market
#----------------------------------------------------------------------------------------
pipenv run python -m Coloring.scripts.train --algo ppo --agents 3 --model 3-ppo-mixed-sm `
    --setting mixed-motive `
    --market sm `
    --max-steps 350 `
    --capture-interval 20
    
pipenv run python -m Coloring.scripts.train --algo ppo --agents 3 --model 3-ppo-mixed-sm-goal `
    --setting mixed-motive `
    --market sm-goal `
    --max-steps 350 `
    --capture-interval 20 `

pipenv run python -m Coloring.scripts.train --algo ppo --agents 3 --model 3-ppo-mixed-sm-no-reset `
    --setting mixed-motive `
    --market sm-no-reset `
    --max-steps 350 `
    --capture-interval 20

pipenv run python -m Coloring.scripts.train --algo ppo --agents 3 --model 3-ppo-mixed-sm-no-debt `
    --setting mixed-motive `
    --market sm-no-debt `
    --max-steps 350 `
    --capture-interval 20

pipenv run python -m Coloring.scripts.train --algo ppo --agents 3 --model 3-ppo-mixed-sm-goal-no-reset-no-debt `
    --setting mixed-motive `
    --market sm-goal-no-reset-no-debt `
    --max-steps 350 `
    --capture-interval 20

# pipenv run python -m Coloring.scripts.train --algo ppo --agents 3 --model 3-ppo-mixed-sm-goal-no-reset `
#     --setting mixed-motive `
#     --market sm-goal-no-reset `
#     --max-steps 350 `
#     --capture-interval 20

#----------------------------------------------------------------------------------------
# 3 PPO Mixed Motive Settings with ACTION Market
#----------------------------------------------------------------------------------------
pipenv run python -m Coloring.scripts.train --algo ppo --agents 3 --model 3-ppo-mixed-am `
    --setting mixed-motive `
    --market am `
    --max-steps 350 `
    --capture-interval 20
    
pipenv run python -m Coloring.scripts.train --algo ppo --agents 3 --model 3-ppo-mixed-am-goal `
    --setting mixed-motive `
    --market am-goal `
    --max-steps 350 `
    --capture-interval 20
    
pipenv run python -m Coloring.scripts.train --algo ppo --agents 3 --model 3-ppo-mixed-am-no-reset `
    --setting mixed-motive `
    --market am-no-reset `
    --max-steps 350 `
    --capture-interval 20

pipenv run python -m Coloring.scripts.train --algo ppo --agents 3 --model 3-ppo-mixed-am-no-debt `
    --setting mixed-motive `
    --market am-no-debt `
    --max-steps 350 `
    --capture-interval 20

pipenv run python -m Coloring.scripts.train --algo ppo --agents 3 --model 3-ppo-mixed-am-goal-no-reset-no-debt `
    --setting mixed-motive `
    --market am-goal-no-reset-no-debt `
    --max-steps 350 `
    --capture-interval 20

# pipenv run python -m Coloring.scripts.train --algo ppo --agents 3 --model 3-ppo-mixed-am-goal-no-reset `
#     --setting mixed-motive `
#     --market am-goal-no-reset `
#     --max-steps 350 `
#     --capture-interval 20