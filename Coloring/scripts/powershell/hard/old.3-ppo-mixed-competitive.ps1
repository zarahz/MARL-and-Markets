#----------------------------------------------------------------------------------------
# 3 PPO Mixed Competitive Settings
#----------------------------------------------------------------------------------------
pipenv run python -m Coloring.scripts.train --algo ppo --agents 3 --model 3-ppo-mixed-competitive `
    --setting mixed-motive-competitive `
    --max-steps 350 `
    --capture-interval 20

#----------------------------------------------------------------------------------------
# 3 PPO Mixed Competitive Settings with SHAREHOLDER Market
#----------------------------------------------------------------------------------------
pipenv run python -m Coloring.scripts.train --algo ppo --agents 3 --model 3-ppo-mixed-competitive-sm `
    --setting mixed-motive-competitive `
    --market sm `
    --max-steps 350 `
    --capture-interval 20

pipenv run python -m Coloring.scripts.train --algo ppo --agents 3 --model 3-ppo-mixed-competitive-sm-goal `
    --setting mixed-motive-competitive `
    --market sm-goal `
    --max-steps 350 `
    --capture-interval 20 

pipenv run python -m Coloring.scripts.train --algo ppo --agents 3 --model 3-ppo-mixed-competitive-sm-no-reset `
    --setting mixed-motive-competitive `
    --market sm-no-reset `
    --max-steps 350 `
    --capture-interval 20

pipenv run python -m Coloring.scripts.train --algo ppo --agents 3 --model 3-ppo-mixed-competitive-sm-no-debt `
    --setting mixed-motive-competitive `
    --market sm-no-debt `
    --max-steps 350 `
    --capture-interval 20

pipenv run python -m Coloring.scripts.train --algo ppo --agents 3 --model 3-ppo-mixed-competitive-sm-goal-no-reset-no-debt `
    --setting mixed-motive-competitive `
    --market sm-goal-no-reset-no-debt `
    --max-steps 350 `
    --capture-interval 20

# pipenv run python -m Coloring.scripts.train --algo ppo --agents 3 --model 3-ppo-mixed-competitive-sm-goal-no-reset `
#     --setting mixed-motive-competitive `
#     --market sm-goal-no-reset `
#     --max-steps 350 `
#     --capture-interval 20 

#----------------------------------------------------------------------------------------
# 3 PPO Settings with ACTION Market
#----------------------------------------------------------------------------------------
pipenv run python -m Coloring.scripts.train --algo ppo --agents 3 --model 3-ppo-mixed-competitive-am `
    --setting mixed-motive-competitive `
    --market am `
    --max-steps 350 `
    --capture-interval 20
    
pipenv run python -m Coloring.scripts.train --algo ppo --agents 3 --model 3-ppo-mixed-competitive-am-goal `
    --setting mixed-motive-competitive `
    --market am-goal `
    --max-steps 350 `
    --capture-interval 20
    
pipenv run python -m Coloring.scripts.train --algo ppo --agents 3 --model 3-ppo-mixed-competitive-am-no-reset `
    --setting mixed-motive-competitive `
    --market am-no-reset `
    --max-steps 350 `
    --capture-interval 20

pipenv run python -m Coloring.scripts.train --algo ppo --agents 3 --model 3-ppo-mixed-competitive-am-no-debt `
    --setting mixed-motive-competitive `
    --market am-no-debt `
    --max-steps 350 `
    --capture-interval 20

pipenv run python -m Coloring.scripts.train --algo ppo --agents 3 --model 3-ppo-mixed-competitive-am-goal-no-reset-no-debt `
    --setting mixed-motive-competitive `
    --market am-goal-no-reset-no-debt `
    --max-steps 350 `
    --capture-interval 20

# pipenv run python -m Coloring.scripts.train --algo ppo --agents 3 --model 3-ppo-mixed-competitive-am-goal-no-reset `
#     --setting mixed-motive-competitive `
#     --market am-goal-no-reset `
#     --max-steps 350 `
#     --capture-interval 20