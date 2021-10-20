
# pipenv run python -m Coloring.scripts.train --algo ppo --agents 2 --model easy\\2-ppo-dr `
#     --setting difference-reward `
#     --max-steps 70  `
#     --grid-size 5 `
#     --frames-per-proc 128 `
#     --frames 100000

#----------------------------------------------------------------------------------------
# 3 PPO Settings
#----------------------------------------------------------------------------------------
# pipenv run python -m Coloring.scripts.train --algo ppo --agents 2 --model easy\\2-ppo `
#     --max-steps 70  `
#     --grid-size 5 `
#     --frames-per-proc 128 `
#     --frames 100000 `

#----------------------------------------------------------------------------------------
# 3 PPO Settings with SHAREHOLDER Market
#----------------------------------------------------------------------------------------
# pipenv run python -m Coloring.scripts.train --algo ppo --agents 2 --model easy\\2-ppo-sm `
#     --market sm `
#     --max-steps 70  `
#     --grid-size 5 `
#     --frames-per-proc 128 `
#     --frames 100000 `

# pipenv run python -m Coloring.scripts.train --algo ppo --agents 2 --model easy\\2-ppo-sm-goal `
#     --market sm-goal `
#     --max-steps 70  `
#     --grid-size 5 `
#     --frames-per-proc 128 `
#     --frames 100000

# pipenv run python -m Coloring.scripts.train --algo ppo --agents 2 --model easy\\2-ppo-sm-no-reset `
#     --market sm-no-reset `
#     --max-steps 70  `
#     --grid-size 5 `
#     --frames-per-proc 128 `
#     --frames 100000

# pipenv run python -m Coloring.scripts.train --algo ppo --agents 2 --model easy\\2-ppo-sm-no-debt `
#     --market sm-no-debt `
#     --max-steps 70  `
#     --grid-size 5 `
#     --frames-per-proc 128 `
#     --frames 100000

pipenv run python -m Coloring.scripts.train --algo ppo --agents 2 --model easy\\2-ppo-sm-goal-no-reset `
    --market sm-goal-no-reset `
    --max-steps 70  `
    --grid-size 5 `
    --frames-per-proc 128 `
    --frames 100000

pipenv run python -m Coloring.scripts.train --algo ppo --agents 2 --model easy\\2-ppo-sm-goal-no-debt `
    --market sm-goal-no-debt `
    --max-steps 70  `
    --grid-size 5 `
    --frames-per-proc 128 `
    --frames 100000

# pipenv run python -m Coloring.scripts.train --algo ppo --agents 2 --model easy\\2-ppo-sm-goal-no-reset-no-debt `
#     --market sm-goal-no-reset-no-debt `
#     --max-steps 70  `
#     --grid-size 5 `
#     --frames-per-proc 128 `
#     --frames 100000

# pipenv run python -m Coloring.scripts.train --algo ppo --agents 2 --model easy\\2-ppo-sm-no-reset-no-debt `
#     --market sm-no-reset-no-debt `
#     --max-steps 70  `
#     --grid-size 5 `
#     --frames-per-proc 128 `
#     --frames 100000

#----------------------------------------------------------------------------------------
# 3 PPO Settings with ACTION Market
#----------------------------------------------------------------------------------------
pipenv run python -m Coloring.scripts.train --algo ppo --agents 2 --model easy\\2-ppo-am `
    --market am `
    --max-steps 70  `
    --grid-size 5 `
    --frames-per-proc 128 `
    --frames 100000 

pipenv run python -m Coloring.scripts.train --algo ppo --agents 2 --model easy\\2-ppo-am-goal `
    --market am-goal `
    --max-steps 70  `
    --grid-size 5 `
    --frames-per-proc 128 `
    --frames 100000
    
pipenv run python -m Coloring.scripts.train --algo ppo --agents 2 --model easy\\2-ppo-am-no-reset `
    --market am-no-reset `
    --max-steps 70  `
    --grid-size 5 `
    --frames-per-proc 128 `
    --frames 100000

pipenv run python -m Coloring.scripts.train --algo ppo --agents 2 --model easy\\2-ppo-am-no-debt `
    --market am-no-debt `
    --max-steps 70  `
    --grid-size 5 `
    --frames-per-proc 128 `
    --frames 100000

pipenv run python -m Coloring.scripts.train --algo ppo --agents 2 --model easy\\2-ppo-am-goal-no-reset `
    --market am-goal-no-reset `
    --max-steps 70  `
    --grid-size 5 `
    --frames-per-proc 128 `
    --frames 100000

pipenv run python -m Coloring.scripts.train --algo ppo --agents 2 --model easy\\2-ppo-am-goal-no-debt `
    --market am-goal-no-debt `
    --max-steps 70  `
    --grid-size 5 `
    --frames-per-proc 128 `
    --frames 100000

# pipenv run python -m Coloring.scripts.train --algo ppo --agents 2 --model easy\\2-ppo-am-goal-no-reset-no-debt `
#     --market am-goal-no-reset-no-debt `
#     --max-steps 70  `
#     --grid-size 5 `
#     --frames-per-proc 128 `
#     --frames 100000

# pipenv run python -m Coloring.scripts.train --algo ppo --agents 2 --model easy\\2-ppo-am-no-reset-no-debt `
#     --market am-no-reset-no-debt `
#     --max-steps 70  `
#     --grid-size 5 `
#     --frames-per-proc 128 `
#     --frames 100000