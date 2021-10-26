# #----------------------------------------------------------------------------------------
# # 3 PPO Mixed Motive Settings
# #----------------------------------------------------------------------------------------
# pipenv run python -m Coloring.scripts.train --algo ppo --agents 2 --model easy\\2-ppo-mixed `
#     --setting mixed-motive `
#     --max-steps 8  `
#     --grid-size 5 `
#     --frames-per-proc 128 `
#     --frames 80000

# #----------------------------------------------------------------------------------------
# # 3 PPO Mixed Motive Settings with SHAREHOLDER Market
# #----------------------------------------------------------------------------------------
pipenv run python -m Coloring.scripts.train --algo ppo --model easy\\2-ppo-mixed-sm `
    --setting mixed-motive `
    --market sm `
    --trading-fee 0.05
    
pipenv run python -m Coloring.scripts.train --algo ppo --model easy\\2-ppo-mixed-sm-goal `
    --setting mixed-motive `
    --market sm-goal `
    --trading-fee 0.05

pipenv run python -m Coloring.scripts.train --algo ppo --model easy\\2-ppo-mixed-sm-no-reset `
    --setting mixed-motive `
    --market sm-no-reset `
    --trading-fee 0.05

pipenv run python -m Coloring.scripts.train --algo ppo --model easy\\2-ppo-mixed-sm-goal-no-reset `
    --setting mixed-motive `
    --market sm-goal-no-reset `
    --trading-fee 0.05

# pipenv run python -m Coloring.scripts.train --algo ppo --agents 2 --model easy\\2-ppo-mixed-sm-no-debt `
#     --setting mixed-motive `
#     --market sm-no-debt `
#     --trading-fee 0.2 `
#     --max-steps 8  `
#     --grid-size 5 `
#     --frames-per-proc 128 `
#     --frames 80000

# pipenv run python -m Coloring.scripts.train --algo ppo --agents 2 --model easy\\2-ppo-mixed-sm-goal-no-debt `
#     --setting mixed-motive `
#     --market sm-goal-no-debt `
#     --trading-fee 0.2 `
#     --max-steps 8  `
#     --grid-size 5 `
#     --frames-per-proc 128 `
#     --frames 80000

# pipenv run python -m Coloring.scripts.train --algo ppo --agents 2 --model easy\\2-ppo-mixed-sm-goal-no-reset-no-debt `
#     --setting mixed-motive `
#     --market sm-goal-no-reset-no-debt `
#     --max-steps 15  `
#     --grid-size 5 `
#     --frames-per-proc 128 `
#     --frames 100000

# pipenv run python -m Coloring.scripts.train --algo ppo --agents 2 --model easy\\2-ppo-mixed-sm-no-reset-no-debt `
#     --setting mixed-motive `
#     --market sm-no-reset-no-debt `
#     --max-steps 15  `
#     --grid-size 5 `
#     --frames-per-proc 128 `
#     --frames 100000

#----------------------------------------------------------------------------------------
# 3 PPO Mixed Motive Settings with ACTION Market
#----------------------------------------------------------------------------------------
pipenv run python -m Coloring.scripts.train --algo ppo --model easy\\2-ppo-mixed-am `
    --setting mixed-motive `
    --market am `
    --trading-fee 0.05
    
pipenv run python -m Coloring.scripts.train --algo ppo --model easy\\2-ppo-mixed-am-goal `
    --setting mixed-motive `
    --market am-goal `
    --trading-fee 0.05
    
pipenv run python -m Coloring.scripts.train --algo ppo --model easy\\2-ppo-mixed-am-no-reset `
    --setting mixed-motive `
    --market am-no-reset `
    --trading-fee 0.05

pipenv run python -m Coloring.scripts.train --algo ppo --model easy\\2-ppo-mixed-am-no-debt `
    --setting mixed-motive `
    --market am-no-debt `
    --trading-fee 0.05

pipenv run python -m Coloring.scripts.train --algo ppo --model easy\\2-ppo-mixed-am-goal-no-reset `
    --setting mixed-motive `
    --market am-goal-no-reset `
    --trading-fee 0.05

pipenv run python -m Coloring.scripts.train --algo ppo --model easy\\2-ppo-mixed-am-goal-no-debt `
    --setting mixed-motive `
    --market am-goal-no-debt `
    --trading-fee 0.05

# pipenv run python -m Coloring.scripts.train --algo ppo --agents 2 --model easy\\2-ppo-mixed-am-goal-no-reset-no-debt `
#     --setting mixed-motive `
#     --market am-goal-no-reset-no-debt `
#     --max-steps 15  `
#     --grid-size 5 `
#     --frames-per-proc 128 `
#     --frames 100000

# pipenv run python -m Coloring.scripts.train --algo ppo --agents 2 --model easy\\2-ppo-mixed-am-no-reset-no-debt `
#     --setting mixed-motive `
#     --market am-no-reset-no-debt `
#     --max-steps 15  `
#     --grid-size 5 `
#     --frames-per-proc 128 `
#     --frames 100000