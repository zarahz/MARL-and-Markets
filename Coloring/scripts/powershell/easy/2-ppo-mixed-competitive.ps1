# #----------------------------------------------------------------------------------------
# # 3 PPO Mixed Competitive Settings
# #----------------------------------------------------------------------------------------
# pipenv run python -m Coloring.scripts.train --algo ppo --agents 2 --model easy\\2-ppo-mixed-competitive `
#     --setting mixed-motive-competitive `
#     --max-steps 8  `
#     --grid-size 5 `
#     --frames-per-proc 128 `
#     --frames 80000

# #----------------------------------------------------------------------------------------
# # 3 PPO Mixed Competitive Settings with SHAREHOLDER Market
# #----------------------------------------------------------------------------------------
pipenv run python -m Coloring.scripts.train --algo ppo --model easy\\2-ppo-mixed-competitive-sm `
    --setting mixed-motive-competitive `
    --market sm `
    --trading-fee 0.05

pipenv run python -m Coloring.scripts.train --algo ppo --model easy\\2-ppo-mixed-competitive-sm-goal `
    --setting mixed-motive-competitive `
    --market sm-goal `
    --trading-fee 0.05

#----------------------------------------------------------------------------------------
# 3 PPO Settings with ACTION Market
#----------------------------------------------------------------------------------------
pipenv run python -m Coloring.scripts.train --algo ppo --model easy\\2-ppo-mixed-competitive-am `
    --setting mixed-motive-competitive `
    --market am `
    --trading-fee 0.05
    
pipenv run python -m Coloring.scripts.train --algo ppo --model easy\\2-ppo-mixed-competitive-am-goal `
    --setting mixed-motive-competitive `
    --market am-goal `
    --trading-fee 0.05

pipenv run python -m Coloring.scripts.train --algo ppo --model easy\\2-ppo-mixed-competitive-am-no-debt `
    --setting mixed-motive-competitive `
    --market am-no-debt `
    --trading-fee 0.05

pipenv run python -m Coloring.scripts.train --algo ppo --model easy\\2-ppo-mixed-competitive-am-goal-no-debt `
    --setting mixed-motive-competitive `
    --market am-goal-no-debt `
    --trading-fee 0.05

#----------------------------------------------------------------------------------------
# Optional settings
#----------------------------------------------------------------------------------------

# competitive mode does not enable reset of fields

# SHAREHOLDER MARKET
# pipenv run python -m Coloring.scripts.train --algo ppo --agents 2 --model easy\\2-ppo-mixed-competitive-sm-no-reset `
#     --setting mixed-motive-competitive `
#     --market sm-no-reset `
#     --trading-fee 0.2 `
#     --max-steps 8  `
#     --grid-size 5 `
#     --frames-per-proc 128 `
#     --frames 80000
# pipenv run python -m Coloring.scripts.train --algo ppo --agents 2 --model easy\\2-ppo-mixed-competitive-sm-goal-no-reset `
#     --setting mixed-motive-competitive `
#     --market sm-goal-no-reset `
#     --trading-fee 0.2 `
#     --max-steps 8  `
#     --grid-size 5 `
#     --frames-per-proc 128 `
#     --frames 80000

# ACTION MARKET
# pipenv run python -m Coloring.scripts.train --algo ppo --agents 2 --model easy\\2-ppo-mixed-competitive-am-goal-no-reset `
#     --setting mixed-motive-competitive `
#     --market am-goal-no-reset `
#     --trading-fee 0.2 `
#     --max-steps 8  `
#     --grid-size 5 `
#     --frames-per-proc 128 `
#     --frames 80000
# pipenv run python -m Coloring.scripts.train --algo ppo --agents 2 --model easy\\2-ppo-mixed-competitive-am-no-reset `
#     --setting mixed-motive-competitive `
#     --market am-no-reset `
#     --trading-fee 0.2 `
#     --max-steps 8  `
#     --grid-size 5 `
#     --frames-per-proc 128 `
#     --frames 80000

# shares do not have prices in this implementation - no debt possible

# SHAREHOLDER MARKET
# pipenv run python -m Coloring.scripts.train --algo ppo --agents 2 --model easy\\2-ppo-mixed-competitive-sm-no-debt `
#     --setting mixed-motive-competitive `
#     --market sm-no-debt `
#     --trading-fee 0.2 `
#     --max-steps 8  `
#     --grid-size 5 `
#     --frames-per-proc 128 `
#     --frames 80000
# pipenv run python -m Coloring.scripts.train --algo ppo --agents 2 --model easy\\2-ppo-mixed-competitive-sm-goal-no-debt `
#     --setting mixed-motive-competitive `
#     --market sm-goal-no-debt `
#     --trading-fee 0.2 `
#     --max-steps 8  `
#     --grid-size 5 `
#     --frames-per-proc 128 `
#     --frames 80000
# pipenv run python -m Coloring.scripts.train --algo ppo --agents 2 --model easy\\2-ppo-mixed-competitive-sm-goal-no-reset-no-debt `
#     --setting mixed-motive-competitive `
#     --market sm-goal-no-reset-no-debt `
#     --max-steps 15  `
#     --grid-size 5 `
#     --frames-per-proc 128 `
#     --frames 100000
# pipenv run python -m Coloring.scripts.train --algo ppo --agents 2 --model easy\\2-ppo-mixed-competitive-sm-no-reset-no-debt `
#     --setting mixed-motive-competitive `
#     --market sm-no-reset-no-debt `
#     --max-steps 15  `
#     --grid-size 5 `
#     --frames-per-proc 128 `
#     --frames 100000

# ACTION MARKET
# pipenv run python -m Coloring.scripts.train --algo ppo --agents 2 --model easy\\2-ppo-mixed-competitive-am-goal-no-reset-no-debt `
#     --setting mixed-motive-competitive `
#     --market am-goal-no-reset-no-debt `
#     --max-steps 15  `
#     --grid-size 5 `
#     --frames-per-proc 128 `
#     --frames 100000
# pipenv run python -m Coloring.scripts.train --algo ppo --agents 2 --model easy\\2-ppo-mixed-competitive-am-no-reset-no-debt `
#     --setting mixed-motive-competitive `
#     --market am-no-reset-no-debt `
#     --max-steps 15  `
#     --grid-size 5 `
#     --frames-per-proc 128 `
#     --frames 100000