
pipenv run python -m Coloring.scripts.train --algo ppo --agents 3 --model hard\\3-ppo-dr `
    --setting difference-reward `
    --grid-size 7 `
    --frames-per-proc 256 `
    --frames 250000 `
    --capture-interval 15

# #----------------------------------------------------------------------------------------
# # 3 PPO Settings
# #----------------------------------------------------------------------------------------
pipenv run python -m Coloring.scripts.train --algo ppo --agents 3 --model hard\\3-ppo `
    --grid-size 7 `
    --frames-per-proc 256 `
    --frames 250000 `
    --capture-interval 15

# #----------------------------------------------------------------------------------------
# # 3 PPO Settings with SHAREHOLDER Market
# #----------------------------------------------------------------------------------------
# pipenv run python -m Coloring.scripts.train --algo ppo --model hard\\2-ppo-sm `
#     --market sm `
#     --trading-fee 0.1

pipenv run python -m Coloring.scripts.train --algo ppo --agents 3 --model hard\\3-ppo-sm-goal `
    --market sm-goal `
    --trading-fee 0.1 `
    --grid-size 7 `
    --frames-per-proc 256 `
    --frames 250000 `
    --capture-interval 15

# pipenv run python -m Coloring.scripts.train --algo ppo --model hard\\2-ppo-sm-no-reset `
#     --market sm-no-reset `
#     --trading-fee 0.1

# pipenv run python -m Coloring.scripts.train --algo ppo --model hard\\2-ppo-sm-goal-no-reset `
#     --market sm-goal-no-reset `
#     --trading-fee 0.1

#---------
# pipenv run python -m Coloring.scripts.train --algo ppo --agents 2 --model hard\\2-ppo-sm-no-debt `
#     --market sm-no-debt `
#     --trading-fee 0.2 `
#     --max-steps 8  `
#     --grid-size 5 `
#     --frames-per-proc 128 `
#     --frames 80000

# pipenv run python -m Coloring.scripts.train --algo ppo --agents 2 --model hard\\2-ppo-sm-goal-no-debt `
#     --market sm-goal-no-debt `
#     --trading-fee 0.2 `
#     --max-steps 8  `
#     --grid-size 5 `
#     --frames-per-proc 128 `
#     --frames 80000

# pipenv run python -m Coloring.scripts.train --algo ppo --agents 2 --model hard\\2-ppo-sm-goal-no-reset-no-debt `
#     --market sm-goal-no-reset-no-debt `
#     --max-steps 15  `
#     --grid-size 5 `
#     --frames-per-proc 128 `
#     --frames 100000

# pipenv run python -m Coloring.scripts.train --algo ppo --agents 2 --model hard\\2-ppo-sm-no-reset-no-debt `
#     --market sm-no-reset-no-debt `
#     --max-steps 15  `
#     --grid-size 5 `
#     --frames-per-proc 128 `
#     --frames 100000

#----------------------------------------------------------------------------------------
# 3 PPO Settings with ACTION Market
#----------------------------------------------------------------------------------------
# pipenv run python -m Coloring.scripts.train --algo ppo --model hard\\2-ppo-am `
#     --market am `
#     --trading-fee 0.1

# pipenv run python -m Coloring.scripts.train --algo ppo --model hard\\2-ppo-am-goal `
#     --market am-goal `
#     --trading-fee 0.1
    
# pipenv run python -m Coloring.scripts.train --algo ppo --model hard\\2-ppo-am-no-reset `
#     --market am-no-reset `
#     --trading-fee 0.1

# pipenv run python -m Coloring.scripts.train --algo ppo --model hard\\2-ppo-am-no-debt `
#     --market am-no-debt `
#     --trading-fee 0.1

# pipenv run python -m Coloring.scripts.train --algo ppo --model hard\\2-ppo-am-goal-no-reset `
#     --market am-goal-no-reset `
#     --trading-fee 0.1

# pipenv run python -m Coloring.scripts.train --algo ppo --model hard\\2-ppo-am-goal-no-debt `
#     --market am-goal-no-debt `
#     --trading-fee 0.1

#----------
# pipenv run python -m Coloring.scripts.train --algo ppo --agents 2 --model hard\\2-ppo-am-goal-no-reset-no-debt `
#     --market am-goal-no-reset-no-debt `
#     --max-steps 15  `
#     --grid-size 5 `
#     --frames-per-proc 128 `
#     --frames 100000

# pipenv run python -m Coloring.scripts.train --algo ppo --agents 2 --model hard\\2-ppo-am-no-reset-no-debt `
#     --market am-no-reset-no-debt `
#     --max-steps 15  `
#     --grid-size 5 `
#     --frames-per-proc 128 `
#     --frames 100000