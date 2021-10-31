
python -m scripts.train --algo ppo --agents 2 --model easy/2-ppo-dr `
    --setting difference-reward `
    --max-steps 8

#----------------------------------------------------------------------------------------
# 3 PPO Settings
#----------------------------------------------------------------------------------------
python -m scripts.train --algo ppo --agents 2 --model easy/2-ppo `
    --max-steps 8

#----------------------------------------------------------------------------------------
# 3 PPO Settings with SHAREHOLDER Market
#----------------------------------------------------------------------------------------
python -m scripts.train --algo ppo --model easy/2-ppo-sm `
    --market sm `
    --trading-fee 0.1 `
    --max-steps 8

python -m scripts.train --algo ppo --model easy/2-ppo-sm-goal `
    --market sm-goal `
    --trading-fee 0.1 `
    --max-steps 8

python -m scripts.train --algo ppo --model easy/2-ppo-sm-no-reset `
    --market sm-no-reset `
    --trading-fee 0.1 `
    --max-steps 8

python -m scripts.train --algo ppo --model easy/2-ppo-sm-goal-no-reset `
    --market sm-goal-no-reset `
    --trading-fee 0.1 `
    --max-steps 8

# python -m scripts.train --algo ppo --agents 2 --model easy/2-ppo-sm-no-debt `
#     --market sm-no-debt `
#     --trading-fee 0.2 `
#     --max-steps 8  `
#     --grid-size 5 `
#     --frames-per-proc 128 `
#     --frames 80000

# python -m scripts.train --algo ppo --agents 2 --model easy/2-ppo-sm-goal-no-debt `
#     --market sm-goal-no-debt `
#     --trading-fee 0.2 `
#     --max-steps 8  `
#     --grid-size 5 `
#     --frames-per-proc 128 `
#     --frames 80000

# python -m scripts.train --algo ppo --agents 2 --model easy/2-ppo-sm-goal-no-reset-no-debt `
#     --market sm-goal-no-reset-no-debt `
#     --max-steps 15  `
#     --grid-size 5 `
#     --frames-per-proc 128 `
#     --frames 100000

# python -m scripts.train --algo ppo --agents 2 --model easy/2-ppo-sm-no-reset-no-debt `
#     --market sm-no-reset-no-debt `
#     --max-steps 15  `
#     --grid-size 5 `
#     --frames-per-proc 128 `
#     --frames 100000

#----------------------------------------------------------------------------------------
# 3 PPO Settings with ACTION Market
#----------------------------------------------------------------------------------------
python -m scripts.train --algo ppo --model easy/2-ppo-am `
    --market am `
    --trading-fee 0.1 `
    --max-steps 8

python -m scripts.train --algo ppo --model easy/2-ppo-am-goal `
    --market am-goal `
    --trading-fee 0.1 `
    --max-steps 8
    
python -m scripts.train --algo ppo --model easy/2-ppo-am-no-reset `
    --market am-no-reset `
    --trading-fee 0.1 `
    --max-steps 8

python -m scripts.train --algo ppo --model easy/2-ppo-am-no-debt `
    --market am-no-debt `
    --trading-fee 0.1 `
    --max-steps 8

python -m scripts.train --algo ppo --model easy/2-ppo-am-goal-no-reset `
    --market am-goal-no-reset `
    --trading-fee 0.1 `
    --max-steps 8

python -m scripts.train --algo ppo --model easy/2-ppo-am-goal-no-debt `
    --market am-goal-no-debt `
    --trading-fee 0.1 `
    --max-steps 8

# python -m scripts.train --algo ppo --agents 2 --model easy/2-ppo-am-goal-no-reset-no-debt `
#     --market am-goal-no-reset-no-debt `
#     --max-steps 15  `
#     --grid-size 5 `
#     --frames-per-proc 128 `
#     --frames 100000

# python -m scripts.train --algo ppo --agents 2 --model easy/2-ppo-am-no-reset-no-debt `
#     --market am-no-reset-no-debt `
#     --max-steps 15  `
#     --grid-size 5 `
#     --frames-per-proc 128 `
#     --frames 100000