
python -m scripts.train --algo ppo --agents 2 --model easy/ppo/2-ppo-dr `
    --setting difference-reward `
    --batch-size 64 `
    --max-steps 8

#----------------------------------------------------------------------------------------
# 3 PPO Settings
#----------------------------------------------------------------------------------------
python -m scripts.train --algo ppo --agents 2 --model easy/ppo/2-ppo `
    --batch-size 64 `
    --max-steps 8

#----------------------------------------------------------------------------------------
# 3 PPO Settings with SHAREHOLDER Market
#----------------------------------------------------------------------------------------
python -m scripts.train --algo ppo --model easy/ppo/2-ppo-sm `
    --market sm `
    --trading-fee 0.1 `
    --batch-size 64 `
    --max-steps 8

python -m scripts.train --algo ppo --model easy/ppo/2-ppo-sm-goal `
    --market sm-goal `
    --trading-fee 0.1 `
    --batch-size 64 `
    --max-steps 8

python -m scripts.train --algo ppo --model easy/ppo/2-ppo-sm-no-reset `
    --market sm-no-reset `
    --trading-fee 0.1 `
    --batch-size 64 `
    --max-steps 8

python -m scripts.train --algo ppo --model easy/ppo/2-ppo-sm-goal-no-reset `
    --market sm-goal-no-reset `
    --trading-fee 0.1 `
    --batch-size 64 `
    --max-steps 8

#----------------------------------------------------------------------------------------
# 3 PPO Settings with ACTION Market
#----------------------------------------------------------------------------------------
python -m scripts.train --algo ppo --model easy/ppo/2-ppo-am `
    --market am `
    --trading-fee 0.1 `
    --batch-size 64 `
    --max-steps 8

python -m scripts.train --algo ppo --model easy/ppo/2-ppo-am-goal `
    --market am-goal `
    --trading-fee 0.1 `
    --batch-size 64 `
    --max-steps 8
    
python -m scripts.train --algo ppo --model easy/ppo/2-ppo-am-no-reset `
    --market am-no-reset `
    --trading-fee 0.1 `
    --batch-size 64 `
    --max-steps 8

python -m scripts.train --algo ppo --model easy/ppo/2-ppo-am-no-debt `
    --market am-no-debt `
    --trading-fee 0.1 `
    --batch-size 64 `
    --max-steps 8

python -m scripts.train --algo ppo --model easy/ppo/2-ppo-am-goal-no-reset `
    --market am-goal-no-reset `
    --trading-fee 0.1 `
    --batch-size 64 `
    --max-steps 8

python -m scripts.train --algo ppo --model easy/ppo/2-ppo-am-goal-no-debt `
    --market am-goal-no-debt `
    --trading-fee 0.1 `
    --batch-size 64 `
    --max-steps 8