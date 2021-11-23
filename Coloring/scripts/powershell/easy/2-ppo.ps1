
python -m scripts.train --algo ppo --agents 2 --model easy/ppo/2-ppo-dr `
    --setting difference-reward `
    --max-steps 8

#----------------------------------------------------------------------------------------
# 3 PPO Settings
#----------------------------------------------------------------------------------------
python -m scripts.train --algo ppo --agents 2 --model easy/ppo/2-ppo `
    --max-steps 8

#----------------------------------------------------------------------------------------
# 3 PPO Settings with SHAREHOLDER Market
#----------------------------------------------------------------------------------------
python -m scripts.train --algo ppo --model easy/ppo/2-ppo-sm `
    --market sm `
    --max-steps 8

python -m scripts.train --algo ppo --model easy/ppo/2-ppo-sm-goal `
    --market sm-goal `
    --max-steps 8

python -m scripts.train --algo ppo --model easy/ppo/2-ppo-sm-no-reset `
    --market sm-no-reset `
    --max-steps 8

python -m scripts.train --algo ppo --model easy/ppo/2-ppo-sm-goal-no-reset `
    --market sm-goal-no-reset `
    --max-steps 8

#----------------------------------------------------------------------------------------
# 3 PPO Settings with ACTION Market
#----------------------------------------------------------------------------------------
python -m scripts.train --algo ppo --model easy/ppo/2-ppo-am `
    --market am `
    --max-steps 8

python -m scripts.train --algo ppo --model easy/ppo/2-ppo-am-goal `
    --market am-goal `
    --max-steps 8
    
python -m scripts.train --algo ppo --model easy/ppo/2-ppo-am-no-reset `
    --market am-no-reset `
    --max-steps 8

python -m scripts.train --algo ppo --model easy/ppo/2-ppo-am-no-debt `
    --market am-no-debt `
    --max-steps 8

python -m scripts.train --algo ppo --model easy/ppo/2-ppo-am-goal-no-reset `
    --market am-goal-no-reset `
    --max-steps 8

python -m scripts.train --algo ppo --model easy/ppo/2-ppo-am-goal-no-debt `
    --market am-goal-no-debt `
    --max-steps 8