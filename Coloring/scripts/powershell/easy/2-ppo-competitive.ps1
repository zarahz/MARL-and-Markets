#----------------------------------------------------------------------------------------
# 3 PPO Mixed Competitive Settings
#----------------------------------------------------------------------------------------
python -m scripts.train --algo ppo --agents 2 --model easy/ppo/2-ppo-competitive `
    --setting mixed-motive-competitive `
    --max-steps 8

#----------------------------------------------------------------------------------------
# 3 PPO Mixed Competitive Settings with SHAREHOLDER Market
#----------------------------------------------------------------------------------------
python -m scripts.train --algo ppo --model easy/ppo/2-ppo-competitive-sm `
    --setting mixed-motive-competitive `
    --market sm `
    --max-steps 8

python -m scripts.train --algo ppo --model easy/ppo/2-ppo-competitive-sm-goal `
    --setting mixed-motive-competitive `
    --market sm-goal `
    --max-steps 8

python -m scripts.train --algo ppo --model easy/ppo/2-ppo-competitive-sm-no-reset `
    --setting mixed-motive-competitive `
    --market sm-no-reset `
    --max-steps 8

python -m scripts.train --algo ppo --model easy/ppo/2-ppo-competitive-sm-goal-no-reset `
    --setting mixed-motive-competitive `
    --market sm-goal-no-reset `
    --max-steps 8

#----------------------------------------------------------------------------------------
# 3 PPO Settings with ACTION Market
#----------------------------------------------------------------------------------------
python -m scripts.train --algo ppo --model easy/ppo/2-ppo-competitive-am `
    --setting mixed-motive-competitive `
    --market am `
    --max-steps 8
    
python -m scripts.train --algo ppo --model easy/ppo/2-ppo-competitive-am-goal `
    --setting mixed-motive-competitive `
    --market am-goal `
    --max-steps 8

python -m scripts.train --algo ppo --model easy/ppo/2-ppo-competitive-am-no-debt `
    --setting mixed-motive-competitive `
    --market am-no-debt `
    --max-steps 8

python -m scripts.train --algo ppo --model easy/ppo/2-ppo-competitive-am-goal-no-debt `
    --setting mixed-motive-competitive `
    --market am-goal-no-debt `
    --max-steps 8

python -m scripts.train --algo ppo --model easy/ppo/2-ppo-competitive-am-no-reset `
    --setting mixed-motive-competitive `
    --market am-no-reset `
    --max-steps 8

python -m scripts.train --algo ppo --model easy/ppo/2-ppo-competitive-am-goal-no-reset `
    --setting mixed-motive-competitive `
    --market am-goal-no-reset `
    --max-steps 8
