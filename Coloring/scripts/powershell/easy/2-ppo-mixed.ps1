#----------------------------------------------------------------------------------------
# 3 PPO Mixed Motive Settings
#----------------------------------------------------------------------------------------
python -m scripts.train --algo ppo --agents 2 --model easy/ppo/2-ppo-mixed `
    --setting mixed-motive `
    --max-steps 8

#----------------------------------------------------------------------------------------
# 3 PPO Mixed Motive Settings with SHAREHOLDER Market
#----------------------------------------------------------------------------------------
python -m scripts.train --algo ppo --model easy/ppo/2-ppo-mixed-sm `
    --setting mixed-motive `
    --market sm `
    --max-steps 8
    
python -m scripts.train --algo ppo --model easy/ppo/2-ppo-mixed-sm-goal `
    --setting mixed-motive `
    --market sm-goal `
    --max-steps 8

python -m scripts.train --algo ppo --model easy/ppo/2-ppo-mixed-sm-no-reset `
    --setting mixed-motive `
    --market sm-no-reset `
    --max-steps 8

python -m scripts.train --algo ppo --model easy/ppo/2-ppo-mixed-sm-goal-no-reset `
    --setting mixed-motive `
    --market sm-goal-no-reset `
    --max-steps 8

#----------------------------------------------------------------------------------------
# 3 PPO Mixed Motive Settings with ACTION Market
#----------------------------------------------------------------------------------------
python -m scripts.train --algo ppo --model easy/ppo/2-ppo-mixed-am `
    --setting mixed-motive `
    --market am `
    --max-steps 8
    
python -m scripts.train --algo ppo --model easy/ppo/2-ppo-mixed-am-goal `
    --setting mixed-motive `
    --market am-goal `
    --max-steps 8
    
python -m scripts.train --algo ppo --model easy/ppo/2-ppo-mixed-am-no-reset `
    --setting mixed-motive `
    --market am-no-reset `
    --max-steps 8

python -m scripts.train --algo ppo --model easy/ppo/2-ppo-mixed-am-no-debt `
    --setting mixed-motive `
    --market am-no-debt `
    --max-steps 8

python -m scripts.train --algo ppo --model easy/ppo/2-ppo-mixed-am-goal-no-reset `
    --setting mixed-motive `
    --market am-goal-no-reset `
    --max-steps 8

python -m scripts.train --algo ppo --model easy/ppo/2-ppo-mixed-am-goal-no-debt `
    --setting mixed-motive `
    --market am-goal-no-debt `
    --max-steps 8
