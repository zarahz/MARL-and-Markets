
python -m scripts.train --algo dqn --agents 2 --model easy/dqn/2-dqn-dr `
    --setting difference-reward `
    --max-steps 8 

#----------------------------------------------------------------------------------------
# 3 dqn Settings
#----------------------------------------------------------------------------------------
python -m scripts.train --algo dqn --agents 2 --model easy/dqn/2-dqn `
    --max-steps 8 

#----------------------------------------------------------------------------------------
# 3 dqn Settings with SHAREHOLDER Market
#----------------------------------------------------------------------------------------
python -m scripts.train --algo dqn --model easy/dqn/2-dqn-sm `
    --market sm `
    --max-steps 8 

python -m scripts.train --algo dqn --model easy/dqn/2-dqn-sm-goal `
    --market sm-goal `
    --max-steps 8 

python -m scripts.train --algo dqn --agents 2 --model easy/dqn/2-dqn-sm-no-reset `
    --market sm-no-reset `
    --max-steps 8 

python -m scripts.train --algo dqn --agents 2 --model easy/dqn/2-dqn-sm-goal-no-reset `
    --market sm-goal-no-reset `
    --max-steps 8 

#----------------------------------------------------------------------------------------
# 3 dqn Settings with ACTION Market
#----------------------------------------------------------------------------------------
python -m scripts.train --algo dqn --agents 2 --model easy/dqn/2-dqn-am `
    --market am `
    --max-steps 8 

python -m scripts.train --algo dqn --agents 2 --model easy/dqn/2-dqn-am-goal `
    --market am-goal `
    --max-steps 8 
    
python -m scripts.train --algo dqn --agents 2 --model easy/dqn/2-dqn-am-no-reset `
    --market am-no-reset `
    --max-steps 8 

python -m scripts.train --algo dqn --agents 2 --model easy/dqn/2-dqn-am-no-debt `
    --market am-no-debt `
    --max-steps 8 

python -m scripts.train --algo dqn --agents 2 --model easy/dqn/2-dqn-am-goal-no-reset `
    --market am-goal-no-reset `
    --max-steps 8 

python -m scripts.train --algo dqn --agents 2 --model easy/dqn/2-dqn-am-goal-no-debt `
    --market am-goal-no-debt `
    --max-steps 8 
