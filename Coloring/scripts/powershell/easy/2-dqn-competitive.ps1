#----------------------------------------------------------------------------------------
# 3 dqn Mixed Competitive Settings
#----------------------------------------------------------------------------------------
python -m scripts.train --algo dqn --agents 2 --model easy/dqn/2-dqn-competitive `
    --setting mixed-motive-competitive `
    --max-steps 8

#----------------------------------------------------------------------------------------
# 3 dqn Mixed Competitive Settings with SHAREHOLDER Market
#----------------------------------------------------------------------------------------
python -m scripts.train --algo dqn --agents 2 --model easy/dqn/2-dqn-competitive-sm `
    --setting mixed-motive-competitive `
    --market sm `
    --max-steps 8

python -m scripts.train --algo dqn --agents 2 --model easy/dqn/2-dqn-competitive-sm-goal `
    --setting mixed-motive-competitive `
    --market sm-goal `
    --max-steps 8 

python -m scripts.train --algo dqn --agents 2 --model easy/dqn/2-dqn-competitive-sm-no-reset `
    --setting mixed-motive-competitive `
    --market sm-no-reset `
    --max-steps 8 

python -m scripts.train --algo dqn --agents 2 --model easy/dqn/2-dqn-competitive-sm-goal-no-reset `
    --setting mixed-motive-competitive `
    --market sm-goal-no-reset `
    --max-steps 8

#----------------------------------------------------------------------------------------
# 3 dqn Settings with ACTION Market
#----------------------------------------------------------------------------------------
python -m scripts.train --algo dqn --agents 2 --model easy/dqn/2-dqn-competitive-am `
    --setting mixed-motive-competitive `
    --market am `
    --max-steps 8 
    
python -m scripts.train --algo dqn --agents 2 --model easy/dqn/2-dqn-competitive-am-goal `
    --setting mixed-motive-competitive `
    --market am-goal `
    --max-steps 8 
    
python -m scripts.train --algo dqn --agents 2 --model easy/dqn/2-dqn-competitive-am-no-reset `
    --setting mixed-motive-competitive `
    --market am-no-reset `
    --max-steps 8 

python -m scripts.train --algo dqn --agents 2 --model easy/dqn/2-dqn-competitive-am-no-debt `
    --setting mixed-motive-competitive `
    --market am-no-debt `
    --max-steps 8

python -m scripts.train --algo dqn --agents 2 --model easy/dqn/2-dqn-competitive-am-goal-no-reset `
    --setting mixed-motive-competitive `
    --market am-goal-no-reset `
    --max-steps 8 

python -m scripts.train --algo dqn --agents 2 --model easy/dqn/2-dqn-competitive-am-goal-no-debt `
    --setting mixed-motive-competitive `
    --market am-goal-no-debt `
    --max-steps 8 