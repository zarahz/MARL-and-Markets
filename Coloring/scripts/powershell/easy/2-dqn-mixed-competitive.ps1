# #----------------------------------------------------------------------------------------
# # 3 dqn Mixed Competitive Settings
# #----------------------------------------------------------------------------------------
# python -m scripts.train --algo dqn --agents 2 --model easy/dqn/2-dqn-mixed-competitive `
#     --setting mixed-motive-competitive `
#     --max-steps 8 `
#     --max-steps 8 `
#     --batch-size 64

# #----------------------------------------------------------------------------------------
# # 3 dqn Mixed Competitive Settings with SHAREHOLDER Market
# #----------------------------------------------------------------------------------------
# python -m scripts.train --algo dqn --agents 2 --model easy/dqn/2-dqn-mixed-competitive-sm `
#     --setting mixed-motive-competitive `
#     --market sm `
#     --trading-fee 0.1 `
#     --max-steps 8 `
#     --batch-size 64

# python -m scripts.train --algo dqn --agents 2 --model easy/dqn/2-dqn-mixed-competitive-sm-goal `
#     --setting mixed-motive-competitive `
#     --market sm-goal `
#     --trading-fee 0.1 `
#     --max-steps 8 `
#     --batch-size 64

python -m scripts.train --algo dqn --agents 2 --model easy/dqn/2-dqn-mixed-competitive-sm-no-reset `
    --setting mixed-motive-competitive `
    --market sm-no-reset `
    --trading-fee 0.1 `
    --max-steps 8 `
    --batch-size 64

python -m scripts.train --algo dqn --agents 2 --model easy/dqn/2-dqn-mixed-competitive-sm-goal-no-reset `
    --setting mixed-motive-competitive `
    --market sm-goal-no-reset `
    --trading-fee 0.1 `
    --max-steps 8 `
    --batch-size 64

#----------------------------------------------------------------------------------------
# 3 dqn Settings with ACTION Market
#----------------------------------------------------------------------------------------
# python -m scripts.train --algo dqn --agents 2 --model easy/dqn/2-dqn-mixed-competitive-am `
#     --setting mixed-motive-competitive `
#     --market am `
#     --trading-fee 0.1 `
#     --max-steps 8 `
#     --batch-size 64
    
# python -m scripts.train --algo dqn --agents 2 --model easy/dqn/2-dqn-mixed-competitive-am-goal `
#     --setting mixed-motive-competitive `
#     --market am-goal `
#     --trading-fee 0.1 `
#     --max-steps 8 `
#     --batch-size 64
    
python -m scripts.train --algo dqn --agents 2 --model easy/dqn/2-dqn-mixed-competitive-am-no-reset `
    --setting mixed-motive-competitive `
    --market am-no-reset `
    --trading-fee 0.1 `
    --max-steps 8 `
    --batch-size 64

python -m scripts.train --algo dqn --agents 2 --model easy/dqn/2-dqn-mixed-competitive-am-no-debt `
    --setting mixed-motive-competitive `
    --market am-no-debt `
    --trading-fee 0.1 `
    --max-steps 8 `
    --batch-size 64

python -m scripts.train --algo dqn --agents 2 --model easy/dqn/2-dqn-mixed-competitive-am-goal-no-reset `
    --setting mixed-motive-competitive `
    --market am-goal-no-reset `
    --trading-fee 0.1 `
    --max-steps 8 `
    --batch-size 64

python -m scripts.train --algo dqn --agents 2 --model easy/dqn/2-dqn-mixed-competitive-am-goal-no-debt `
    --setting mixed-motive-competitive `
    --market am-goal-no-debt `
    --trading-fee 0.1 `
    --max-steps 8 `
    --batch-size 64