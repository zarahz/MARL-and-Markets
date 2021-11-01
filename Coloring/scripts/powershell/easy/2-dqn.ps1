
# python -m scripts.train --algo dqn --agents 2 --model easy/dqn/2-dqn-dr `
#     --setting difference-reward `
#     --max-steps 8 `
#     --batch-size 64

# #----------------------------------------------------------------------------------------
# # 3 dqn Settings
# #----------------------------------------------------------------------------------------
# python -m scripts.train --algo dqn --agents 2 --model easy/dqn/2-dqn `
#     --max-steps 8 `
#     --batch-size 64

# #----------------------------------------------------------------------------------------
# # 3 dqn Settings with SHAREHOLDER Market
# #----------------------------------------------------------------------------------------
# python -m scripts.train --algo dqn --model easy/dqn/2-dqn-sm `
#     --market sm `
#     --trading-fee 0.1 `
#     --max-steps 8 `
#     --batch-size 64

# python -m scripts.train --algo dqn --model easy/dqn/2-dqn-sm-goal `
#     --market sm-goal `
#     --trading-fee 0.1 `
#     --max-steps 8 `
#     --batch-size 64

python -m scripts.train --algo dqn --agents 2 --model easy/dqn/2-dqn-sm-no-reset `
    --market sm-no-reset `
    --trading-fee 0.1 `
    --max-steps 8 `
    --batch-size 64

python -m scripts.train --algo dqn --agents 2 --model easy/dqn/2-dqn-sm-goal-no-reset `
    --market sm-goal-no-reset `
    --trading-fee 0.1 `
    --max-steps 8 `
    --batch-size 64

#----------------------------------------------------------------------------------------
# 3 dqn Settings with ACTION Market
#----------------------------------------------------------------------------------------
# python -m scripts.train --algo dqn --agents 2 --model easy/dqn/2-dqn-am `
#     --market am `
#     --trading-fee 0.1 `
#     --max-steps 8 `
#     --batch-size 64

# python -m scripts.train --algo dqn --agents 2 --model easy/dqn/2-dqn-am-goal `
#     --market am-goal `
#     --trading-fee 0.1 `
#     --max-steps 8 `
#     --batch-size 64
    
python -m scripts.train --algo dqn --agents 2 --model easy/dqn/2-dqn-am-no-reset `
    --market am-no-reset `
    --trading-fee 0.1 `
    --max-steps 8 `
    --batch-size 64

python -m scripts.train --algo dqn --agents 2 --model easy/dqn/2-dqn-am-no-debt `
    --market am-no-debt `
    --trading-fee 0.1 `
    --max-steps 8 `
    --batch-size 64

python -m scripts.train --algo dqn --agents 2 --model easy/dqn/2-dqn-am-goal-no-reset `
    --market am-goal-no-reset `
    --trading-fee 0.1 `
    --max-steps 8 `
    --batch-size 64

python -m scripts.train --algo dqn --agents 2 --model easy/dqn/2-dqn-am-goal-no-debt `
    --market am-goal-no-debt `
    --trading-fee 0.1 `
    --max-steps 8 `
    --batch-size 64
