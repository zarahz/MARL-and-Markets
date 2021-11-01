# #----------------------------------------------------------------------------------------
# # 3 dqn Mixed Motive Settings
# #----------------------------------------------------------------------------------------
# python -m scripts.train --algo dqn --agents 2 --model easy/dqn/2-dqn-mixed `
#     --setting mixed-motive `
#     --max-steps 8 `
#     --batch-size 64

# #----------------------------------------------------------------------------------------
# # 3 dqn Mixed Motive Settings with SHAREHOLDER Market
# #----------------------------------------------------------------------------------------
# python -m scripts.train --algo dqn --agents 2 --model easy/dqn/2-dqn-mixed-sm `
#     --setting mixed-motive `
#     --market sm `
#     --trading-fee 0.1 `
#     --max-steps 8 `
#     --batch-size 64
    
# python -m scripts.train --algo dqn --agents 2 --model easy/dqn/2-dqn-mixed-sm-goal `
#     --setting mixed-motive `
#     --market sm-goal `
#     --trading-fee 0.1 `
#     --max-steps 8 `
#     --batch-size 64

python -m scripts.train --algo dqn --agents 2 --model easy/dqn/2-dqn-mixed-sm-no-reset `
    --setting mixed-motive `
    --market sm-no-reset `
    --trading-fee 0.1 `
    --max-steps 8 `
    --batch-size 64

python -m scripts.train --algo dqn --agents 2 --model easy/dqn/2-dqn-mixed-sm-goal-no-reset `
    --setting mixed-motive `
    --market sm-goal-no-reset `
    --trading-fee 0.1 `
    --max-steps 8 `
    --batch-size 64

#----------------------------------------------------------------------------------------
# 3 dqn Mixed Motive Settings with ACTION Market
#----------------------------------------------------------------------------------------
# python -m scripts.train --algo dqn --agents 2 --model easy/dqn/2-dqn-mixed-am `
#     --setting mixed-motive `
#     --market am `
#     --trading-fee 0.1 `
#     --max-steps 8 `
#     --batch-size 64
    
# python -m scripts.train --algo dqn --agents 2 --model easy/dqn/2-dqn-mixed-am-goal `
#     --setting mixed-motive `
#     --market am-goal `
#     --trading-fee 0.1 `
#     --max-steps 8 `
#     --batch-size 64
    
python -m scripts.train --algo dqn --agents 2 --model easy/dqn/2-dqn-mixed-am-no-reset `
    --setting mixed-motive `
    --market am-no-reset `
    --trading-fee 0.1 `
    --max-steps 8 `
    --batch-size 64

python -m scripts.train --algo dqn --agents 2 --model easy/dqn/2-dqn-mixed-am-no-debt `
    --setting mixed-motive `
    --market am-no-debt `
    --trading-fee 0.1 `
    --max-steps 8 `
    --batch-size 64

python -m scripts.train --algo dqn --agents 2 --model easy/dqn/2-dqn-mixed-am-goal-no-reset `
    --setting mixed-motive `
    --market am-goal-no-reset `
    --trading-fee 0.1 `
    --max-steps 8 `
    --batch-size 64

python -m scripts.train --algo dqn --agents 2 --model easy/dqn/2-dqn-mixed-am-goal-no-debt `
    --setting mixed-motive `
    --market am-goal-no-debt `
    --trading-fee 0.1 `
    --max-steps 8 `
    --batch-size 64