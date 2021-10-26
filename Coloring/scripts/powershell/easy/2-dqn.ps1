
# pipenv run python -m Coloring.scripts.train --algo dqn --agents 2 --model easy\\2-dqn-dr `
#     --setting difference-reward `
#     --max-steps 8  `
#     --grid-size 5 `
#     --frames-per-proc 128 `
#     --frames 80000 `
#     --batch-size 64 `
#     --initial-target-update 1000 `
#     --target-update 15000 `
#     --replay-size 40000 `
#     --epsilon-decay 5000

# #----------------------------------------------------------------------------------------
# # 3 dqn Settings
# #----------------------------------------------------------------------------------------
# pipenv run python -m Coloring.scripts.train --algo dqn --agents 2 --model easy\\2-dqn `
#     --max-steps 8  `
#     --grid-size 5 `
#     --frames-per-proc 128 `
#     --frames 80000 `
#     --batch-size 64 `
#     --initial-target-update 1000 `
#     --target-update 15000 `
#     --replay-size 40000 `
#     --epsilon-decay 5000

# #----------------------------------------------------------------------------------------
# # 3 dqn Settings with SHAREHOLDER Market
# #----------------------------------------------------------------------------------------
pipenv run python -m Coloring.scripts.train --algo dqn --model easy\\2-dqn-sm `
    --market sm `
    --trading-fee 0.05 `
    --batch-size 64

pipenv run python -m Coloring.scripts.train --algo dqn --model easy\\2-dqn-sm-goal `
    --market sm-goal `
    --trading-fee 0.05 `
    --batch-size 64

pipenv run python -m Coloring.scripts.train --algo dqn --agents 2 --model easy\\2-dqn-sm-no-reset `
    --market sm-no-reset `
    --trading-fee 0.05 `
    --batch-size 64

pipenv run python -m Coloring.scripts.train --algo dqn --agents 2 --model easy\\2-dqn-sm-goal-no-reset `
    --market sm-goal-no-reset `
    --trading-fee 0.05 `
    --batch-size 64

# pipenv run python -m Coloring.scripts.train --algo dqn --agents 2 --model easy\\2-dqn-sm-no-debt `
#     --market sm-no-debt `
#     --trading-fee 0.2 `
#     --max-steps 8  `
#     --grid-size 5 `
#     --frames-per-proc 128 `
#     --frames 80000 `
#     --batch-size 64 `
#     --initial-target-update 1000 `
#     --target-update 15000 `
#     --replay-size 40000 `
#     --epsilon-decay 5000

# pipenv run python -m Coloring.scripts.train --algo dqn --agents 2 --model easy\\2-dqn-sm-goal-no-debt `
#     --market sm-goal-no-debt `
#     --trading-fee 0.2 `
#     --max-steps 8  `
#     --grid-size 5 `
#     --frames-per-proc 128 `
#     --frames 80000 `
#     --batch-size 64 `
#     --initial-target-update 1000 `
#     --target-update 15000 `
#     --replay-size 40000 `
#     --epsilon-decay 5000

# pipenv run python -m Coloring.scripts.train --algo dqn --agents 2 --model easy\\2-dqn-sm-goal-no-reset-no-debt `
# --market sm-goal-no-reset-no-debt `
# --max-steps 15  `
# --grid-size 5 `
# --frames-per-proc 128 `
# --frames 100000 `
# --batch-size 64 `
# --initial-target-update 1000 `
# --target-update 15000 `
# --replay-size 40000 `
# --epsilon-decay 5000

# pipenv run python -m Coloring.scripts.train --algo dqn --agents 2 --model easy\\2-dqn-sm-no-reset-no-debt `
# --market sm-no-reset-no-debt `
# --max-steps 15  `
# --grid-size 5 `
# --frames-per-proc 128 `
# --frames 100000 `
# --batch-size 64 `
# --initial-target-update 1000 `
# --target-update 15000 `
# --replay-size 40000 `
# --epsilon-decay 5000

#----------------------------------------------------------------------------------------
# 3 dqn Settings with ACTION Market
#----------------------------------------------------------------------------------------
pipenv run python -m Coloring.scripts.train --algo dqn --agents 2 --model easy\\2-dqn-am `
    --market am `
    --trading-fee 0.05 `
    --batch-size 64

pipenv run python -m Coloring.scripts.train --algo dqn --agents 2 --model easy\\2-dqn-am-goal `
    --market am-goal `
    --trading-fee 0.05 `
    --batch-size 64
    
pipenv run python -m Coloring.scripts.train --algo dqn --agents 2 --model easy\\2-dqn-am-no-reset `
    --market am-no-reset `
    --trading-fee 0.05 `
    --batch-size 64

pipenv run python -m Coloring.scripts.train --algo dqn --agents 2 --model easy\\2-dqn-am-no-debt `
    --market am-no-debt `
    --trading-fee 0.05 `
    --batch-size 64

pipenv run python -m Coloring.scripts.train --algo dqn --agents 2 --model easy\\2-dqn-am-goal-no-reset `
    --market am-goal-no-reset `
    --trading-fee 0.05 `
    --batch-size 64

pipenv run python -m Coloring.scripts.train --algo dqn --agents 2 --model easy\\2-dqn-am-goal-no-debt `
    --market am-goal-no-debt `
    --trading-fee 0.05 `
    --batch-size 64

# pipenv run python -m Coloring.scripts.train --algo dqn --agents 2 --model easy\\2-dqn-am-goal-no-reset-no-debt `
# --market am-goal-no-reset-no-debt `
# --max-steps 15  `
# --grid-size 5 `
# --frames-per-proc 128 `
# --frames 100000 `
# --batch-size 64 `
# --initial-target-update 1000 `
# --target-update 15000 `
# --replay-size 40000 `
# --epsilon-decay 5000

# pipenv run python -m Coloring.scripts.train --algo dqn --agents 2 --model easy\\2-dqn-am-no-reset-no-debt `
# --market am-no-reset-no-debt `
# --max-steps 15  `
# --grid-size 5 `
# --frames-per-proc 128 `
# --frames 100000 `
# --batch-size 64 `
# --initial-target-update 1000 `
# --target-update 15000 `
# --replay-size 40000 `
# --epsilon-decay 5000