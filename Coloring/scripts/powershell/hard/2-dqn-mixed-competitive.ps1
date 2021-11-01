# #----------------------------------------------------------------------------------------
# # 3 dqn Mixed Competitive Settings
# #----------------------------------------------------------------------------------------
# pipenv run python -m Coloring.scripts.train --algo dqn --agents 2 --model hard\\2-dqn-mixed-competitive `
#     --setting mixed-motive-competitive `
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
# # 3 dqn Mixed Competitive Settings with SHAREHOLDER Market
# #----------------------------------------------------------------------------------------
pipenv run python -m Coloring.scripts.train --algo dqn --agents 2 --model hard\\2-dqn-mixed-competitive-sm `
    --setting mixed-motive-competitive `
    --market sm `
    --trading-fee 0.05 `
    --batch-size 64

pipenv run python -m Coloring.scripts.train --algo dqn --agents 2 --model hard\\2-dqn-mixed-competitive-sm-goal `
    --setting mixed-motive-competitive `
    --market sm-goal `
    --trading-fee 0.05 `
    --batch-size 64

# pipenv run python -m Coloring.scripts.train --algo dqn --agents 2 --model hard\\2-dqn-mixed-competitive-sm-no-reset `
#     --setting mixed-motive-competitive `
#     --market sm-no-reset `
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

# pipenv run python -m Coloring.scripts.train --algo dqn --agents 2 --model hard\\2-dqn-mixed-competitive-sm-goal-no-reset `
#     --setting mixed-motive-competitive `
#     --market sm-goal-no-reset `
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

# pipenv run python -m Coloring.scripts.train --algo dqn --agents 2 --model hard\\2-dqn-mixed-competitive-sm-no-debt `
#     --setting mixed-motive-competitive `
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

# pipenv run python -m Coloring.scripts.train --algo dqn --agents 2 --model hard\\2-dqn-mixed-competitive-sm-goal-no-debt `
#     --setting mixed-motive-competitive `
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

# pipenv run python -m Coloring.scripts.train --algo dqn --agents 2 --model hard\\2-dqn-mixed-competitive-sm-no-reset-no-debt `
# --setting mixed-motive-competitive `
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

# pipenv run python -m Coloring.scripts.train --algo dqn --agents 2 --model hard\\2-dqn-mixed-competitive-sm-goal-no-reset-no-debt `
# --setting mixed-motive-competitive `
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

#----------------------------------------------------------------------------------------
# 3 dqn Settings with ACTION Market
#----------------------------------------------------------------------------------------
pipenv run python -m Coloring.scripts.train --algo dqn --agents 2 --model hard\\2-dqn-mixed-competitive-am `
    --setting mixed-motive-competitive `
    --market am `
    --trading-fee 0.05 `
    --batch-size 64
    
pipenv run python -m Coloring.scripts.train --algo dqn --agents 2 --model hard\\2-dqn-mixed-competitive-am-goal `
    --setting mixed-motive-competitive `
    --market am-goal `
    --trading-fee 0.05 `
    --batch-size 64
    
# pipenv run python -m Coloring.scripts.train --algo dqn --agents 2 --model hard\\2-dqn-mixed-competitive-am-no-reset `
#     --setting mixed-motive-competitive `
#     --market am-no-reset `
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

pipenv run python -m Coloring.scripts.train --algo dqn --agents 2 --model hard\\2-dqn-mixed-competitive-am-no-debt `
    --setting mixed-motive-competitive `
    --market am-no-debt `
    --trading-fee 0.05 `
    --batch-size 64

# pipenv run python -m Coloring.scripts.train --algo dqn --agents 2 --model hard\\2-dqn-mixed-competitive-am-goal-no-reset `
#     --setting mixed-motive-competitive `
#     --market am-goal-no-reset `
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

pipenv run python -m Coloring.scripts.train --algo dqn --agents 2 --model hard\\2-dqn-mixed-competitive-am-goal-no-debt `
    --setting mixed-motive-competitive `
    --market am-goal-no-debt `
    --trading-fee 0.05 `
    --batch-size 64

# pipenv run python -m Coloring.scripts.train --algo dqn --agents 2 --model hard\\2-dqn-mixed-competitive-am-goal-no-reset-no-debt `
# --setting mixed-motive-competitive `
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

# pipenv run python -m Coloring.scripts.train --algo dqn --agents 2 --model hard\\2-dqn-mixed-competitive-am-no-reset-no-debt `
# --setting mixed-motive-competitive `
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