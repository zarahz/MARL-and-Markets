#----------------------------------------------------------------------------------------
# 3 dqn Mixed Competitive Settings
#----------------------------------------------------------------------------------------
# pipenv run python -m Coloring.scripts.train --algo dqn --agents 2 --model easy\\2-dqn-mixed-competitive `
#     --setting mixed-motive-competitive `
#     --max-steps 70  `
#     --grid-size 5 `
#     --frames-per-proc 128 `
#     --frames 100000 `
#     --batch-size 32 `
#     --initial-target-update 1000 `
#     --target-update 20000 `
#     --replay-size 60000 `
#     --epsilon-decay 30000 `

#----------------------------------------------------------------------------------------
# 3 dqn Mixed Competitive Settings with SHAREHOLDER Market
#----------------------------------------------------------------------------------------
pipenv run python -m Coloring.scripts.train --algo dqn --agents 2 --model easy\\2-dqn-mixed-competitive-sm `
    --setting mixed-motive-competitive `
    --market sm `
    --max-steps 70  `
    --grid-size 5 `
    --frames-per-proc 128 `
    --frames 100000 `
    --batch-size 32 `
    --initial-target-update 1000 `
    --target-update 20000 `
    --replay-size 60000 `
    --epsilon-decay 30000 `

pipenv run python -m Coloring.scripts.train --algo dqn --agents 2 --model easy\\2-dqn-mixed-competitive-sm-goal `
    --setting mixed-motive-competitive `
    --market sm-goal `
    --max-steps 70  `
    --grid-size 5 `
    --frames-per-proc 128 `
    --frames 100000 `
    --batch-size 32 `
    --initial-target-update 1000 `
    --target-update 20000 `
    --replay-size 60000 `
    --epsilon-decay 30000 `

pipenv run python -m Coloring.scripts.train --algo dqn --agents 2 --model easy\\2-dqn-mixed-competitive-sm-no-reset `
    --setting mixed-motive-competitive `
    --market sm-no-reset `
    --max-steps 70  `
    --grid-size 5 `
    --frames-per-proc 128 `
    --frames 100000 `
    --batch-size 32 `
    --initial-target-update 1000 `
    --target-update 20000 `
    --replay-size 60000 `
    --epsilon-decay 30000 `

pipenv run python -m Coloring.scripts.train --algo dqn --agents 2 --model easy\\2-dqn-mixed-competitive-sm-no-debt `
    --setting mixed-motive-competitive `
    --market sm-no-debt `
    --max-steps 70  `
    --grid-size 5 `
    --frames-per-proc 128 `
    --frames 100000 `
    --batch-size 32 `
    --initial-target-update 1000 `
    --target-update 20000 `
    --replay-size 60000 `
    --epsilon-decay 30000 `

pipenv run python -m Coloring.scripts.train --algo dqn --agents 2 --model easy\\2-dqn-mixed-competitive-sm-goal-no-reset-no-debt `
    --setting mixed-motive-competitive `
    --market sm-goal-no-reset-no-debt `
    --max-steps 70  `
    --grid-size 5 `
    --frames-per-proc 128 `
    --frames 100000 `
    --batch-size 32 `
    --initial-target-update 1000 `
    --target-update 20000 `
    --replay-size 60000 `
    --epsilon-decay 30000 `

# pipenv run python -m Coloring.scripts.train --algo dqn --agents 2 --model easy\\2-dqn-mixed-competitive-sm-goal-no-reset `
#     --setting mixed-motive-competitive `
#     --market sm-goal-no-reset `
#     --max-steps 70  `
# --grid-size 5 `
# --frames-per-proc 128 `
# --frames 100000 `
# --batch-size 32 `
# --initial-target-update 1000 `
# --target-update 20000 `
# --replay-size 60000 `
# --epsilon-decay 30000 `

#----------------------------------------------------------------------------------------
# 3 dqn Settings with ACTION Market
#----------------------------------------------------------------------------------------
pipenv run python -m Coloring.scripts.train --algo dqn --agents 2 --model easy\\2-dqn-mixed-competitive-am `
    --setting mixed-motive-competitive `
    --market am `
    --max-steps 70  `
    --grid-size 5 `
    --frames-per-proc 128 `
    --frames 100000 `
    --batch-size 32 `
    --initial-target-update 1000 `
    --target-update 20000 `
    --replay-size 60000 `
    --epsilon-decay 30000 `
    
pipenv run python -m Coloring.scripts.train --algo dqn --agents 2 --model easy\\2-dqn-mixed-competitive-am-goal `
    --setting mixed-motive-competitive `
    --market am-goal `
    --max-steps 70  `
    --grid-size 5 `
    --frames-per-proc 128 `
    --frames 100000 `
    --batch-size 32 `
    --initial-target-update 1000 `
    --target-update 20000 `
    --replay-size 60000 `
    --epsilon-decay 30000 `
    
pipenv run python -m Coloring.scripts.train --algo dqn --agents 2 --model easy\\2-dqn-mixed-competitive-am-no-reset `
    --setting mixed-motive-competitive `
    --market am-no-reset `
    --max-steps 70  `
    --grid-size 5 `
    --frames-per-proc 128 `
    --frames 100000 `
    --batch-size 32 `
    --initial-target-update 1000 `
    --target-update 20000 `
    --replay-size 60000 `
    --epsilon-decay 30000 `

pipenv run python -m Coloring.scripts.train --algo dqn --agents 2 --model easy\\2-dqn-mixed-competitive-am-no-debt `
    --setting mixed-motive-competitive `
    --market am-no-debt `
    --max-steps 70  `
    --grid-size 5 `
    --frames-per-proc 128 `
    --frames 100000 `
    --batch-size 32 `
    --initial-target-update 1000 `
    --target-update 20000 `
    --replay-size 60000 `
    --epsilon-decay 30000 `

pipenv run python -m Coloring.scripts.train --algo dqn --agents 2 --model easy\\2-dqn-mixed-competitive-am-goal-no-reset-no-debt `
    --setting mixed-motive-competitive `
    --market am-goal-no-reset-no-debt `
    --max-steps 70  `
    --grid-size 5 `
    --frames-per-proc 128 `
    --frames 100000 `
    --batch-size 32 `
    --initial-target-update 1000 `
    --target-update 20000 `
    --replay-size 60000 `
    --epsilon-decay 30000 `

# pipenv run python -m Coloring.scripts.train --algo dqn --agents 2 --model easy\\2-dqn-mixed-competitive-am-goal-no-reset `
#     --setting mixed-motive-competitive `
#     --market am-goal-no-reset `
#     --max-steps 70  `
# --grid-size 5 `
# --frames-per-proc 128 `
# --frames 100000 `
# --batch-size 32 `
# --initial-target-update 1000 `
# --target-update 20000 `
# --replay-size 60000 `
# --epsilon-decay 30000 `