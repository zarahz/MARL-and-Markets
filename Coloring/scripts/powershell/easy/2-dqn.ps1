
pipenv run python -m Coloring.scripts.train --algo dqn --agents 2 --model easy\\2-dqn-dr `
    --setting difference-reward `
    --max-steps 70  `
    --grid-size 5 `
    --frames-per-proc 128 `
    --frames 100000 `
    --batch-size 32 `
    --initial-target-update 1000 `
    --target-update 20000 `
    --replay-size 60000 `
    --epsilon-decay 30000

#----------------------------------------------------------------------------------------
# 3 dqn Settings
#----------------------------------------------------------------------------------------
pipenv run python -m Coloring.scripts.train --algo dqn --agents 2 --model easy\\2-dqn `
    --max-steps 70  `
    --grid-size 5 `
    --frames-per-proc 128 `
    --frames 100000 `
    --batch-size 32 `
    --initial-target-update 1000 `
    --target-update 20000 `
    --replay-size 60000 `
    --epsilon-decay 30000

#----------------------------------------------------------------------------------------
# 3 dqn Settings with SHAREHOLDER Market
#----------------------------------------------------------------------------------------
pipenv run python -m Coloring.scripts.train --algo dqn --agents 2 --model easy\\2-dqn-sm `
    --market sm `
    --max-steps 70  `
    --grid-size 5 `
    --frames-per-proc 128 `
    --frames 100000 `
    --batch-size 32 `
    --initial-target-update 1000 `
    --target-update 20000 `
    --replay-size 60000 `
    --epsilon-decay 30000

pipenv run python -m Coloring.scripts.train --algo dqn --agents 2 --model easy\\2-dqn-sm-goal `
    --market sm-goal `
    --max-steps 70  `
    --grid-size 5 `
    --frames-per-proc 128 `
    --frames 100000 `
    --batch-size 32 `
    --initial-target-update 1000 `
    --target-update 20000 `
    --replay-size 60000 `
    --epsilon-decay 30000

pipenv run python -m Coloring.scripts.train --algo dqn --agents 2 --model easy\\2-dqn-sm-no-reset `
    --market sm-no-reset `
    --max-steps 70  `
    --grid-size 5 `
    --frames-per-proc 128 `
    --frames 100000 `
    --batch-size 32 `
    --initial-target-update 1000 `
    --target-update 20000 `
    --replay-size 60000 `
    --epsilon-decay 30000

pipenv run python -m Coloring.scripts.train --algo dqn --agents 2 --model easy\\2-dqn-sm-no-debt `
    --market sm-no-debt `
    --max-steps 70  `
    --grid-size 5 `
    --frames-per-proc 128 `
    --frames 100000 `
    --batch-size 32 `
    --initial-target-update 1000 `
    --target-update 20000 `
    --replay-size 60000 `
    --epsilon-decay 30000

pipenv run python -m Coloring.scripts.train --algo dqn --agents 2 --model easy\\2-dqn-sm-goal-no-reset `
    --market sm-goal-no-reset `
    --max-steps 70  `
    --grid-size 5 `
    --frames-per-proc 128 `
    --frames 100000 `
    --batch-size 32 `
    --initial-target-update 1000 `
    --target-update 20000 `
    --replay-size 60000 `
    --epsilon-decay 30000

pipenv run python -m Coloring.scripts.train --algo dqn --agents 2 --model easy\\2-dqn-sm-goal-no-debt `
    --market sm-goal-no-debt `
    --max-steps 70  `
    --grid-size 5 `
    --frames-per-proc 128 `
    --frames 100000 `
    --batch-size 32 `
    --initial-target-update 1000 `
    --target-update 20000 `
    --replay-size 60000 `
    --epsilon-decay 30000

# pipenv run python -m Coloring.scripts.train --algo dqn --agents 2 --model easy\\2-dqn-sm-goal-no-reset-no-debt `
#     --market sm-goal-no-reset-no-debt `
#     --max-steps 70  `
#     --grid-size 5 `
#     --frames-per-proc 128 `
#     --frames 100000 `
#     --batch-size 32 `
#     --initial-target-update 1000 `
#     --target-update 20000 `
#     --replay-size 60000 `
#     --epsilon-decay 30000

# pipenv run python -m Coloring.scripts.train --algo dqn --agents 2 --model easy\\2-dqn-sm-no-reset-no-debt `
#     --market sm-no-reset-no-debt `
#     --max-steps 70  `
#     --grid-size 5 `
#     --frames-per-proc 128 `
#     --frames 100000 `
#     --batch-size 32 `
#     --initial-target-update 1000 `
#     --target-update 20000 `
#     --replay-size 60000 `
#     --epsilon-decay 30000

#----------------------------------------------------------------------------------------
# 3 dqn Settings with ACTION Market
#----------------------------------------------------------------------------------------
pipenv run python -m Coloring.scripts.train --algo dqn --agents 2 --model easy\\2-dqn-am `
    --market am `
    --max-steps 70  `
    --grid-size 5 `
    --frames-per-proc 128 `
    --frames 100000 `
    --batch-size 32 `
    --initial-target-update 1000 `
    --target-update 20000 `
    --replay-size 60000 `
    --epsilon-decay 30000

pipenv run python -m Coloring.scripts.train --algo dqn --agents 2 --model easy\\2-dqn-am-goal `
    --market am-goal `
    --max-steps 70  `
    --grid-size 5 `
    --frames-per-proc 128 `
    --frames 100000 `
    --batch-size 32 `
    --initial-target-update 1000 `
    --target-update 20000 `
    --replay-size 60000 `
    --epsilon-decay 30000
    
pipenv run python -m Coloring.scripts.train --algo dqn --agents 2 --model easy\\2-dqn-am-no-reset `
    --market am-no-reset `
    --max-steps 70  `
    --grid-size 5 `
    --frames-per-proc 128 `
    --frames 100000 `
    --batch-size 32 `
    --initial-target-update 1000 `
    --target-update 20000 `
    --replay-size 60000 `
    --epsilon-decay 30000

pipenv run python -m Coloring.scripts.train --algo dqn --agents 2 --model easy\\2-dqn-am-no-debt `
    --market am-no-debt `
    --max-steps 70  `
    --grid-size 5 `
    --frames-per-proc 128 `
    --frames 100000 `
    --batch-size 32 `
    --initial-target-update 1000 `
    --target-update 20000 `
    --replay-size 60000 `
    --epsilon-decay 30000

pipenv run python -m Coloring.scripts.train --algo dqn --agents 2 --model easy\\2-dqn-am-goal-no-reset `
    --market am-goal-no-reset `
    --max-steps 70  `
    --grid-size 5 `
    --frames-per-proc 128 `
    --frames 100000 `
    --batch-size 32 `
    --initial-target-update 1000 `
    --target-update 20000 `
    --replay-size 60000 `
    --epsilon-decay 30000

pipenv run python -m Coloring.scripts.train --algo dqn --agents 2 --model easy\\2-dqn-am-goal-no-debt `
    --market am-goal-no-debt `
    --max-steps 70  `
    --grid-size 5 `
    --frames-per-proc 128 `
    --frames 100000 `
    --batch-size 32 `
    --initial-target-update 1000 `
    --target-update 20000 `
    --replay-size 60000 `
    --epsilon-decay 30000

# pipenv run python -m Coloring.scripts.train --algo dqn --agents 2 --model easy\\2-dqn-am-goal-no-reset-no-debt `
#     --market am-goal-no-reset-no-debt `
#     --max-steps 70  `
#     --grid-size 5 `
#     --frames-per-proc 128 `
#     --frames 100000 `
#     --batch-size 32 `
#     --initial-target-update 1000 `
#     --target-update 20000 `
#     --replay-size 60000 `
#     --epsilon-decay 30000

# pipenv run python -m Coloring.scripts.train --algo dqn --agents 2 --model easy\\2-dqn-am-no-reset-no-debt `
#     --market am-no-reset-no-debt `
#     --max-steps 70  `
#     --grid-size 5 `
#     --frames-per-proc 128 `
#     --frames 100000 `
#     --batch-size 32 `
#     --initial-target-update 1000 `
#     --target-update 20000 `
#     --replay-size 60000 `
#     --epsilon-decay 30000