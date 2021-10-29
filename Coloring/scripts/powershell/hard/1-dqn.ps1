#----------------------------------------------------------------------------------------
# 1 DQN Settings
#----------------------------------------------------------------------------------------
# pipenv run python -m Coloring.scripts.train --algo dqn --agents 1 --model hard\\1-dqn-1 `
#     --batch-size 64 `
#     --epsilon-decay 10000 `
#     --grid-size 7 `
#     --frames-per-proc 512 `
#     --frames 250000

#     pipenv run python -m Coloring.scripts.train --algo dqn --agents 1 --model hard\\1-dqn-2 `
#     --batch-size 128 `
#     --initial-target-update 1000 `
#     --target-update 50000 `
#     --replay-size 125000 `
#     --epsilon-decay 10000 `
#     --grid-size 7 `
#     --frames-per-proc 512 `
#     --frames 250000

#     pipenv run python -m Coloring.scripts.train --algo dqn --agents 1 --model hard\\1-dqn-3 `
#     --batch-size 256 `
#     --initial-target-update 1000 `
#     --target-update 50000 `
#     --replay-size 125000 `
#     --epsilon-decay 20000 `
#     --grid-size 7 `
#     --frames-per-proc 512 `
#     --frames 250000

#     pipenv run python -m Coloring.scripts.train --algo dqn --agents 1 --model hard\\1-dqn-4 `
#     --batch-size 64 `
#     --initial-target-update 1000 `
#     --target-update 25000 `
#     --replay-size 125000 `
#     --epsilon-decay 10000 `
#     --grid-size 7 `
#     --frames-per-proc 512 `
#     --frames 250000

#     pipenv run python -m Coloring.scripts.train --algo dqn --agents 1 --model hard\\1-dqn-5 `
#     --batch-size 64 `
#     --initial-target-update 1000 `
#     --target-update 75000 `
#     --replay-size 125000 `
#     --epsilon-decay 10000 `
#     --grid-size 7 `
#     --frames-per-proc 512 `
#     --frames 250000

#     pipenv run python -m Coloring.scripts.train --algo dqn --agents 1 --model hard\\1-dqn-6 `
#     --batch-size 64 `
#     --initial-target-update 1000 `
#     --target-update 50000 `
#     --replay-size 175000 `
#     --epsilon-decay 10000 `
#     --grid-size 7 `
#     --frames-per-proc 512 `
#     --frames 250000

#     pipenv run python -m Coloring.scripts.train --algo dqn --agents 1 --model hard\\1-dqn-7 `
#     --batch-size 64 `
#     --initial-target-update 1000 `
#     --target-update 50000 `
#     --replay-size 75000 `
#     --epsilon-decay 10000 `
#     --grid-size 7 `
#     --frames-per-proc 512 `
#     --frames 250000

pipenv run python -m Coloring.scripts.train --algo dqn --agents 1 --model hard\\1-dqn-8 `
    --batch-size 64 `
    --epsilon-decay 20000 `
    --grid-size 7 `
    --frames-per-proc 512 `
    --frames 250000

pipenv run python -m Coloring.scripts.train --algo dqn --agents 1 --model hard\\1-dqn-9 `
    --batch-size 64 `
    --epsilon-decay 20000 `
    --replay-size 100000 `
    --grid-size 7 `
    --frames-per-proc 512 `
    --frames 250000

pipenv run python -m Coloring.scripts.train --algo dqn --agents 1 --model hard\\1-dqn-10 `
    --batch-size 64 `
    --epsilon-decay 20000 `
    --target-update 30000 `
    --grid-size 7 `
    --frames-per-proc 512 `
    --frames 250000
    
#----------------------------------------------------------------------------------------
# 1 DQN Room Settings
#----------------------------------------------------------------------------------------
# pipenv run python -m Coloring.scripts.train --algo dqn --agents 1 --model hard\\1-dqn-rooms `
#     --env FourRooms-Grid-v0 `
#     --max-steps 400 `
#     --capture-interval 20 `
#     --initial-target-update 35000 `
#     --target-update 350000 `
#     --epsilon-decay 30000 `
#     --frames 1500000