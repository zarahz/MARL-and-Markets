#----------------------------------------------------------------------------------------
# 1 DQN Settings
#----------------------------------------------------------------------------------------
pipenv run python -m Coloring.scripts.train --algo dqn --agents 1 --model easy\\1-dqn `
    --grid-size 5 `
    --frames-per-proc 128 `
    --frames 100000 `
    --batch-size 32 `
    --initial-target-update 1000 `
    --target-update 20000 `
    --replay-size 60000 `
    --epsilon-decay 30000

#----------------------------------------------------------------------------------------
# 1 DQN Room Settings
#----------------------------------------------------------------------------------------
# pipenv run python -m Coloring.scripts.train --algo dqn --agents 1 --model easy\\1-dqn-rooms `
#     --env FourRooms-Grid-v0 `
#     --grid-size 5 `
#     --frames-per-proc 128 `
#     --frames 100000 `
#     --batch-size 32 `
#     --initial-target-update 1000 `
#     --target-update 20000 `
#     --replay-size 60000 `
#     --epsilon-decay 30000 `
