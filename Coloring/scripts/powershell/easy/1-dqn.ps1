#----------------------------------------------------------------------------------------
# 1 DQN Settings
#----------------------------------------------------------------------------------------
pipenv run python -m Coloring.scripts.train --algo dqn --agents 1 --model easy\\1-dqn `
    --grid-size 5 `
    --frames-per-proc 128 `
    --frames 100000 `
    --batch-size 64 `

#----------------------------------------------------------------------------------------
# 1 DQN Room Settings
#----------------------------------------------------------------------------------------
# pipenv run python -m Coloring.scripts.train --algo dqn --agents 1 --model easy\\1-dqn-rooms `
#     --env FourRooms-Grid-v0 `
# --grid-size 5 `
# --frames-per-proc 128 `
# --frames 100000 `
# --batch-size 64 `
