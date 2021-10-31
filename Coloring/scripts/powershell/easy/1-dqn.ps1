#----------------------------------------------------------------------------------------
# 1 DQN Settings
#----------------------------------------------------------------------------------------
python -m scripts.train --algo dqn --agents 1 --model easy/1-dqn `
    --batch-size 64 `
    --max-steps 10 `
    # --grid-size 5 `
    # --frames-per-proc 128 `
# --frames 100000 `

#----------------------------------------------------------------------------------------
# 1 DQN Room Settings
#----------------------------------------------------------------------------------------
# python -m scripts.train --algo dqn --agents 1 --model easy/1-dqn-rooms `
#     --env FourRooms-Grid-v0 `
# --grid-size 5 `
# --frames-per-proc 128 `
# --frames 100000 `
# --batch-size 64 `
