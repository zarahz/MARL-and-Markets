#----------------------------------------------------------------------------------------
# 1 PPO Settings
#----------------------------------------------------------------------------------------
pipenv run python -m Coloring.scripts.train --algo ppo --agents 1 --model easy\\1-ppo `
    --max-steps 10
# --grid-size 5 `
# --frames-per-proc 128 `
# --frames 100000


#----------------------------------------------------------------------------------------
# 1 PPO Room Settings
#----------------------------------------------------------------------------------------
# pipenv run python -m Coloring.scripts.train --algo ppo --agents 1 --model easy\\1-ppo-rooms `
#     --env FourRooms-Grid-v0 `
#     --grid-size 5 `
#     --frames-per-proc 128 `
#     --frames 150000