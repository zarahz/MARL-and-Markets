#----------------------------------------------------------------------------------------
# 1 PPO Settings
#----------------------------------------------------------------------------------------
pipenv run python -m Coloring.scripts.train --algo ppo --agents 1 --model hard\\1-ppo `
    --grid-size 7 `
    --frames-per-proc 512 `
    --frames 250000

#----------------------------------------------------------------------------------------
# 1 PPO Room Settings
#----------------------------------------------------------------------------------------
# pipenv run python -m Coloring.scripts.train --algo ppo --agents 1 --model hard\\1-ppo-rooms `
#     --env FourRooms-Grid-v0 `
#     --max-steps 400 `
#     --capture-interval 20 `
#     --frames 1500000