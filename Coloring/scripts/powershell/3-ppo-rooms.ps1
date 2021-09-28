#----------------------------------------------------------------------------------------
# 3 PPO Room Settings
#----------------------------------------------------------------------------------------
pipenv run python -m Coloring.scripts.train --algo ppo --agents 3 --model 3-ppo-rooms-percentage `
    --env FourRooms-Grid-v0 `
    --max-steps 400 `
    --capture-interval 20 `
    --frames 1500000