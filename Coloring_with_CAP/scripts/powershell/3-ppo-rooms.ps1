#----------------------------------------------------------------------------------------
# 1 PPO Room Settings
#----------------------------------------------------------------------------------------
pipenv run python -m Coloring_with_CAP.scripts.train --agents 1 --model 1-ppo-rooms `
    --env FourRooms-Grid-v0 `
    --max-steps 400 `
    --capture-interval 20 `
    --frames 1500000

#----------------------------------------------------------------------------------------
# 3 PPO Room Settings
#----------------------------------------------------------------------------------------
pipenv run python -m Coloring_with_CAP.scripts.train --agents 3 --model 3-ppo-rooms-percentage `
    --env FourRooms-Grid-v0 `
    --max-steps 400 `
    --capture-interval 20 `
    --frames 1500000