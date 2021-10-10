#----------------------------------------------------------------------------------------
# 1 PPO Settings
#----------------------------------------------------------------------------------------
pipenv run python -m Coloring.scripts.train --algo ppo --agents 1 --model 1-ppo `
    --max-steps 300 `
    --capture-interval 20

#----------------------------------------------------------------------------------------
# 1 PPO Room Settings
#----------------------------------------------------------------------------------------
pipenv run python -m Coloring.scripts.train --algo ppo --agents 1 --model 1-ppo-rooms `
    --env FourRooms-Grid-v0 `
    --max-steps 400 `
    --capture-interval 20 `
    --frames 1500000