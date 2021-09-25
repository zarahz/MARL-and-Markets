#----------------------------------------------------------------------------------------
# 1 PPO Settings
#----------------------------------------------------------------------------------------
pipenv run python -m Coloring_with_CAP.scripts.train --agents 1 --model 1-ppo `
    --max-steps 300 `
    --capture-interval 20
