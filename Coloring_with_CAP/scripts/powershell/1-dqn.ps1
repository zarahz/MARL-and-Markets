#----------------------------------------------------------------------------------------
# 1 DQN Settings
#----------------------------------------------------------------------------------------
pipenv run python -m Coloring_with_CAP.scripts.train_dqn --agents 1 --model 1-dqn `
    --max-steps 300 `
    --capture-interval 20 `
    --initial-target-update 35000 `
    --target-update 350000 `
    --epsilon-decay 30000 `

#----------------------------------------------------------------------------------------
# 1 DQN Room Settings
#----------------------------------------------------------------------------------------
# pipenv run python -m Coloring_with_CAP.scripts.train --agents 1 --model 1-dqn-rooms `
#     --env FourRooms-Grid-v0 `
#     --max-steps 400 `
#     --capture-interval 20 `
#     --initial-target-update 35000 `
#     --target-update 350000 `
#     --epsilon-decay 30000 `
#     --frames 1500000