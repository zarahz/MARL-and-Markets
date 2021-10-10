#----------------------------------------------------------------------------------------
# 1 DQN Settings
#----------------------------------------------------------------------------------------
pipenv run python -m Coloring.scripts.train --algo dqn --agents 1 --model 1-dqn-15 `
    --max-steps 300 `
    --capture-interval 20 `
    --batch-size 64 `
    --initial-target-update 500 `
    --target-update 100000 `
    --replay-size 100000 `
    --epsilon-decay 400000 `

#----------------------------------------------------------------------------------------
# 1 DQN Room Settings
#----------------------------------------------------------------------------------------
# pipenv run python -m Coloring.scripts.train --algo dqn --agents 1 --model 1-dqn-rooms `
#     --env FourRooms-Grid-v0 `
#     --max-steps 400 `
#     --capture-interval 20 `
#     --initial-target-update 35000 `
#     --target-update 350000 `
#     --epsilon-decay 30000 `
#     --frames 1500000