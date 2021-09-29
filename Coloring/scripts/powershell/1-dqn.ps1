#----------------------------------------------------------------------------------------
# 1 DQN Settings
#----------------------------------------------------------------------------------------
pipenv run python -m Coloring.scripts.train --algo dqn --agents 1 --model 1-dqn-1 `
    --max-steps 300 `
    --capture-interval 20 `
    --batch-size 64 `
    --initial-target-update 1000 `
    --target-update 10000 `
    --replay-size 400000 `
    --epsilon-decay 400000 `

# pipenv run python -m Coloring.scripts.train --algo dqn --agents 1 --model 1-dqn-2 `
#     --max-steps 300 `
#     --capture-interval 20 `
#     --batch-size 32 `
#     --initial-target-update 1000 `
#     --target-update 10000 `
#     --epsilon-decay 400000 `

# pipenv run python -m Coloring.scripts.train --algo dqn --agents 1 --model 1-dqn-3 `
#     --max-steps 300 `
#     --capture-interval 20 `
#     --batch-size 32 `
#     --initial-target-update 500 `
#     --target-update 5000 `
#     --epsilon-decay 400000 `

# pipenv run python -m Coloring.scripts.train --algo dqn --agents 1 --model 1-dqn-4 `
#     --max-steps 300 `
#     --capture-interval 20 `
#     --batch-size 32 `
#     --initial-target-update 1000 `
#     --target-update 10000 `
#     --epsilon-decay 400000 `

# pipenv run python -m Coloring.scripts.train --algo dqn --agents 1 --model 1-dqn-5 `
#     --max-steps 300 `
#     --capture-interval 20 `
#     --batch-size 64 `
#     --initial-target-update 500 `
#     --target-update 5000 `
#     --epsilon-decay 400000 `

# pipenv run python -m Coloring.scripts.train --algo dqn --agents 1 --model 1-dqn-6 `
#     --max-steps 300 `
#     --capture-interval 20 `
#     --batch-size 64 `
#     --initial-target-update 1000 `
#     --target-update 10000 `
#     --epsilon-decay 400000 `

# pipenv run python -m Coloring.scripts.train --algo dqn --agents 1 --model 1-dqn-6 `
#     --max-steps 300 `
#     --capture-interval 20 `
#     --batch-size 64 `
#     --initial-target-update 10000 `
#     --target-update 100000 `
#     --epsilon-decay 400000 `

# pipenv run python -m Coloring.scripts.train --algo dqn --agents 1 --model 1-dqn-7 `
#     --max-steps 300 `
#     --capture-interval 20 `
#     --batch-size 128 `
#     --initial-target-update 500 `
#     --target-update 5000 `
#     --epsilon-decay 400000 `

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