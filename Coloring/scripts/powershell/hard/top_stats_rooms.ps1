# PPO COOP
python -m scripts.train --algo ppo --agents 3 --model hard_rooms/ppo/3-ppo-dr `
    --env FourRooms-Grid-v0 `
    --setting difference-reward `
    --grid-size 9 `
    --batch-size 64 `
    --max-steps 30 `
    --frames-per-proc 256 `
    --frames 200000 `
    --capture-interval 15

# PPO MIXED
python -m scripts.train --algo ppo --agents 3 --model hard_rooms/ppo/3-ppo-mixed `
    --env FourRooms-Grid-v0 `
    --setting mixed-motive `
    --grid-size 9 `
    --batch-size 64 `
    --max-steps 30 `
    --frames-per-proc 256 `
    --frames 200000 `
    --capture-interval 15

# PPO COMP
python -m scripts.train --algo ppo --agents 3 --model hard_rooms/ppo/3-ppo-competitive `
    --env FourRooms-Grid-v0 `
    --setting mixed-motive-competitive `
    --grid-size 9 `
    --batch-size 64 `
    --max-steps 30 `
    --frames-per-proc 256 `
    --frames 200000 `
    --capture-interval 15

# DQN COOP
# python -m scripts.train --algo dqn --agents 3 --model hard_rooms/dqn/3-dqn-dr `
#     --env FourRooms-Grid-v0 `
#     --setting difference-reward `
#     --batch-size 64 `
#     --initial-target-update 1000 `
#     --target-update 10000 `
#     --replay-size 700000 `
#     --epsilon-decay 20000 `
#     --grid-size 9 `
#     --max-steps 30 `
#     --frames-per-proc 256 `
#     --frames 200000 `
#     --capture-interval 15

# # DQN MIXED
# python -m scripts.train --algo dqn --agents 3 --model hard_rooms/dqn/3-dqn-mixed-sm-goal `
#     --env FourRooms-Grid-v0 `
#     --setting mixed-motive `
#     --market sm-goal `
#     --batch-size 64 `
#     --initial-target-update 1000 `
#     --target-update 10000 `
#     --replay-size 700000 `
#     --epsilon-decay 20000 `
#     --grid-size 9 `
#     --max-steps 30 `
#     --frames-per-proc 256 `
#     --frames 200000 `
#     --capture-interval 15

# DQN COMP
# python -m scripts.train --algo dqn --agents 3 --model hard_rooms/dqn/3-dqn-competitive-sm-goal-no-reset `
#     --env FourRooms-Grid-v0 `
#     --setting mixed-motive-competitive `
#     --market sm-goal-no-reset `
#     --batch-size 64 `
#     --initial-target-update 1000 `
#     --target-update 10000 `
#     --replay-size 700000 `
#     --epsilon-decay 20000 `
#     --grid-size 9 `
#     --max-steps 30 `
#     --frames-per-proc 256 `
#     --frames 200000 `
#     --capture-interval 15