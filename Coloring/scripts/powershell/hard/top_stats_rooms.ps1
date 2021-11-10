
python -m scripts.train --algo dqn --agents 3 --model hard_rooms/dqn/3-dqn-competitive-sm-goal-no-reset `
    --env FourRooms-Grid-v0 `
    --setting mixed-motive-competitive `
    --market sm-goal-no-reset `
    --batch-size 64 `
    --initial-target-update 1000 `
    --target-update 10000 `
    --replay-size 700000 `
    --epsilon-decay 40200 `
    --grid-size 9 `
    --max-steps 30 `
    --frames-per-proc 512 `
    --frames 402000 `
    --capture-interval 15

python -m scripts.train --algo ppo --agents 3 --model hard_rooms/ppo/3-ppo-competitive `
    --env FourRooms-Grid-v0 `
    --setting mixed-motive-competitive `
    --grid-size 9 `
    --max-steps 30 `
    --frames-per-proc 512 `
    --frames 402000 `
    --capture-interval 15

python -m scripts.train --algo dqn --agents 3 --model hard_rooms/dqn/3-dqn-competitive-sm `
    --env FourRooms-Grid-v0 `
    --setting mixed-motive-competitive `
    --market sm `
    --batch-size 64 `
    --initial-target-update 1000 `
    --target-update 10000 `
    --replay-size 700000 `
    --epsilon-decay 40200 `
    --grid-size 9 `
    --max-steps 30 `
    --frames-per-proc 512 `
    --frames 402000 `
    --capture-interval 15