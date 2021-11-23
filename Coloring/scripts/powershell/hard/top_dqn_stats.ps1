#----------------------------------------------------------------------------------------
# Single agent
#----------------------------------------------------------------------------------------

python -m scripts.train --algo dqn --agents 1 --model hard/dqn/1-dqn `
    --target-update 10000 `
    --replay-size 700000 `
    --epsilon-decay 20000 `
    --grid-size 7 `
    --frames-per-proc 256 `
    --frames 200000 `
    --capture-interval 15

#----------------------------------------------------------------------------------------
# COOP 3 Agents Settings
#----------------------------------------------------------------------------------------
python -m scripts.train --algo dqn --agents 3 --model hard/dqn/3-dqn-dr `
    --setting difference-reward `
    --target-update 10000 `
    --replay-size 700000 `
    --epsilon-decay 20000 `
    --grid-size 7 `
    --max-steps 20 `
    --frames-per-proc 256 `
    --frames 200000 `
    --capture-interval 15

python -m scripts.train --algo dqn --agents 3 --model hard/dqn/3-dqn-am-goal-no-debt `
    --market am-goal-no-debt `
    --target-update 10000 `
    --replay-size 700000 `
    --epsilon-decay 20000 `
    --grid-size 7 `
    --max-steps 20 `
    --frames-per-proc 256 `
    --frames 200000 `
    --capture-interval 15

python -m scripts.train --algo dqn --agents 3 --model hard/dqn/3-dqn-am-goal `
    --market am-goal `
    --target-update 10000 `
    --replay-size 700000 `
    --epsilon-decay 20000 `
    --grid-size 7 `
    --max-steps 20 `
    --frames-per-proc 256 `
    --frames 200000 `
    --capture-interval 15

#----------------------------------------------------------------------------------------
# Mixed 3 Agents Settings
#----------------------------------------------------------------------------------------

python -m scripts.train --algo dqn --agents 3 --model hard/dqn/3-dqn-mixed-sm `
    --setting mixed-motive `
    --market sm `
    --target-update 10000 `
    --replay-size 700000 `
    --epsilon-decay 20000 `
    --grid-size 7 `
    --max-steps 20 `
    --frames-per-proc 256 `
    --frames 200000 `
    --capture-interval 15

python -m scripts.train --algo dqn --agents 3 --model hard/dqn/3-dqn-mixed-am-no-reset `
    --setting mixed-motive `
    --market am-no-reset `
    --target-update 10000 `
    --replay-size 700000 `
    --epsilon-decay 20000 `
    --grid-size 7 `
    --max-steps 20 `
    --frames-per-proc 256 `
    --frames 200000 `
    --capture-interval 15

python -m scripts.train --algo dqn --agents 3 --model hard/dqn/3-dqn-mixed-sm-goal `
    --setting mixed-motive `
    --market sm-goal `
    --target-update 10000 `
    --replay-size 700000 `
    --epsilon-decay 20000 `
    --grid-size 7 `
    --max-steps 20 `
    --frames-per-proc 256 `
    --frames 200000 `
    --capture-interval 15

#----------------------------------------------------------------------------------------
# Competitive 3 Agents Settings
#----------------------------------------------------------------------------------------

python -m scripts.train --algo dqn --agents 3 --model hard/dqn/3-dqn-competitive-sm-goal-no-reset `
    --setting mixed-motive-competitive `
    --market sm-goal-no-reset `
    --target-update 10000 `
    --replay-size 700000 `
    --epsilon-decay 20000 `
    --grid-size 7 `
    --max-steps 20 `
    --frames-per-proc 256 `
    --frames 200000 `
    --capture-interval 15

python -m scripts.train --algo dqn --agents 3 --model hard/dqn/3-dqn-competitive-sm `
    --setting mixed-motive-competitive `
    --market sm `
    --target-update 10000 `
    --replay-size 700000 `
    --epsilon-decay 20000 `
    --grid-size 7 `
    --max-steps 20 `
    --frames-per-proc 256 `
    --frames 200000 `
    --capture-interval 15

python -m scripts.train --algo dqn --agents 3 --model hard/dqn/3-dqn-competitive-am-goal-no-debt `
    --setting mixed-motive-competitive `
    --market am-goal-no-debt `
    --target-update 10000 `
    --replay-size 700000 `
    --epsilon-decay 20000 `
    --grid-size 7 `
    --max-steps 20 `
    --frames-per-proc 256 `
    --frames 200000 `
    --capture-interval 15