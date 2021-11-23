#----------------------------------------------------------------------------------------
# Single agent
#----------------------------------------------------------------------------------------

python -m scripts.train --algo ppo --agents 1 --model hard/ppo/1-ppo `
    --grid-size 7 `
    --frames-per-proc 256 `
    --frames 200000 `
    --capture-interval 15

#----------------------------------------------------------------------------------------
# COOP 3 Agents Settings
#----------------------------------------------------------------------------------------
python -m scripts.train --algo ppo --agents 3 --model hard/ppo/3-ppo-dr `
    --setting difference-reward `
    --grid-size 7 `
    --max-steps 20 `
    --frames-per-proc 256 `
    --frames 200000 `
    --capture-interval 15

python -m scripts.train --algo ppo --agents 3 --model hard/ppo/3-ppo `
    --grid-size 7 `
    --max-steps 20 `
    --frames-per-proc 256 `
    --frames 200000 `
    --capture-interval 15

python -m scripts.train --algo ppo --agents 3 --model hard/ppo/3-ppo-sm-goal-no-reset `
    --market sm-goal-no-reset `
    --grid-size 7 `
    --max-steps 20 `
    --frames-per-proc 256 `
    --frames 200000 `
    --capture-interval 15

#----------------------------------------------------------------------------------------
# Mixed 3 Agents Settings
#----------------------------------------------------------------------------------------

python -m scripts.train --algo ppo --agents 3 --model hard/ppo/3-ppo-mixed `
    --setting mixed-motive `
    --grid-size 7 `
    --max-steps 20 `
    --frames-per-proc 256 `
    --frames 200000 `
    --capture-interval 15

python -m scripts.train --algo ppo --agents 3 --model hard/ppo/3-ppo-mixed-sm-no-reset `
    --setting mixed-motive `
    --market sm-no-reset `
    --grid-size 7 `
    --max-steps 20 `
    --frames-per-proc 256 `
    --frames 200000 `
    --capture-interval 15

python -m scripts.train --algo ppo --agents 3 --model hard/ppo/3-ppo-mixed-sm-goal-no-reset `
    --setting mixed-motive `
    --market sm-goal-no-reset `
    --grid-size 7 `
    --max-steps 20 `
    --frames-per-proc 256 `
    --frames 200000 `
    --capture-interval 15

#----------------------------------------------------------------------------------------
# Competitive 3 Agents Settings
#----------------------------------------------------------------------------------------

python -m scripts.train --algo ppo --agents 3 --model hard/ppo/3-ppo-competitive-sm-goal-no-reset `
    --setting mixed-motive-competitive `
    --market sm-goal-no-reset `
    --grid-size 7 `
    --max-steps 20 `
    --frames-per-proc 256 `
    --frames 200000 `
    --capture-interval 15

python -m scripts.train --algo ppo --agents 3 --model hard/ppo/3-ppo-competitive `
    --setting mixed-motive-competitive `
    --grid-size 7 `
    --max-steps 20 `
    --frames-per-proc 256 `
    --frames 200000 `
    --capture-interval 15

python -m scripts.train --algo ppo --agents 3 --model hard/ppo/3-ppo-competitive-sm-goal `
    --setting mixed-motive-competitive `
    --market sm-goal `
    --grid-size 7 `
    --max-steps 20 `
    --frames-per-proc 256 `
    --frames 200000 `
    --capture-interval 15