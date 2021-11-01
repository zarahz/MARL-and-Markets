# grid size 7 == 25 fields for agents to color
#----------------------------------------------------------------------------------------
# COOP 3 PPO Settings
#----------------------------------------------------------------------------------------
python -m Coloring.scripts.train --algo ppo --agents 3 --model hard/ppo/3-ppo-dr `
    --setting difference-reward `
    --grid-size 7 `
    --max-steps 30 `
    --frames-per-proc 256 `
    --frames 250000 `
    --capture-interval 15

python -m Coloring.scripts.train --algo ppo --agents 3 --model hard/ppo/3-ppo `
    --grid-size 7 `
    --max-steps 30 `
    --frames-per-proc 256 `
    --frames 250000 `
    --capture-interval 15

python -m Coloring.scripts.train --algo ppo --model hard/ppo/2-ppo-sm-goal-no-reset `
    --market sm-goal-no-reset `
    --trading-fee 0.1 `
    --grid-size 7 `
    --max-steps 30 `
    --frames-per-proc 256 `
    --frames 250000 `
    --capture-interval 15

#----------------------------------------------------------------------------------------
# Mixed 3 PPO Settings
#----------------------------------------------------------------------------------------

python -m Coloring.scripts.train --algo ppo --agents 3 --model hard\\3-ppo-mixed `
    --setting mixed-motive `
    --grid-size 7 `
    --frames-per-proc 256 `
    --frames 250000 `
    --capture-interval 15