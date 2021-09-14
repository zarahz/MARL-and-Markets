# set-executionpolicy remotesigned to remove trust privileges
Set-Location -Path "C:\Users\Zarah\Documents\workspace\MA\"

& C:/Users/Zarah/.virtualenvs/Coloring_with_CAP-xNNGJax5/Scripts/Activate.ps1

# ------------- 1 Agent ------------- #
pipenv run python -m Coloring_with_CAP.scripts.train --agents 1 --model 1-ppo `
    --grid-size 9 `
    --max-steps 300 `
    --capture-interval 20 `
    --save-interval 10 `
    --frames-per-proc 1024 `
    --frames 1000000
pipenv run python -m Coloring_with_CAP.scripts.train --agents 1 --model 1-ppo-rooms `
    --env FourRooms-Grid-v0 `
    --grid-size 9 `
    --max-steps 400 `
    --capture-interval 20 `
    --save-interval 10 `
    --frames-per-proc 1024 `
    --frames 1500000

# ------------- 3 Agents ------------- #
pipenv run python -m Coloring_with_CAP.scripts.train --agents 3 --model 3-ppo `
    --grid-size 9 `
    --max-steps 350 `
    --capture-interval 20 `
    --save-interval 10 `
    --frames-per-proc 1024 `
    --frames 1000000
pipenv run python -m Coloring_with_CAP.scripts.train --agents 3 --model 3-ppo-percentage `
    --setting percentage-reward `
    --grid-size 9 `
    --max-steps 350 `
    --capture-interval 20 `
    --save-interval 10 `
    --frames-per-proc 1024 `
    --frames 1000000
pipenv run python -m Coloring_with_CAP.scripts.train --agents 3 --model 3-ppo-mixed `
    --setting mixed-motive `
    --grid-size 9 `
    --max-steps 350 `
    --capture-interval 20 `
    --save-interval 10 `
    --frames-per-proc 1024 `
    --frames 1000000
pipenv run python -m Coloring_with_CAP.scripts.train --agents 3 --model 3-ppo-mixed-competitive `
    --setting mixed-motive-competitive `
    --grid-size 9 `
    --max-steps 350 `
    --capture-interval 20 `
    --save-interval 10 `
    --frames-per-proc 1024 `
    --frames 1000000
# Hard mode (room env)
pipenv run python -m Coloring_with_CAP.scripts.train --agents 3 --model 3-ppo-rooms-percentage `
    --setting percentage-reward `
    --env FourRooms-Grid-v0 `
    --grid-size 9 `
    --max-steps 400 `
    --capture-interval 20 `
    --save-interval 10 `
    --frames-per-proc 1024 `
    --frames 1500000


# ------------- 3 Agents SHAREHOLDER market ------------- #
pipenv run python -m Coloring_with_CAP.scripts.train --agents 3 --model 3-ppo-sm `
    --market sm `
    --grid-size 9 `
    --max-steps 350 `
    --capture-interval 20 `
    --save-interval 10 `
    --frames-per-proc 1024 `
    --frames 800000 
pipenv run python -m Coloring_with_CAP.scripts.train --agents 3 --model 3-ppo-sm-goal `
    --market sm-goal `
    --grid-size 9 `
    --max-steps 350 `
    --capture-interval 20 `
    --save-interval 10 `
    --frames-per-proc 1024 `
    --frames 800000 
pipenv run python -m Coloring_with_CAP.scripts.train --agents 3 --model 3-ppo-sm-goal-no-reset `
    --market sm-goal-no-reset `
    --grid-size 9 `
    --max-steps 350 `
    --capture-interval 20 `
    --save-interval 10 `
    --frames-per-proc 1024 `
    --frames 800000 

# --- 3 agents sm + percentage
pipenv run python -m Coloring_with_CAP.scripts.train --agents 3 --model 3-ppo-percentage-sm `
    --setting percentage-reward `
    --market sm `
    --grid-size 9 `
    --max-steps 350 `
    --capture-interval 20 `
    --save-interval 10 `
    --frames-per-proc 1024 `
    --frames 800000 
pipenv run python -m Coloring_with_CAP.scripts.train --agents 3 --model 3-ppo-percentage-sm-goal `
    --setting percentage-reward `
    --market sm-goal `
    --grid-size 9 `
    --max-steps 350 `
    --capture-interval 20 `
    --save-interval 10 `
    --frames-per-proc 1024 `
    --frames 800000 
pipenv run python -m Coloring_with_CAP.scripts.train --agents 3 --model 3-ppo-percentage-sm-goal-no-reset `
    --setting percentage-reward `
    --market sm-goal-no-reset `
    --grid-size 9 `
    --max-steps 350 `
    --capture-interval 20 `
    --save-interval 10 `
    --frames-per-proc 1024 `
    --frames 800000 
# 3 agents sm + percentage HARD MODE
pipenv run python -m Coloring_with_CAP.scripts.train --agents 3 --model 3-ppo-rooms-percentage-sm `
    --setting percentage-reward `
    --market sm `
    --env FourRooms-Grid-v0 `
    --grid-size 9 `
    --max-steps 400 `
    --capture-interval 20 `
    --save-interval 10 `
    --frames-per-proc 1024 `
    --frames 1500000
pipenv run python -m Coloring_with_CAP.scripts.train --agents 3 --model 3-ppo-rooms-percentage-sm-goal `
    --setting percentage-reward `
    --market sm-goal `
    --env FourRooms-Grid-v0 `
    --grid-size 9 `
    --max-steps 400 `
    --capture-interval 20 `
    --save-interval 10 `
    --frames-per-proc 1024 `
    --frames 1500000
pipenv run python -m Coloring_with_CAP.scripts.train --agents 3 --model 3-ppo-rooms-percentage-sm-goal-no-reset `
    --setting percentage-reward `
    --market sm-goal-no-reset `
    --env FourRooms-Grid-v0 `
    --grid-size 9 `
    --max-steps 400 `
    --capture-interval 20 `
    --save-interval 10 `
    --frames-per-proc 1024 `
    --frames 1500000
# --- 3 agents sm + mixed
pipenv run python -m Coloring_with_CAP.scripts.train --agents 3 --model 3-ppo-mixed-sm `
    --setting mixed-motive `
    --market sm `
    --grid-size 9 `
    --max-steps 350 `
    --capture-interval 20 `
    --save-interval 10 `
    --frames-per-proc 1024 `
    --frames 800000 
pipenv run python -m Coloring_with_CAP.scripts.train --agents 3 --model 3-ppo-mixed-sm-goal `
    --setting mixed-motive `
    --market sm-goal `
    --grid-size 9 `
    --max-steps 350 `
    --capture-interval 20 `
    --save-interval 10 `
    --frames-per-proc 1024 `
    --frames 800000 
pipenv run python -m Coloring_with_CAP.scripts.train --agents 3 --model 3-ppo-mixed-sm-goal-no-reset `
    --setting mixed-motive `
    --market sm-goal-no-reset `
    --grid-size 9 `
    --max-steps 350 `
    --capture-interval 20 `
    --save-interval 10 `
    --frames-per-proc 1024 `
    --frames 800000 

# --- 3 agents sm + mixed competitive
pipenv run python -m Coloring_with_CAP.scripts.train --agents 3 --model 3-ppo-mixed-competitive-sm `
    --setting mixed-motive-competitive `
    --market sm `
    --grid-size 9 `
    --max-steps 350 `
    --capture-interval 20 `
    --save-interval 10 `
    --frames-per-proc 1024 `
    --frames 800000 
pipenv run python -m Coloring_with_CAP.scripts.train --agents 3 --model 3-ppo-mixed-competitive-sm-goal `
    --setting mixed-motive-competitive `
    --market sm-goal `
    --grid-size 9 `
    --max-steps 350 `
    --capture-interval 20 `
    --save-interval 10 `
    --frames-per-proc 1024 `
    --frames 800000 
pipenv run python -m Coloring_with_CAP.scripts.train --agents 3 --model 3-ppo-mixed-competitive-sm-goal-no-reset `
    --setting mixed-motive-competitive `
    --market sm-goal-no-reset `
    --grid-size 9 `
    --max-steps 350 `
    --capture-interval 20 `
    --save-interval 10 `
    --frames-per-proc 1024 `
    --frames 800000 


# ------------- 3 Agents ACTION market ------------- #
pipenv run python -m Coloring_with_CAP.scripts.train --agents 3 --model 3-ppo-am `
    --market am `
    --grid-size 9 `
    --max-steps 350 `
    --capture-interval 20 `
    --save-interval 10 `
    --frames-per-proc 1024 `
    --frames 800000 
pipenv run python -m Coloring_with_CAP.scripts.train --agents 3 --model 3-ppo-am-goal `
    --market am-goal `
    --grid-size 9 `
    --max-steps 350 `
    --capture-interval 20 `
    --save-interval 10 `
    --frames-per-proc 1024 `
    --frames 800000 
pipenv run python -m Coloring_with_CAP.scripts.train --agents 3 --model 3-ppo-am-goal-no-reset `
    --market am-goal-no-reset `
    --grid-size 9 `
    --max-steps 350 `
    --capture-interval 20 `
    --save-interval 10 `
    --frames-per-proc 1024 `
    --frames 800000 

# --- 3 agents am + percentage
pipenv run python -m Coloring_with_CAP.scripts.train --agents 3 --model 3-ppo-percentage-am `
    --setting percentage-reward `
    --market am `
    --grid-size 9 `
    --max-steps 350 `
    --capture-interval 20 `
    --save-interval 10 `
    --frames-per-proc 1024 `
    --frames 800000 
pipenv run python -m Coloring_with_CAP.scripts.train --agents 3 --model 3-ppo-percentage-am-goal `
    --setting percentage-reward `
    --market am-goal `
    --grid-size 9 `
    --max-steps 350 `
    --capture-interval 20 `
    --save-interval 10 `
    --frames-per-proc 1024 `
    --frames 800000 
pipenv run python -m Coloring_with_CAP.scripts.train --agents 3 --model 3-ppo-percentage-am-goal-no-reset `
    --setting percentage-reward `
    --market am-goal-no-reset `
    --grid-size 9 `
    --max-steps 350 `
    --capture-interval 20 `
    --save-interval 10 `
    --frames-per-proc 1024 `
    --frames 800000 
# 3 agents am + percentage HARD MODE
pipenv run python -m Coloring_with_CAP.scripts.train --agents 3 --model 3-ppo-rooms-percentage-am `
    --setting percentage-reward `
    --market am `
    --env FourRooms-Grid-v0 `
    --grid-size 9 `
    --max-steps 400 `
    --capture-interval 20 `
    --save-interval 10 `
    --frames-per-proc 1024 `
    --frames 1500000
pipenv run python -m Coloring_with_CAP.scripts.train --agents 3 --model 3-ppo-rooms-percentage-am-goal `
    --setting percentage-reward `
    --market am-goal `
    --env FourRooms-Grid-v0 `
    --grid-size 9 `
    --max-steps 400 `
    --capture-interval 20 `
    --save-interval 10 `
    --frames-per-proc 1024 `
    --frames 1500000
pipenv run python -m Coloring_with_CAP.scripts.train --agents 3 --model 3-ppo-rooms-percentage-am-goal-no-reset `
    --setting percentage-reward `
    --market am-goal-no-reset `
    --env FourRooms-Grid-v0 `
    --grid-size 9 `
    --max-steps 400 `
    --capture-interval 20 `
    --save-interval 10 `
    --frames-per-proc 1024 `
    --frames 1500000

# --- 3 agents am + mixed
pipenv run python -m Coloring_with_CAP.scripts.train --agents 3 --model 3-ppo-mixed-am `
    --setting mixed-motive `
    --market am `
    --grid-size 9 `
    --max-steps 350 `
    --capture-interval 20 `
    --save-interval 10 `
    --frames-per-proc 1024 `
    --frames 800000 
pipenv run python -m Coloring_with_CAP.scripts.train --agents 3 --model 3-ppo-mixed-am-goal `
    --setting mixed-motive `
    --market am-goal `
    --grid-size 9 `
    --max-steps 350 `
    --capture-interval 20 `
    --save-interval 10 `
    --frames-per-proc 1024 `
    --frames 800000 
pipenv run python -m Coloring_with_CAP.scripts.train --agents 3 --model 3-ppo-mixed-am-goal-no-reset `
    --setting mixed-motive `
    --market am-goal-no-reset `
    --grid-size 9 `
    --max-steps 350 `
    --capture-interval 20 `
    --save-interval 10 `
    --frames-per-proc 1024 `
    --frames 800000 

# --- 3 agents am + competitive
pipenv run python -m Coloring_with_CAP.scripts.train --agents 3 --model 3-ppo-mixed-competitive-am `
    --setting mixed-motive-competitive `
    --market am `
    --grid-size 9 `
    --max-steps 350 `
    --capture-interval 20 `
    --save-interval 10 `
    --frames-per-proc 1024 `
    --frames 800000 
pipenv run python -m Coloring_with_CAP.scripts.train --agents 3 --model 3-ppo-mixed-competitive-am-goal `
    --setting mixed-motive-competitive `
    --market am-goal `
    --grid-size 9 `
    --max-steps 350 `
    --capture-interval 20 `
    --save-interval 10 `
    --frames-per-proc 1024 `
    --frames 800000 
pipenv run python -m Coloring_with_CAP.scripts.train --agents 3 --model 3-ppo-mixed-competitive-am-goal-no-reset `
    --setting mixed-motive-competitive `
    --market am-goal-no-reset `
    --grid-size 9 `
    --max-steps 350 `
    --capture-interval 20 `
    --save-interval 10 `
    --frames-per-proc 1024 `
    --frames 800000 


# # ------------- 6 Agents ------------- #
# pipenv run python -m Coloring_with_CAP.scripts.train --agents 6 --model 6-ppo-rooms --env FourRooms-Grid-v0 --grid-size 9 --capture-interval 30 --save-interval 10 --frames-per-proc 1024 --frames 1500000
# pipenv run python -m Coloring_with_CAP.scripts.train --agents 6 --model 6-ppo-rooms-percentage --setting percentage-reward --env FourRooms-Grid-v0 --grid-size 9 --capture-interval 30 --save-interval 10 --frames-per-proc 1024 --frames 1500000
# pipenv run python -m Coloring_with_CAP.scripts.train --agents 6 --model 6-ppo-rooms-mixed --setting mixed-motive --env FourRooms-Grid-v0 --grid-size 9 --capture-interval 30 --save-interval 10 --frames-per-proc 1024 --frames 1500000
# pipenv run python -m Coloring_with_CAP.scripts.train --agents 6 --model 6-ppo-rooms-mixed-competitive --setting mixed-motive-competitive --env FourRooms-Grid-v0 --grid-size 9 --capture-interval 30 --save-interval 10 --frames-per-proc 1024 --frames 1500000

# # sm market
# pipenv run python -m Coloring_with_CAP.scripts.train --agents 6 --model 6-ppo-rooms-sm --market sm --env FourRooms-Grid-v0 --grid-size 9 --capture-interval 30 --save-interval 10 --frames-per-proc 1024 --frames 1500000
# pipenv run python -m Coloring_with_CAP.scripts.train --agents 6 --model 6-ppo-rooms-sm-goal --market sm-goal --env FourRooms-Grid-v0 --grid-size 9 --capture-interval 30 --save-interval 10 --frames-per-proc 1024 --frames 1500000
# pipenv run python -m Coloring_with_CAP.scripts.train --agents 6 --model 6-ppo-rooms-sm-goal-no-reset --market sm-goal-no-reset --env FourRooms-Grid-v0 --grid-size 9 --capture-interval 30 --save-interval 10 --frames-per-proc 1024 --frames 1500000

# pipenv run python -m Coloring_with_CAP.scripts.train --agents 6 --model 6-ppo-rooms-percentage-sm --setting percentage-reward --market sm --env FourRooms-Grid-v0 --grid-size 9 --capture-interval 30 --save-interval 10 --frames-per-proc 1024 --frames 1500000
# pipenv run python -m Coloring_with_CAP.scripts.train --agents 6 --model 6-ppo-rooms-percentage-sm-goal --setting percentage-reward --market sm-goal --env FourRooms-Grid-v0 --grid-size 9 --capture-interval 30 --save-interval 10 --frames-per-proc 1024 --frames 1500000
# pipenv run python -m Coloring_with_CAP.scripts.train --agents 6 --model 6-ppo-rooms-percentage-sm-goal-no-reset --setting percentage-reward --market sm-goal-no-reset --env FourRooms-Grid-v0 --grid-size 9 --capture-interval 30 --save-interval 10 --frames-per-proc 1024 --frames 1500000

# pipenv run python -m Coloring_with_CAP.scripts.train --agents 6 --model 6-ppo-rooms-mixed-sm --setting mixed-motive --market sm --env FourRooms-Grid-v0 --grid-size 9 --capture-interval 30 --save-interval 10 --frames-per-proc 1024 --frames 1500000
# pipenv run python -m Coloring_with_CAP.scripts.train --agents 6 --model 6-ppo-rooms-mixed-sm-goal --setting mixed-motive --market sm-goal --env FourRooms-Grid-v0 --grid-size 9 --capture-interval 30 --save-interval 10 --frames-per-proc 1024 --frames 1500000
# pipenv run python -m Coloring_with_CAP.scripts.train --agents 6 --model 6-ppo-rooms-mixed-sm-goal-no-reset --setting mixed-motive --market sm-goal-no-reset --env FourRooms-Grid-v0 --grid-size 9 --capture-interval 30 --save-interval 10 --frames-per-proc 1024 --frames 1500000

# pipenv run python -m Coloring_with_CAP.scripts.train --agents 6 --model 6-ppo-rooms-mixed-competitive-sm --setting mixed-motive-competitive --market sm --env FourRooms-Grid-v0 --grid-size 9 --capture-interval 30 --save-interval 10 --frames-per-proc 1024 --frames 1500000
# pipenv run python -m Coloring_with_CAP.scripts.train --agents 6 --model 6-ppo-rooms-mixed-competitive-sm-goal --setting mixed-motive-competitive --market sm-goal --env FourRooms-Grid-v0 --grid-size 9 --capture-interval 30 --save-interval 10 --frames-per-proc 1024 --frames 1500000
# pipenv run python -m Coloring_with_CAP.scripts.train --agents 6 --model 6-ppo-rooms-mixed-competitive-sm-goal-no-reset --setting mixed-motive-competitive --market sm-goal-no-reset --env FourRooms-Grid-v0 --grid-size 9 --capture-interval 30 --save-interval 10 --frames-per-proc 1024 --frames 1500000

# # am market
# pipenv run python -m Coloring_with_CAP.scripts.train --agents 6 --model 6-ppo-rooms-am --market am --env FourRooms-Grid-v0 --grid-size 9 --capture-interval 30 --save-interval 10 --frames-per-proc 1024 --frames 1500000
# pipenv run python -m Coloring_with_CAP.scripts.train --agents 6 --model 6-ppo-rooms-am-goal --market am-goal --env FourRooms-Grid-v0 --grid-size 9 --capture-interval 30 --save-interval 10 --frames-per-proc 1024 --frames 1500000
# pipenv run python -m Coloring_with_CAP.scripts.train --agents 6 --model 6-ppo-rooms-am-goal-no-reset --market am-goal-no-reset --env FourRooms-Grid-v0 --grid-size 9 --capture-interval 30 --save-interval 10 --frames-per-proc 1024 --frames 1500000

# pipenv run python -m Coloring_with_CAP.scripts.train --agents 6 --model 6-ppo-rooms-percentage-am --setting percentage-reward --market am --env FourRooms-Grid-v0 --grid-size 9 --capture-interval 30 --save-interval 10 --frames-per-proc 1024 --frames 1500000
# pipenv run python -m Coloring_with_CAP.scripts.train --agents 6 --model 6-ppo-rooms-percentage-am-goal --setting percentage-reward --market am-goal --env FourRooms-Grid-v0 --grid-size 9 --capture-interval 30 --save-interval 10 --frames-per-proc 1024 --frames 1500000
# pipenv run python -m Coloring_with_CAP.scripts.train --agents 6 --model 6-ppo-rooms-percentage-am-goal-no-reset --setting percentage-reward --market am-goal-no-reset --env FourRooms-Grid-v0 --grid-size 9 --capture-interval 30 --save-interval 10 --frames-per-proc 1024 --frames 1500000

# pipenv run python -m Coloring_with_CAP.scripts.train --agents 6 --model 6-ppo-rooms-mixed-am --setting mixed-motive --market am --env FourRooms-Grid-v0 --grid-size 9 --capture-interval 30 --save-interval 10 --frames-per-proc 1024 --frames 1500000
# pipenv run python -m Coloring_with_CAP.scripts.train --agents 6 --model 6-ppo-rooms-mixed-am-goal --setting mixed-motive --market am-goal --env FourRooms-Grid-v0 --grid-size 9 --capture-interval 30 --save-interval 10 --frames-per-proc 1024 --frames 1500000
# pipenv run python -m Coloring_with_CAP.scripts.train --agents 6 --model 6-ppo-rooms-mixed-am-goal-no-reset --setting mixed-motive --market am-goal-no-reset --env FourRooms-Grid-v0 --grid-size 9 --capture-interval 30 --save-interval 10 --frames-per-proc 1024 --frames 1500000

# pipenv run python -m Coloring_with_CAP.scripts.train --agents 6 --model 6-ppo-rooms-mixed-competitive-am --setting mixed-motive-competitive --market am --env FourRooms-Grid-v0 --grid-size 9 --capture-interval 30 --save-interval 10 --frames-per-proc 1024 --frames 1500000
# pipenv run python -m Coloring_with_CAP.scripts.train --agents 6 --model 6-ppo-rooms-mixed-competitive-am-goal --setting mixed-motive-competitive --market am-goal --env FourRooms-Grid-v0 --grid-size 9 --capture-interval 30 --save-interval 10 --frames-per-proc 1024 --frames 1500000
# pipenv run python -m Coloring_with_CAP.scripts.train --agents 6 --model 6-ppo-rooms-mixed-competitive-am-goal-no-reset --setting mixed-motive-competitive --market am-goal-no-reset --env FourRooms-Grid-v0 --grid-size 9 --capture-interval 30 --save-interval 10 --frames-per-proc 1024 --frames 1500000

Set-Location -Path "C:\Users\Zarah\Documents\workspace\MA\Coloring_with_CAP\scripts\"