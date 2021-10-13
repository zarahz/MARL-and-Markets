# set-executionpolicy remotesigned to remove trust privileges
Set-Location -Path "C:\Users\Zarah\Documents\workspace\MA\"

& C:/Users/Zarah/.virtualenvs/Coloring_with_CAP-xNNGJax5/Scripts/Activate.ps1

#EASY
# & .\Coloring\scripts\powershell\easy\1-ppo.ps1
# & .\Coloring\scripts\powershell\easy\2-ppo.ps1
# & .\Coloring\scripts\powershell\easy\2-ppo-mixed.ps1
# & .\Coloring\scripts\powershell\easy\2-ppo-mixed-competitive.ps1
# & .\Coloring\scripts\powershell\easy\1-dqn.ps1
& .\Coloring\scripts\powershell\easy\2-dqn.ps1
# CONTINUE HERE
& .\Coloring\scripts\powershell\easy\2-dqn-mixed.ps1
& .\Coloring\scripts\powershell\easy\2-dqn-mixed-competitive.ps1

#TODO call other scripts!
# & .\Coloring\scripts\powershell\easy\3-ppo-percentage.ps1
# & .\Coloring\scripts\powershell\easy\3-ppo-rooms.ps1
# & .\Coloring\scripts\powershell\easy\dqn_comparisons.ps1


# HARD
# & .\Coloring\scripts\powershell\hard\1-ppo.ps1
# & .\Coloring\scripts\powershell\hard\1-dqn.ps1
# & .\Coloring\scripts\powershell\hard\3-ppo.ps1
# & .\Coloring\scripts\powershell\hard\3-ppo-mixed.ps1
# & .\Coloring\scripts\powershell\hard\3-ppo-mixed-competitive.ps1
#TODO call other scripts!
# & .\Coloring\scripts\powershell\hard\3-ppo-percentage.ps1
# & .\Coloring\scripts\powershell\hard\3-ppo-rooms.ps1

Set-Location -Path "C:\Users\Zarah\Documents\workspace\MA\Coloring\scripts\"

# ------------------------------------------------------------ PPO --------------------------------------------------------------- #
# # ------------- 6 Agents ------------- #
# pipenv run python -m Coloring.scripts.train --agents 6 --model 6-ppo-rooms --env FourRooms-Grid-v0 --grid-size 9 --capture-interval 30 --save-interval 10 --frames-per-proc 1024 --frames 1500000
# pipenv run python -m Coloring.scripts.train --agents 6 --model 6-ppo-rooms-percentage --setting percentage-reward --env FourRooms-Grid-v0 --grid-size 9 --capture-interval 30 --save-interval 10 --frames-per-proc 1024 --frames 1500000
# pipenv run python -m Coloring.scripts.train --agents 6 --model 6-ppo-rooms-mixed --setting mixed-motive --env FourRooms-Grid-v0 --grid-size 9 --capture-interval 30 --save-interval 10 --frames-per-proc 1024 --frames 1500000
# pipenv run python -m Coloring.scripts.train --agents 6 --model 6-ppo-rooms-mixed-competitive --setting mixed-motive-competitive --env FourRooms-Grid-v0 --grid-size 9 --capture-interval 30 --save-interval 10 --frames-per-proc 1024 --frames 1500000

# # sm market
# pipenv run python -m Coloring.scripts.train --agents 6 --model 6-ppo-rooms-sm --market sm --env FourRooms-Grid-v0 --grid-size 9 --capture-interval 30 --save-interval 10 --frames-per-proc 1024 --frames 1500000
# pipenv run python -m Coloring.scripts.train --agents 6 --model 6-ppo-rooms-sm-goal --market sm-goal --env FourRooms-Grid-v0 --grid-size 9 --capture-interval 30 --save-interval 10 --frames-per-proc 1024 --frames 1500000
# pipenv run python -m Coloring.scripts.train --agents 6 --model 6-ppo-rooms-sm-goal-no-reset --market sm-goal-no-reset --env FourRooms-Grid-v0 --grid-size 9 --capture-interval 30 --save-interval 10 --frames-per-proc 1024 --frames 1500000

# pipenv run python -m Coloring.scripts.train --agents 6 --model 6-ppo-rooms-percentage-sm --setting percentage-reward --market sm --env FourRooms-Grid-v0 --grid-size 9 --capture-interval 30 --save-interval 10 --frames-per-proc 1024 --frames 1500000
# pipenv run python -m Coloring.scripts.train --agents 6 --model 6-ppo-rooms-percentage-sm-goal --setting percentage-reward --market sm-goal --env FourRooms-Grid-v0 --grid-size 9 --capture-interval 30 --save-interval 10 --frames-per-proc 1024 --frames 1500000
# pipenv run python -m Coloring.scripts.train --agents 6 --model 6-ppo-rooms-percentage-sm-goal-no-reset --setting percentage-reward --market sm-goal-no-reset --env FourRooms-Grid-v0 --grid-size 9 --capture-interval 30 --save-interval 10 --frames-per-proc 1024 --frames 1500000

# pipenv run python -m Coloring.scripts.train --agents 6 --model 6-ppo-rooms-mixed-sm --setting mixed-motive --market sm --env FourRooms-Grid-v0 --grid-size 9 --capture-interval 30 --save-interval 10 --frames-per-proc 1024 --frames 1500000
# pipenv run python -m Coloring.scripts.train --agents 6 --model 6-ppo-rooms-mixed-sm-goal --setting mixed-motive --market sm-goal --env FourRooms-Grid-v0 --grid-size 9 --capture-interval 30 --save-interval 10 --frames-per-proc 1024 --frames 1500000
# pipenv run python -m Coloring.scripts.train --agents 6 --model 6-ppo-rooms-mixed-sm-goal-no-reset --setting mixed-motive --market sm-goal-no-reset --env FourRooms-Grid-v0 --grid-size 9 --capture-interval 30 --save-interval 10 --frames-per-proc 1024 --frames 1500000

# pipenv run python -m Coloring.scripts.train --agents 6 --model 6-ppo-rooms-mixed-competitive-sm --setting mixed-motive-competitive --market sm --env FourRooms-Grid-v0 --grid-size 9 --capture-interval 30 --save-interval 10 --frames-per-proc 1024 --frames 1500000
# pipenv run python -m Coloring.scripts.train --agents 6 --model 6-ppo-rooms-mixed-competitive-sm-goal --setting mixed-motive-competitive --market sm-goal --env FourRooms-Grid-v0 --grid-size 9 --capture-interval 30 --save-interval 10 --frames-per-proc 1024 --frames 1500000
# pipenv run python -m Coloring.scripts.train --agents 6 --model 6-ppo-rooms-mixed-competitive-sm-goal-no-reset --setting mixed-motive-competitive --market sm-goal-no-reset --env FourRooms-Grid-v0 --grid-size 9 --capture-interval 30 --save-interval 10 --frames-per-proc 1024 --frames 1500000

# # am market
# pipenv run python -m Coloring.scripts.train --agents 6 --model 6-ppo-rooms-am --market am --env FourRooms-Grid-v0 --grid-size 9 --capture-interval 30 --save-interval 10 --frames-per-proc 1024 --frames 1500000
# pipenv run python -m Coloring.scripts.train --agents 6 --model 6-ppo-rooms-am-goal --market am-goal --env FourRooms-Grid-v0 --grid-size 9 --capture-interval 30 --save-interval 10 --frames-per-proc 1024 --frames 1500000
# pipenv run python -m Coloring.scripts.train --agents 6 --model 6-ppo-rooms-am-goal-no-reset --market am-goal-no-reset --env FourRooms-Grid-v0 --grid-size 9 --capture-interval 30 --save-interval 10 --frames-per-proc 1024 --frames 1500000

# pipenv run python -m Coloring.scripts.train --agents 6 --model 6-ppo-rooms-percentage-am --setting percentage-reward --market am --env FourRooms-Grid-v0 --grid-size 9 --capture-interval 30 --save-interval 10 --frames-per-proc 1024 --frames 1500000
# pipenv run python -m Coloring.scripts.train --agents 6 --model 6-ppo-rooms-percentage-am-goal --setting percentage-reward --market am-goal --env FourRooms-Grid-v0 --grid-size 9 --capture-interval 30 --save-interval 10 --frames-per-proc 1024 --frames 1500000
# pipenv run python -m Coloring.scripts.train --agents 6 --model 6-ppo-rooms-percentage-am-goal-no-reset --setting percentage-reward --market am-goal-no-reset --env FourRooms-Grid-v0 --grid-size 9 --capture-interval 30 --save-interval 10 --frames-per-proc 1024 --frames 1500000

# pipenv run python -m Coloring.scripts.train --agents 6 --model 6-ppo-rooms-mixed-am --setting mixed-motive --market am --env FourRooms-Grid-v0 --grid-size 9 --capture-interval 30 --save-interval 10 --frames-per-proc 1024 --frames 1500000
# pipenv run python -m Coloring.scripts.train --agents 6 --model 6-ppo-rooms-mixed-am-goal --setting mixed-motive --market am-goal --env FourRooms-Grid-v0 --grid-size 9 --capture-interval 30 --save-interval 10 --frames-per-proc 1024 --frames 1500000
# pipenv run python -m Coloring.scripts.train --agents 6 --model 6-ppo-rooms-mixed-am-goal-no-reset --setting mixed-motive --market am-goal-no-reset --env FourRooms-Grid-v0 --grid-size 9 --capture-interval 30 --save-interval 10 --frames-per-proc 1024 --frames 1500000

# pipenv run python -m Coloring.scripts.train --agents 6 --model 6-ppo-rooms-mixed-competitive-am --setting mixed-motive-competitive --market am --env FourRooms-Grid-v0 --grid-size 9 --capture-interval 30 --save-interval 10 --frames-per-proc 1024 --frames 1500000
# pipenv run python -m Coloring.scripts.train --agents 6 --model 6-ppo-rooms-mixed-competitive-am-goal --setting mixed-motive-competitive --market am-goal --env FourRooms-Grid-v0 --grid-size 9 --capture-interval 30 --save-interval 10 --frames-per-proc 1024 --frames 1500000
# pipenv run python -m Coloring.scripts.train --agents 6 --model 6-ppo-rooms-mixed-competitive-am-goal-no-reset --setting mixed-motive-competitive --market am-goal-no-reset --env FourRooms-Grid-v0 --grid-size 9 --capture-interval 30 --save-interval 10 --frames-per-proc 1024 --frames 1500000