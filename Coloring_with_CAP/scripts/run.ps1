# set-executionpolicy remotesigned to remove trust privileges
Set-Location -Path "C:\Users\Zarah\Documents\workspace\MA\"

& C:/Users/Zarah/.virtualenvs/Coloring_with_CAP-xNNGJax5/Scripts/Activate.ps1

# pipenv run python -m Coloring_with_CAP.scripts.train --agents 1 --model one-agent --save-interval 10 --frames 80000 

# 3 Agents
pipenv run python -m Coloring_with_CAP.scripts.train --agents 3 --model 3-ppo --grid-size 10 --capture-interval 200 --capture-frames 30 --save-interval 10 --frames 600000
pipenv run python -m Coloring_with_CAP.scripts.train --agents 3 --model 3-ppo-percentage --setting percentage-reward --grid-size 10 --capture-interval 200 --capture-frames 30 --save-interval 10 --frames 600000
pipenv run python -m Coloring_with_CAP.scripts.train --agents 3 --model 3-ppo-mixed --setting mixed-motive --grid-size 10 --capture-interval 200 --capture-frames 30 --save-interval 10 --frames 600000
pipenv run python -m Coloring_with_CAP.scripts.train --agents 3 --model 3-ppo-mixed-competitive --setting mixed-motive-competitive --grid-size 10 --capture-interval 200 --capture-frames 30 --save-interval 10 --frames 600000

# 3 Agents + sm market
pipenv run python -m Coloring_with_CAP.scripts.train --agents 3 --model 3-ppo-sm --market sm --grid-size 10 --capture-interval 200 --capture-frames 30 --save-interval 10 --frames 600000
pipenv run python -m Coloring_with_CAP.scripts.train --agents 3 --model 3-ppo-sm-goal --market sm-goal --grid-size 10 --capture-interval 200 --capture-frames 30 --save-interval 10 --frames 600000
pipenv run python -m Coloring_with_CAP.scripts.train --agents 3 --model 3-ppo-sm-goal-no-reset --market sm-goal-no-reset --grid-size 10 --capture-interval 200 --capture-frames 30 --save-interval 10 --frames 600000

pipenv run python -m Coloring_with_CAP.scripts.train --agents 3 --model 3-ppo-percentage-sm --setting percentage-reward --market sm --grid-size 10 --capture-interval 200 --capture-frames 30 --save-interval 10 --frames 600000
pipenv run python -m Coloring_with_CAP.scripts.train --agents 3 --model 3-ppo-percentage-sm-goal --setting percentage-reward --market sm-goal --grid-size 10 --capture-interval 200 --capture-frames 30 --save-interval 10 --frames 600000
pipenv run python -m Coloring_with_CAP.scripts.train --agents 3 --model 3-ppo-percentage-sm-goal-no-reset --setting percentage-reward --market sm-goal-no-reset --grid-size 10 --capture-interval 200 --capture-frames 30 --save-interval 10 --frames 600000

pipenv run python -m Coloring_with_CAP.scripts.train --agents 3 --model 3-ppo-mixed-sm --setting mixed-motive --market sm --grid-size 10 --capture-interval 200 --capture-frames 30 --save-interval 10 --frames 600000
pipenv run python -m Coloring_with_CAP.scripts.train --agents 3 --model 3-ppo-mixed-sm-goal --setting mixed-motive --market sm-goal --grid-size 10 --capture-interval 200 --capture-frames 30 --save-interval 10 --frames 600000
pipenv run python -m Coloring_with_CAP.scripts.train --agents 3 --model 3-ppo-mixed-sm-goal-no-reset --setting mixed-motive --market sm-goal-no-reset --grid-size 10 --capture-interval 200 --capture-frames 30 --save-interval 10 --frames 600000

pipenv run python -m Coloring_with_CAP.scripts.train --agents 3 --model 3-ppo-mixed-competitive-sm --setting mixed-motive-competitive --market sm --grid-size 10 --capture-interval 200 --capture-frames 30 --save-interval 10 --frames 600000
pipenv run python -m Coloring_with_CAP.scripts.train --agents 3 --model 3-ppo-mixed-competitive-sm-goal --setting mixed-motive-competitive --market sm-goal --grid-size 10 --capture-interval 200 --capture-frames 30 --save-interval 10 --frames 600000
pipenv run python -m Coloring_with_CAP.scripts.train --agents 3 --model 3-ppo-mixed-competitive-sm-goal-no-reset --setting mixed-motive-competitive --market sm-goal-no-reset --grid-size 10 --capture-interval 200 --capture-frames 30 --save-interval 10 --frames 600000

# 3 Agents + am market
pipenv run python -m Coloring_with_CAP.scripts.train --agents 3 --model 3-ppo-am --market am --grid-size 10 --capture-interval 200 --capture-frames 30 --save-interval 10 --frames 600000
pipenv run python -m Coloring_with_CAP.scripts.train --agents 3 --model 3-ppo-am-goal --market am-goal --grid-size 10 --capture-interval 200 --capture-frames 30 --save-interval 10 --frames 600000
pipenv run python -m Coloring_with_CAP.scripts.train --agents 3 --model 3-ppo-am-goal-no-reset --market am-goal-no-reset --grid-size 10 --capture-interval 200 --capture-frames 30 --save-interval 10 --frames 600000

pipenv run python -m Coloring_with_CAP.scripts.train --agents 3 --model 3-ppo-percentage-am --setting percentage-reward --market am --grid-size 10 --capture-interval 200 --capture-frames 30 --save-interval 10 --frames 600000
pipenv run python -m Coloring_with_CAP.scripts.train --agents 3 --model 3-ppo-percentage-am-goal --setting percentage-reward --market am-goal --grid-size 10 --capture-interval 200 --capture-frames 30 --save-interval 10 --frames 600000
pipenv run python -m Coloring_with_CAP.scripts.train --agents 3 --model 3-ppo-percentage-am-goal-no-reset --setting percentage-reward --market am-goal-no-reset --grid-size 10 --capture-interval 200 --capture-frames 30 --save-interval 10 --frames 600000

pipenv run python -m Coloring_with_CAP.scripts.train --agents 3 --model 3-ppo-mixed-am --setting mixed-motive --market am --grid-size 10 --capture-interval 200 --capture-frames 30 --save-interval 10 --frames 600000
pipenv run python -m Coloring_with_CAP.scripts.train --agents 3 --model 3-ppo-mixed-am-goal --setting mixed-motive --market am-goal --grid-size 10 --capture-interval 200 --capture-frames 30 --save-interval 10 --frames 600000
pipenv run python -m Coloring_with_CAP.scripts.train --agents 3 --model 3-ppo-mixed-am-goal-no-reset --setting mixed-motive --market am-goal-no-reset --grid-size 10 --capture-interval 200 --capture-frames 30 --save-interval 10 --frames 600000

pipenv run python -m Coloring_with_CAP.scripts.train --agents 3 --model 3-ppo-mixed-competitive-am --setting mixed-motive-competitive --market am --grid-size 10 --capture-interval 200 --capture-frames 30 --save-interval 10 --frames 600000
pipenv run python -m Coloring_with_CAP.scripts.train --agents 3 --model 3-ppo-mixed-competitive-am-goal --setting mixed-motive-competitive --market am-goal --grid-size 10 --capture-interval 200 --capture-frames 30 --save-interval 10 --frames 600000
pipenv run python -m Coloring_with_CAP.scripts.train --agents 3 --model 3-ppo-mixed-competitive-am-goal-no-reset --setting mixed-motive-competitive --market am-goal-no-reset --grid-size 10 --capture-interval 200 --capture-frames 30 --save-interval 10 --frames 600000


Set-Location -Path "C:\Users\Zarah\Documents\workspace\MA\Coloring_with_CAP\scripts\"