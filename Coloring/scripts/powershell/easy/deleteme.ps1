pipenv run python -m Coloring.scripts.train --algo ppo --model easy\\12-ppo-mixed-sm `
    --setting mixed-motive `
    --market sm `
    --trading-fee 0.1
    
pipenv run python -m Coloring.scripts.train --algo ppo --model easy\\12-ppo-mixed-sm-goal `
    --setting mixed-motive `
    --market sm-goal `
    --trading-fee 0.1

pipenv run python -m Coloring.scripts.train --algo ppo --model easy\\12-ppo-mixed-sm-no-reset `
    --setting mixed-motive `
    --market sm-no-reset `
    --trading-fee 0.1

pipenv run python -m Coloring.scripts.train --algo ppo --model easy\\12-ppo-mixed-sm-goal-no-reset `
    --setting mixed-motive `
    --market sm-goal-no-reset `
    --trading-fee 0.1

pipenv run python -m Coloring.scripts.train --algo ppo --model easy\\12-ppo-mixed-am `
    --setting mixed-motive `
    --market am `
    --trading-fee 0.1
    
pipenv run python -m Coloring.scripts.train --algo ppo --model easy\\12-ppo-mixed-am-goal `
    --setting mixed-motive `
    --market am-goal `
    --trading-fee 0.1
    
pipenv run python -m Coloring.scripts.train --algo ppo --model easy\\12-ppo-mixed-am-no-reset `
    --setting mixed-motive `
    --market am-no-reset `
    --trading-fee 0.1

pipenv run python -m Coloring.scripts.train --algo ppo --model easy\\12-ppo-mixed-am-no-debt `
    --setting mixed-motive `
    --market am-no-debt `
    --trading-fee 0.1

pipenv run python -m Coloring.scripts.train --algo ppo --model easy\\12-ppo-mixed-am-goal-no-reset `
    --setting mixed-motive `
    --market am-goal-no-reset `
    --trading-fee 0.1

pipenv run python -m Coloring.scripts.train --algo ppo --model easy\\12-ppo-mixed-am-goal-no-debt `
    --setting mixed-motive `
    --market am-goal-no-debt `
    --trading-fee 0.1