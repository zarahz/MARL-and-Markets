#----------------------------------------------------------------------------------------
# 1 PPO Settings
#----------------------------------------------------------------------------------------
python -m scripts.train --algo ppo --agents 1 --model easy/ppo/1-ppo `
--batch-size 64 `
--max-steps 10
# --grid-size 5 `
# --frames-per-proc 128 `
# --frames 100000


#----------------------------------------------------------------------------------------
# 1 PPO Room Settings
#----------------------------------------------------------------------------------------
# python -m scripts.train --algo ppo --agents 1 --model easy/ppo/1-ppo-rooms `
#     --env FourRooms-Grid-v0 `
#     --grid-size 5 `
#     --frames-per-proc 128 `
#     --frames 150000