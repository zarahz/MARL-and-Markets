# set-executionpolicy remotesigned to remove trust privileges
Set-Location -Path "/Users/zarah/Documents/workspace/CMARL-CAP-and-Markets/Coloring"

# & C:/Users/Zarah/.virtualenvs/Coloring_with_CAP-xNNGJax5/Scripts/Activate.ps1

#EASY
& .\scripts\powershell\easy\1-ppo.ps1
& .\scripts\powershell\easy\2-ppo.ps1
& .\scripts\powershell\easy\2-ppo-mixed.ps1
& .\scripts\powershell\easy\2-ppo-mixed-competitive.ps1

& .\scripts\powershell\easy\1-dqn.ps1
& .\scripts\powershell\easy\2-dqn.ps1
& .\scripts\powershell\easy\2-dqn-mixed.ps1
& .\scripts\powershell\easy\2-dqn-mixed-competitive.ps1

#other scripts
# & .\scripts\powershell\easy\dqn_comparisons.ps1

# HARD
# & .\scripts\powershell\hard\1-ppo.ps1
# & .\scripts\powershell\hard\3-ppo.ps1
# & .\scripts\powershell\hard\3-ppo-mixed.ps1
# & .\scripts\powershell\hard\3-ppo-mixed-competitive.ps1

# & .\scripts\powershell\hard\1-dqn.ps1
# & .\scripts\powershell\hard\3-dqn.ps1
# & .\scripts\powershell\hard\3-dqn-mixed.ps1
# & .\scripts\powershell\hard\3-dqn-mixed-competitive.ps1

# Set-Location -Path "C:\Users\Zarah\Documents\workspace\MA\Coloring\scripts\"
