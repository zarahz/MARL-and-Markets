TODO 
- fine tuning to achieve colored grid with one agent
- implement another learning algorithm? I.e. DQN/Team-DQN? Or start with MCAC/DR,... 
- add run parameter market to enable actions as touples and implement reward distribution with shares
- adjust structure (q learning integration into visualization script with algo as file in learning)

# Installation
First clone this repository and navigate into the domain Folder
```
git clone https://github.com/zarahz/CMARL-CAP-and-Markets.git
cd CMARL-CAP-and-Markets/Coloring_with_CAP
pip install -e .
```

# Execution
Now you can run the domain with the following algorithms

### Q Learning 
For a very basic Q Learning algorithm you can execute
```
> python .\Coloring_with_CAP\scripts\q_learning.py --env "Empty-Grid-v0" --agents 2 --agent_view_size 5 --max_steps 10 --episodes 5 --size 5
```
required arguments are `--env` and `--agents`

### PPO (Fine tuning of parameters is still in process, so learning is not optimal)
To train the model first execute the corresponding script:
```
> python -m Coloring_with_CAP.scripts.train --algo ppo --model "EmptyGrid" --save-interval 10 --frames 80000
```
required argument is `--algo`.

Visualization of the environment based on the trained model can be achieved with:
```
> python -m Coloring_with_CAP.scripts.train --model "EmptyGrid"
```
required argument is `--model`.

# Untersuchung der Auswirkungen von Märkten auf das Belohnungs-Zuweisungsproblem in kooperativen Multiagenten-Umgebungen

Bei einer kooperativen Multi Agenten Umgebung sind die
erbrachten Leistungen oftmals unausgeglichen. Der Reward fällt allerdings für alle gleich aus.
Dementsprechend kann es dazu kommen, dass Agenten für schlechte Aktionen belohnt oder
für gute bestraft werden. Durch den globalen Reward haben Agenten keine Möglichkeiten zu
lernen, da sie nicht beurteilen können, ob sie gute Aktionen wählen. Diese Problematik wird
Credit Assignment Problem (CAP) genannt.
In dieser Arbeit wird untersucht, wie sich die Einführung eines Marktes auf dieses Problem
auswirkt. Durch einen Markt können Agenten beispielsweise die Unterstützung anderer
erwerben, indem Anteile des Rewards versteigert werden (Shareholder Market). Eine
weitere Variante eines Marktes ist es, Aktionen anderer Agenten direkt zu erkaufen (Action
Market). Agenten die wesentlich zum Erreichen des Ziels beitragen, sollten auch einen
höheren Reward erzielen. Dadurch könnte das CAP (teilweise) gelöst werden.

Die Umgebung in der getestet wird ist eine zweidimensionale Gridworld, dessen begehbare
Felder zwei Zustände besitzen. Zustand 1 bedeutet das Feld ist eingefärbt, andernfalls ist es
0. Das Ziel ist es die Gridworld komplett zu färben. Agenten erreichen dies, indem sie Felder
besuchen, wodurch dessen Bits umgedreht werden. Laufen beispielsweise zwei Agenten im
selben Zug auf dasselbe Feld, so ändert sich das Bit nicht. Läuft ein Agent über ein bereits
eingefärbtes Feld wird es zurückgesetzt. Der Aktionsraum der Agenten ist dabei sich in eine
Richtung zu bewegen oder zu warten.

Im Verlauf der Ausarbeitung werden folgende Konstellationen und Algorithmen auf die
Umgebung angewendet, um die Schnelligkeit, Leistung und Zielführung zu untersuchen:
- Vergleich von unterschiedlichen Ansätzen zur Lösung des CAPs, z.B. Mean collective
actor critc (MCAC), Difference Rewards (WLU, AU) und COMA mit dem Marktansatz
- Unterschiedliche Reward Vergaben, z.B. negative Rewards pro Schritt im Vergleich zu
spärlichen Rewards am Ende der Episode (hier auch der Vergleich möglich zwischen
ausschließlich positiven Werten bei erreichen des Ziels, sonst Null oder Strafe)
- Vergleich zwischen dem kooperativen Setup gegen einer Mixed-Motive Umgebung,
in denen Agenten nur Rewards basierend auf ihre eigenen färbung erhalten
- Unterschiedliche Schwierigkeitsstufen der Umgebung (größere Felder mit
Raumstrukturen)
- Unterschiedliche Anzahl an Agenten, z.B. 2 im Vergleich zu 5 oder 50 Agenten