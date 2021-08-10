TODO 
====== Unklarheiten
- start implementation of shareholder market
    - TBD:
        - "agents can only make either selling or buying offers at a time"? Not implemented yet, all agents can sell and buy 
          sm: Agent 1: [1,2,1] - Agent 2: [2,1,1] Agent 1 would sell a share to 2 but 2 would sell a share back (implemented in market/action_match)
          am: Agent 1: [1,2,2] - Agent 2: [2,1,1] Agent 1 buys action 2 from 2 but 2 also buys action 1 from 1
          Is Buying or Selling regarded first?
        - Currently every step includes action markets -> should there be a default to not buy market actions? Or should there be a bool to specify
          if the agent wants to sell or buy (or do nothing)?
        - How to define the action space? Problem is the current Linear NN which returns a number within the action_space
            is an encoding sufficient? sm: action_space.n=5 NN returns a number of 5x5 i.e. 25 = [4,4] -> hotencoding
=======
- ANMERKUNG: two agents mixed resetted fields: agenten setzen gegnerische felder nicht gezielt zurück, da sie dadurch selbst keine direkten vorteil erzielen (feld ist dann nur resetted aber nicht ihnen zugehörig) (vermutlich würden sie eher diese taktik spielen, wenn die felder nicht zurückgesetzt werden, sondern direkt ihre farbe übernehmen)
    - vielleicht noch neue settings: 
        - competitive mixed, bei denen nur der agent mit höherem prozent den reward erhält?
        - mixed without reset, bei denen felder direkt die farbe des agenten übernehmen ohne zurück gesetzt zu werden (kein bit switching mehr)
- plot feedback:
    - for all settings show coloration percentage
    - one plot with overall percentage of mixed AND percentage setting of two agents
    - plot during training w csv maybe (i.e. take last n episode)
        - two criteria 1) how fast converge (zB bei der wievielten episode hat ein agent es gelernt?) and 2) how stable is it? (zB bei einem agent -> macht er trotzdem noch oft fehler?) 
        - take training logged csv, convert to pandas and plot in jupyter
- market adaptation:
    - encoding of actions
    - market matrix: agent can EITHER buy or sell in one step!
        - sm: 
            - (aktion, kaufen/verkaufen/nichtstun, kaufaktion) -> wenn verkaufen der fall ist und mehrere agenten kaufen möchten kriegt jeder eine aktie -> beim ende einer episode und der reward berechnung erhalten die aktionäre dann die dividenden in einer gewissen betragshöhe (kaufpreis hier gleich dividende zB 0.5)
            - Fällt balance des agents beim trade unter null findet der trade nicht statt. möchten mehrere agents kaufen wird in ZUFÄLLIGER REIHENFOLGE verkauft
        - am: 
            - ????? zu fragen:
            aktion sieht vermutlich so aus: (aktion, kaufen/verkaufen/nichtstun, agent)
                - wenn nichtstun gewählt wurde führt man die aktion in der umgebung aus?
                - wenn kaufen gewählt wurde kauft man vom gewähltem agent die aktion, aber was macht man wenn der gewählte agent nicht die aktion ausführt? In der Umgebung wird bei einem kauf nichts ausgeführt, ansonsten schon?
                - wenn verkaufen gewählt wurde aber niemand kauft macht man nichts? Wenn gekauft wird, führt man die aktion aus?
            - vermutung -> [aktion, agent/nichtstun, kaufaktion] beim Actionmarkt kauft man aktionen 
            
-------------
- implement another learning algorithm? I.e. (single) DQN?

done:
- plot that show the CAP
    - mean reward/episode for a one agent environment training vs multi agent environments (i.e. 2 and 3 agents)
    - amount of reset fields per episode in a one agent environment training vs multi agent environments (i.e. 2 and 3 agents)
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


# Current Observations
For a run of 100 Episodes with max steps of 25 each, here are the current observations: 
![plot](./Coloring_with_CAP/visualization/plots/Rewards_per_episode.png)

Number of times the environment was solved, counted with reward=1 feedback.
![plot](./Coloring_with_CAP/visualization/plots/Goal_achievements_per_setting.png)

Number of times cells were reset.
![plot](./Coloring_with_CAP/visualization/plots/Reset_fields_per_setting.png)

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