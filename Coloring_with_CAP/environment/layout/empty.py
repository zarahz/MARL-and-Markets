from environment.grid import *
from environment.register import register


class EmptyEnv(GridEnv):
    """
    Empty grid environment, no obstacles, sparse reward
    """

    def __init__(
        self,
        agents=2,
        agent_view_size=5,
        max_steps=100,
        size=8
    ):
        print(agents)
        super().__init__(
            grid_size=size,
            agents=agents,
            max_steps=max_steps,
            agent_view_size=agent_view_size,
            # Set this to True for maximum speed
            # see_through_walls=True
        )

    def _gen_grid(self, width, height, agents):
        # Create an empty grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        for agent in range(agents):
            # Place the agent at a random position
            self.place_agent(agent)

        self.mission = "Color in all Fields"


register(
    id='Empty-Grid-v0',
    entry_point='environment.layout:EmptyEnv'  # default size
)
