#!/usr/bin/env python
# -*- coding: utf-8 -*-

from environment.grid import *
from environment.register import register


class FourRoomsEnv(GridEnv):
    """
    Classic 4 rooms gridworld environment.
    Can specify agent and goal position, if not it set at random.
    """

    def __init__(
        self,
        agents=1,
        agent_view_size=7,  # currently training with view size smaller than 7 fails! TODO!
        max_steps=None,
        competitive=False,
        market="",
        trading_fee=0.05,
        size=19
    ):
        # if not max_steps:
        #     # since env is pretty big and not easy to solve set steps to high number!
        #     max_steps = size*size*2  # 19*19*2 = 722

        super().__init__(
            grid_size=size,
            agents=agents,
            max_steps=max_steps,
            agent_view_size=agent_view_size,
            competitive=competitive,
            market=market,
            trading_fee=trading_fee
            # Set this to True for maximum speed
            # see_through_walls=True
        )

    def _gen_grid(self, width, height, agents):
        # Create the grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.horz_wall(0, 0)
        self.grid.horz_wall(0, height - 1)
        self.grid.vert_wall(0, 0)
        self.grid.vert_wall(width - 1, 0)

        room_w = width // 2
        room_h = height // 2

        # For each row of rooms
        for j in range(0, 2):

            # For each column
            for i in range(0, 2):
                xL = i * room_w
                yT = j * room_h
                xR = xL + room_w
                yB = yT + room_h

                # Bottom wall and door
                if i + 1 < 2:
                    self.grid.vert_wall(xR, yT, room_h)
                    pos = (xR, self._rand_int(yT + 1, yB))
                    self.grid.place_floor(*pos)

                # Bottom wall and door
                if j + 1 < 2:
                    self.grid.horz_wall(xL, yB, room_w)
                    pos = (self._rand_int(xL + 1, xR), yB)
                    self.grid.place_floor(*pos)

        # Randomize the player start position and orientation
        for agent in range(agents):
            # Place the agent at a random position
            self.place_agent(agent)

        self.mission = 'Reach the goal'


register(
    id='FourRooms-Grid-v0',
    entry_point='environment.layout:FourRoomsEnv'
)
