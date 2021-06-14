from environment.grid import *
from environment.register import register


class DoorKeyEnv(GridEnv):
    """
    Environment with a door and key, sparse reward
    """

    def __init__(self, size=8):
        super().__init__(
            grid_size=size,
            max_steps=10*size*size,
            agents=2
        )

    def _gen_grid(self, width, height, agents_amount):
        # Create an empty grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        # Place a goal in the bottom-right corner
        self.put_obj(Goal(), width - 2, height - 2)

        # Create a vertical splitting wall
        splitIdx = 3  # self._rand_int(2, width/2)
        self.grid.vert_wall(splitIdx, 0)

        for agent in range(agents_amount):
            # Place the agent at a random position and orientation
            # on the left side of the splitting wall
            self.place_agent(index=agent)  # size=(splitIdx, height)

        # Place a door in the wall
        doorIdx = self._rand_int(1, width-2)
        self.put_obj(Door('yellow', is_locked=True), splitIdx, 1)

        # Place a yellow key on the left side
        self.place_obj(
            obj=Key('yellow'),
            top=(0, 0),
            size=(splitIdx, height)
        )
        self.door_unlocked = False
        self.mission = "use the key to open the door and then get to the goal"

    def step(self, index, action):
        obs, reward, done, info = super().step(index, action)

        if info['door_unlocked'] and not done:
            self.door_unlocked = True

        if done:
            reward = self.adjust_reward(index, reward)

        return obs, reward, done, info

    def adjust_reward(self, index, reward):
        if reward < 0:
            if self.door_unlocked:
                # agent did not reach the goal but unlocked the door
                reward = reward + 1
        else:
            # reward for reaching the goal is up to 20
            if self.door_unlocked:
                # upgrade reward if agent reaches goal and unlocked the door
                reward = reward + 5
                print("reached goal with open door ", str(reward))

        # TODO upgrade reward to max. if second agent reaches the goal in x steps
        return reward


class DoorKeyEnv7x7(DoorKeyEnv):
    def __init__(self):
        super().__init__(size=7)


register(
    id='DoorKey-Grid-7x7-v0',
    entry_point='environment.layout:DoorKeyEnv7x7'
)
