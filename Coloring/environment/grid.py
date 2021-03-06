import math
import hashlib
from os import stat
import gym
from enum import IntEnum
import numpy as np
from gym import spaces
from gym.utils import seeding
from .rendering import *
from environment.colors import *
import copy

# Size in pixels of a tile in the full-scale human view
TILE_PIXELS = 32

# Map of object type to integers
STATUS_TO_IDX = {
    'not_colored': 0,
    'colored': 1
}
IDX_TO_STATUS = dict(zip(STATUS_TO_IDX.values(), STATUS_TO_IDX.keys()))

# Map of object type to integers
OBJECT_TO_IDX = {
    'unseen': 0,
    'empty': 1,
    'wall': 2,
    'floor': 3,
    'agent': 4
}

IDX_TO_OBJECT = dict(zip(OBJECT_TO_IDX.values(), OBJECT_TO_IDX.keys()))


class WorldObj:
    """
    Base class for grid world objects
    """

    def __init__(self, type, is_colored, color, opacity=1):
        assert type in OBJECT_TO_IDX, type
        # assert color in IDX_TO_COLOR, color
        self.type = type
        self.color = color
        self.is_colored = is_colored
        self.contains = None

        # Initial position of the object
        self.init_pos = None

        # Current position of the object
        self.cur_pos = None

    def can_overlap(self):
        """Can the agent overlap with this?"""
        return False

    def can_pickup(self):
        """Can the agent pick this up?"""
        return False

    def can_contain(self):
        """Can this contain another object?"""
        return False

    def see_behind(self):
        """Can the agent see behind this object?"""
        return True

    def encode(self):
        """Encode the a description of this object as a 3-tuple of integers"""
        return (OBJECT_TO_IDX[self.type], self.is_colored, self.color)

    @staticmethod
    def decode(type_idx, is_colored, color_idx):
        """Create an object from a 3-tuple state description"""

        obj_type = IDX_TO_OBJECT[type_idx]
        color = IDX_TO_COLOR[color_idx]

        if obj_type == 'empty' or obj_type == 'unseen':
            return None

        if obj_type == 'wall':
            v = Wall(color)
        elif obj_type == 'floor':
            v = Floor(is_colored, color)
        elif obj_type == 'agent':
            v = Agent(is_colored, color)
        else:
            assert False, "unknown object type in decode '%s'" % obj_type

        return v

    def render(self, r):
        """Draw this object with the given renderer"""
        raise NotImplementedError


class Floor(WorldObj):
    """
    Colored floor tile the agent can walk over
    """

    def __init__(self, is_colored=0, color=1):
        if is_colored == 0:
            # if a floor is not colored always set it to white regardless of the incoming color!
            color = 1
        super().__init__('floor', is_colored, color)

    def can_overlap(self):
        return True

    def render(self, img):
        # Give the floor a pale color
        color = COLORS[self.color] if type(
            self.color) == str else list(COLORS.values())[self.color]
        fill_coords(img, point_in_rect(0.031, 1, 0.031, 1),
                    0.8 * np.array(color, dtype=np.uint8))


class Agent(WorldObj):
    """
    Agent as Worldobject
    """

    def __init__(self, is_colored, color):
        super().__init__('agent', is_colored, color)

    def can_overlap(self):
        # if isinstance(obj, Agent) or isinstance(obj, Floor):
        return True
        # return False

    def render(self, img):
        color = COLORS[self.color] if type(
            self.color) == str else list(COLORS.values())[self.color]
        if self.is_colored:
            # rendered pale Floor (0.8 alpha)
            fill_coords(img, point_in_rect(0.031, 1, 0.031, 1),
                        0.8 * np.array(color, dtype=np.uint8))
        # rendered Agent as circle
        fill_coords(img, point_in_circle(0.5, 0.5, 0.31), color)


class Wall(WorldObj):
    def __init__(self, color=0):
        # is_colored status 1 to make it easier to check if the whole grid is colored
        super().__init__('wall', 1, color)
        self.color = color

    def see_behind(self):
        return False

    def render(self, img):
        color = COLORS[self.color] if type(
            self.color) == str else list(COLORS.values())[self.color]
        fill_coords(img, point_in_rect(0, 1, 0, 1),
                    color)


class Grid:
    """
    Represent a grid and operations on it
    """

    # Static cache of pre-renderer tiles
    tile_cache = {}

    def __init__(self, width, height):
        assert width >= 3
        assert height >= 3

        self.width = width
        self.height = height

        self.grid = [Floor()] * width * height  # [None] * width * height

    def __contains__(self, key):
        if isinstance(key, WorldObj):
            for e in self.grid:
                if e is key:
                    return True
        elif isinstance(key, tuple):
            for e in self.grid:
                if e is None:
                    continue
                if (e.color, e.type) == key:
                    return True
                if key[0] is None and key[1] == e.type:
                    return True
        return False

    def __eq__(self, other):
        grid1 = self.encode_grid_objects()
        grid2 = other.encode_grid_objects()
        return np.array_equal(grid2, grid1)

    def __ne__(self, other):
        return not self == other

    def copy(self):
        from copy import deepcopy
        return deepcopy(self)

    def set(self, i, j, v):
        assert i >= 0 and i < self.width
        assert j >= 0 and j < self.height
        self.grid[j * self.width + i] = v

    def get(self, i, j):
        assert i >= 0 and i < self.width
        assert j >= 0 and j < self.height
        return self.grid[j * self.width + i]

    def horz_wall(self, x, y, length=None, obj_type=Wall):
        if length is None:
            length = self.width - x
        for i in range(0, length):
            self.set(x + i, y, obj_type())

    def vert_wall(self, x, y, length=None, obj_type=Wall):
        if length is None:
            length = self.height - y
        for j in range(0, length):
            self.set(x, y + j, obj_type())

    def wall_rect(self, x, y, w, h):
        self.horz_wall(x, y, w)
        self.horz_wall(x, y+h-1, w)
        self.vert_wall(x, y, h)
        self.vert_wall(x+w-1, y, h)

    def place_floor(self, x, y):
        self.set(x, y, Floor())

    def rotate_left(self):
        """
        Rotate the grid to the left (counter-clockwise)
        """

        grid = Grid(self.height, self.width)

        for i in range(self.width):
            for j in range(self.height):
                v = self.get(i, j)
                grid.set(j, grid.height - 1 - i, v)

        return grid

    def slice(self, topX, topY, width, height):
        """
        Get a subset of the grid
        """

        grid = Grid(width, height)

        for j in range(0, height):
            for i in range(0, width):
                x = topX + i
                y = topY + j

                if x >= 0 and x < self.width and \
                   y >= 0 and y < self.height:
                    v = self.get(x, y)
                else:
                    v = Wall()

                grid.set(i, j, v)

        return grid

    @classmethod
    def render_tile(
        cls,
        obj,
        agent,
        highlight=False,
        tile_size=TILE_PIXELS,
        subdivs=3
    ):
        """
        Render a tile and cache the result
        """
        # color, x, y, ?, highlight, size
        key = (None, None, None, None, highlight, tile_size)
        # Hash map lookup key for the cache
        if agent and agent['pos'] is not None:
            key = (agent['color'], agent['pos'][0],
                   agent['pos'][1], highlight, tile_size)

        # (type, is_colored, color) + key = (color, x, y, ?, highlight, size)
        key = obj.encode() + key if obj else key

        if key in cls.tile_cache:  # what does that do?
            return cls.tile_cache[key]

        img = np.full(shape=(tile_size * subdivs,
                             tile_size * subdivs, 3), fill_value=100, dtype=np.uint8)

        # Draw the grid lines (top and left edges)
        fill_coords(img, point_in_rect(0, 0.031, 0, 1), (100, 100, 100))
        fill_coords(img, point_in_rect(0, 1, 0, 0.031), (100, 100, 100))

        if obj != None:
            obj.render(img)

        # Overlay the agent on top
        if agent:
            fill_coords(img, point_in_circle(0.5, 0.5, 0.31),
                        COLOR_VALUES[agent['color']])

        # Highlight the cell if needed
        if highlight:
            highlight_img(img)

        # Downsample the image to perform supersampling/anti-aliasing
        img = downsample(img, subdivs)

        # Cache the rendered tile
        cls.tile_cache[key] = img

        return img

    def render(
        self,
        tile_size,
        agents,
        highlight_mask=None
    ):
        """
        Render this grid at a given scale
        :param r: target renderer object
        :param tile_size: tile size in pixels
        """
        if highlight_mask is None:
            highlight_mask = np.zeros(
                shape=(self.width, self.height), dtype=np.bool)

        # Compute the total grid size
        width_px = self.width * tile_size
        height_px = self.height * tile_size

        img = np.zeros(shape=(height_px, width_px, 3), dtype=np.uint8)

        # Render the grid
        for j in range(0, self.height):
            for i in range(0, self.width):
                cell = self.get(i, j)
                for agent in agents:
                    agent_in_cell = np.array_equal(
                        agents[agent]['pos'], (i, j))
                    if agent_in_cell:
                        break
                tile_img = Grid.render_tile(
                    cell,
                    agent=agents[agent] if agent_in_cell else None,
                    highlight=highlight_mask[i, j],
                    tile_size=tile_size
                )

                agent_in_cell = False  # reset

                ymin = j * tile_size
                ymax = (j+1) * tile_size
                xmin = i * tile_size
                xmax = (i+1) * tile_size
                img[ymin:ymax, xmin:xmax, :] = tile_img
        return img

    def encode_grid_objects(self, vis_mask=None):
        """
        Produce a compact numpy encoding of the grid
        """

        if vis_mask is None:
            vis_mask = np.ones((self.width, self.height), dtype=bool)

        array = np.zeros((self.width, self.height, 3), dtype='uint8')

        for i in range(self.width):
            for j in range(self.height):
                if vis_mask[i, j]:
                    v = self.get(i, j)

                    if v is None:
                        array[i, j, 0] = OBJECT_TO_IDX['empty']
                        array[i, j, 1] = 0
                        array[i, j, 2] = 0

                    else:
                        array[i, j, :] = v.encode()

        return array

    @staticmethod
    def decode(array):
        """
        Decode an array grid encoding back into a grid
        """

        width, height, channels = array.shape
        assert channels == 3

        vis_mask = np.ones(shape=(width, height), dtype=np.bool)

        grid = Grid(width, height)
        for i in range(width):
            for j in range(height):
                type_idx, is_colored, color_idx = array[i, j]
                v = WorldObj.decode(type_idx, is_colored, color_idx)
                grid.set(i, j, v)
                vis_mask[i, j] = (type_idx != OBJECT_TO_IDX['unseen'])

        return grid, vis_mask

    def process_vis(grid, agent_pos):
        mask = np.zeros(shape=(grid.width, grid.height), dtype=np.bool)

        # agent_pos = agents[agent]['pos']
        mask[agent_pos[0], agent_pos[1]] = True

        for j in reversed(range(0, grid.height)):
            for i in range(0, grid.width-1):
                if not mask[i, j]:
                    continue

                cell = grid.get(i, j)
                if cell and not cell.see_behind():
                    continue

                mask[i+1, j] = True
                if j > 0:
                    mask[i+1, j-1] = True
                    mask[i, j-1] = True

            for i in reversed(range(1, grid.width)):
                if not mask[i, j]:
                    continue

                cell = grid.get(i, j)
                if cell and not cell.see_behind():
                    continue

                mask[i-1, j] = True
                if j > 0:
                    mask[i-1, j-1] = True
                    mask[i, j-1] = True

        for j in range(0, grid.height):
            for i in range(0, grid.width):
                if not mask[i, j]:
                    grid.set(i, j, None)

        return mask


class GridEnv(gym.Env):
    """
    2D grid world game environment
    """

    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 10
    }

    # Enumeration of possible actions
    class Actions(IntEnum):
        # Move in direction
        left = 1
        right = 2
        up = 3
        down = 4

        wait = 0

    def __init__(
        self,
        grid_size=None,
        width=None,
        height=None,
        max_steps=None,
        see_through_walls=True,
        seed=1337,
        agent_view_size=5,
        competitive=False,
        market="",
        agents=2
    ):
        generate_colors(agents)

        # Can't set both grid_size and width/height
        if grid_size:
            assert width == None and height == None
            width = grid_size
            height = grid_size

        # Action enumeration for this environment
        self.actions = GridEnv.Actions
        self.market = market
        self.competitive = competitive
        if market:
            if "sm" in market:
                # market actions are unconditional (no recieving agent)
                # action = [env_action, buying(0=buy from agent 0, num>agents=dont buy), selling(0=donothing, 1=sell)]
                self.action_space = gym.spaces.MultiDiscrete(
                    [len(self.actions), agents+1, 2])
            else:  # market == "am"
                # market actions are conditional (a recieving agent is set)
                self.action_space = gym.spaces.MultiDiscrete(
                    [len(self.actions), agents+1, len(self.actions)])
            self.action_table = self.encode_actions()
        else:
            self.action_space = spaces.Discrete(len(self.actions))

        # Number of cells (width and height) in the agent view
        assert agent_view_size % 2 == 1
        assert agent_view_size >= 3
        self.agent_view_size = agent_view_size

        # Observations are dictionaries containing an
        # encoding of the grid and a textual 'mission' string
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(self.agent_view_size, self.agent_view_size, 3),
            dtype='uint8'
        )
        self.observation_space = spaces.Dict({
            'image': self.observation_space
        })

        # Range of possible rewards
        self.reward_range = (0, 1)  # (0, 1)

        # Window to use for human rendering mode
        self.window = None

        # Environment configuration
        self.width = width
        self.height = height
        self.max_steps = max_steps if max_steps else width*height
        print("---------> max steps: ", self.max_steps)
        self.see_through_walls = see_through_walls

        self.agents = {}
        for agent in range(agents):
            self.agents[agent] = {'pos': None, 'color': agent+2}
        # Initialize the RNG
        self.seed(seed=seed)

        # Initialize the state
        self.reset()

    def reset(self):
        for agent in self.agents:
            # Current position of the agent
            self.agents[agent] = {**self.agents[agent], 'pos': None}

        if not IDX_TO_COLOR:
            # ensure colors are generated
            generate_colors(len(self.agents))
        # Generate a new random grid at the start of each episode
        # To keep the same grid for each episode, call env.seed() with
        # the same seed before calling env.reset()
        self._gen_grid(self.width, self.height, len(self.agents))

        # Return observation
        obs = {}

        # These fields should be defined by _gen_grid
        for agent in self.agents:
            assert self.agents[agent]['pos'] is not None

            # Check that the agent doesn't overlap with an object
            start_cell = self.grid.get(*self.agents[agent]['pos'])
            assert start_cell is None or start_cell.can_overlap()

            obs[agent] = self.gen_obs(agent)
        # Step count since episode start
        self.step_count = 0

        return obs

    def seed(self, seed=1337):
        # Seed the random number generator
        self.np_random, _ = seeding.np_random(seed)
        return [seed]

    def hash(self, size=16):
        """Compute a hash that uniquely identifies the current state of the environment.
        :param size: Size of the hashing
        """
        sample_hash = hashlib.sha256()

        to_encode = [self.grid.encode_grid_objects()]
        for agent in self.agents:
            to_encode.append(self.agents[agent]
                             ['pos'])
        for item in to_encode:
            sample_hash.update(str(item).encode('utf8'))

        return sample_hash.hexdigest()[:size]

    @property
    def steps_remaining(self):
        return self.max_steps - self.step_count

    def __str__(self):
        """
        Produce a pretty string of the environment's grid along with the agent.
        A grid cell is represented by 2-character string, the first one for
        the object and the second one for the color.
        """

        # Map of object types to short string
        OBJECT_TO_STR = {
            'wall': 'W',
            'floor': 'F',
            'door': 'D',
            'key': 'K',
            'ball': 'A',
            'box': 'B',
            'goal': 'G',
            'lava': 'V',
        }

        # Short string for opened door
        OPENDED_DOOR_IDS = '_'

        str = ''

        for j in range(self.grid.height):

            for i in range(self.grid.width):
                for agent in self.agents:
                    if i == self.agents[agent]['pos'][0] and j == self.agents[agent]['pos'][1]:
                        str += 'X'
                        continue

                c = self.grid.get(i, j)

                if c == None:
                    str += '  '
                    continue

                if c.type == 'door':
                    if c.is_open:
                        str += '__'
                    elif c.is_locked:
                        str += 'L' + c.color[0].upper()
                    else:
                        str += 'D' + c.color[0].upper()
                    continue

                str += OBJECT_TO_STR[c.type] + c.color[0].upper()

            if j < self.grid.height - 1:
                str += '\n'

        return str

    def _gen_grid(self, width, height):
        assert False, "_gen_grid needs to be implemented by each environment"

    def encode_actions(self):
        ''' 
        When actions cointain multiple information, i.e. selling/buying info, the nn
        only returns a number. That number can be mapped to the corresponding action.
        In order to do that encode all actions with a number. 
        Example: key = 0 -> maps to (0,0,0) that means an agent executes action zero 
        and (in case of market=sm) buys share from agent 0, and (third 0) does not sell its shares 
        '''
        shape = self.action_space.nvec
        # All points in a 3D grid within the given ranges
        combinations = np.mgrid[0:shape[0], 0:shape[1], 0:shape[2]]
        # Make the 0th axis into the last axis
        combinations = np.rollaxis(combinations, 0, 4)
        # Now you can safely reshape while preserving order
        combinations = combinations.reshape((shape.prod(), 3))
        # start with index 1: dict(enumerate(combinations, 1))
        return dict(enumerate(combinations))

    def decode_actions(self, keys):
        ''' 
        When actions cointain multiple information, i.e. selling/buying info, the nn
        only returns a number. That number can be mapped to the corresponding action
        here. 
        Example: key = 0 -> maps to (0,0,0) that means an agent executes action zero 
        and (in case of market=sm) buys share from agent 0, and (third 0) does not sell its shares 
        '''
        return np.array([self.action_table[key] for key in keys])  # self.action_table[key]

    def _reward(self, agent, cell_status_changed, reward, agent_reset_field, reset_fields_by):
        """
        Compute the reward to be given upon success
        """

        # if agent stays on its colors (by waiting or in competitive) reward should stay 0
        if not cell_status_changed:
            return reward, reset_fields_by

        if agent_reset_field:
            # agent reset a field here so return a penalty!
            reward[agent] -= 0.1
            # if agent reset the field save it to info in order to (possibly) restrict market transactions
            if agent not in reset_fields_by:
                reset_fields_by.append(agent)
        else:
            # add small positive reward, since agent colored the new cell!
            reward[agent] += 0.1
        return reward, reset_fields_by

    def _rand_int(self, low, high):
        """
        Generate random integer in [low,high[
        """

        return self.np_random.randint(low, high)

    def _rand_float(self, low, high):
        """
        Generate random float in [low,high[
        """

        return self.np_random.uniform(low, high)

    def _rand_bool(self):
        """
        Generate random boolean value
        """

        return (self.np_random.randint(0, 2) == 0)

    def _rand_elem(self, iterable):
        """
        Pick a random element in a list
        """

        lst = list(iterable)
        idx = self._rand_int(0, len(lst))
        return lst[idx]

    def _rand_subset(self, iterable, num_elems):
        """
        Sample a random subset of distinct elements of a list
        """

        lst = list(iterable)
        assert num_elems <= len(lst)

        out = []

        while len(out) < num_elems:
            elem = self._rand_elem(lst)
            lst.remove(elem)
            out.append(elem)

        return out

    def _rand_color(self):
        """
        Generate a random color name (string)
        """

        return self._rand_elem(COLOR_NAMES)

    def _rand_pos(self, xLow, xHigh, yLow, yHigh):
        """
        Generate a random (x,y) position tuple
        """

        return (
            self.np_random.randint(xLow, xHigh),
            self.np_random.randint(yLow, yHigh)
        )

    def place_obj(self,
                  obj,
                  top=None,
                  size=None,
                  reject_fn=None,
                  max_tries=math.inf,
                  agent=0
                  ):
        """
        Place an object at an empty position in the grid

        :param top: top-left position of the rectangle where to place
        :param size: size of the rectangle where to place
        :param reject_fn: function to filter out potential positions
        """

        if top is None:
            top = (0, 0)
        else:
            top = (max(top[0], 0), max(top[1], 0))

        if size is None:
            size = (self.grid.width, self.grid.height)

        num_tries = 0

        while True:
            # This is to handle with rare cases where rejection sampling
            # gets stuck in an infinite loop
            if num_tries > max_tries:
                raise RecursionError('rejection sampling failed in place_obj')

            num_tries += 1

            pos = np.array((
                self._rand_int(top[0], min(top[0] + size[0], self.grid.width)),
                self._rand_int(top[1], min(top[1] + size[1], self.grid.height))
            ))

            # Don't place the object on top of another object
            if self.grid.get(*pos) and not self.grid.get(*pos).can_overlap():
                continue

            # Check if there is a filtering criterion
            if reject_fn and reject_fn(self, pos):
                continue

            break

        if obj is None:  # placing agent
            self.toggle_is_colored(self.grid, self.grid.get(
                *pos), self.agents[agent]['color'], pos, None)
        else:
            self.grid.set(*pos, obj)
            obj.init_pos = pos
            obj.cur_pos = pos

        return pos

    def put_obj(self, grid, obj, i, j):
        """
        Put an object at a specific position in the grid
        """

        grid.set(i, j, obj)
        obj.init_pos = (i, j)
        obj.cur_pos = (i, j)

    def place_agent(
        self,
        agent,
        top=None,
        size=None,
        max_tries=math.inf
    ):
        """
        Set the agent's starting point at an empty position in the grid
        """
        self.agents[agent] = {**self.agents[agent], 'pos': None}

        pos = self.place_obj(None, top, size, max_tries=max_tries, agent=agent)
        self.agents[agent]['pos'] = pos
        return pos

    def get_view_exts(self, agent):
        """
        Get the extents of the square set of tiles visible to the agent
        Note: the bottom extent indices are not included in the set
        """
        agent_pos = self.agents[agent]['pos']

        topX = agent_pos[0]-self.agent_view_size // 2
        topY = agent_pos[1]-self.agent_view_size // 2
        botX = topX + self.agent_view_size-1
        botY = topY + self.agent_view_size-1
        return (topX, topY, botX, botY)

    def step(self, actions, calc_difference_reward=False):
        if calc_difference_reward:
            dr_agent_grids = [copy.deepcopy(self.grid) for _ in self.agents]
            all_old_pos = []
        self.step_count += 1
        reward = [0]*len(self.agents)
        done = False
        obs = {}
        reset_fields = 0
        reset_fields_by = []
        for agent in self.agents:
            x = self.agents[agent]['pos'][0]
            y = self.agents[agent]['pos'][1]
            new_pos = None
            old_pos = (x, y)
            action = actions[agent]
            # compute new position
            if action == self.actions.left:
                new_pos = np.array([x-1, y])
            elif action == self.actions.right:
                new_pos = np.array([x+1, y])
            elif action == self.actions.down:
                new_pos = np.array([x, y+1])
            elif action == self.actions.up:
                new_pos = np.array([x, y-1])
            elif action != self.actions.wait:
                assert False, "unknown action"

            if calc_difference_reward:
                all_old_pos.append(old_pos)

            # move to new position if possible
            if new_pos is not None:
                reset_field, cell_status_changed = self.move_agent(self.grid,
                                                                   agent, old_pos, new_pos)
                # update reset counter
                reset_fields += reset_field
                # calculate env reward
                reward, reset_fields_by = self._reward(agent, cell_status_changed, reward,
                                                       reset_field, reset_fields_by)

            obs[agent] = self.gen_obs(agent)

        info = {"reset_fields": reset_fields, "reset_fields_by": reset_fields_by,
                "coloration_percentage": round(self.grid_colored_percentage(), 2), "fully_colored": 0}

        if self.whole_grid_colored(self.grid) or self.step_count >= self.max_steps:
            info["fully_colored"] = 1 if self.whole_grid_colored(
                self.grid) else 0
            done = True
            # reward = [1]*len(self.agents)

        if calc_difference_reward and done:
            # only needed to calculate the DR when env is done, 
            # since then the percentage are of importance.
            # Otherwise the dr is rather simple and executed in the wrapper 
            dr_info = self.calculate_difference_rewards(
                dr_agent_grids, all_old_pos)
            info.update({"difference_rewards": dr_info})

        return obs, reward, done, info

    def calculate_difference_rewards(self, grids, all_old_pos):
        dr = [0]*len(self.agents)
        for agent in self.agents:
            for moving_agent, old_pos in enumerate(all_old_pos):
                if old_pos is not None:
                    # the current agent executes default "waiting" action -> stays in old position
                    new_pos = old_pos if moving_agent == agent else self.agents[moving_agent]["pos"]
                    if tuple(new_pos) == old_pos:
                        continue
                    _, _ = self.move_agent(grids[agent], moving_agent, old_pos, new_pos, calc_difference_reward=True)
            dr[agent] = self.grid_colored_percentage(grids[agent])
        return dr

    def move_agent(self, grid, agent, old_pos, new_pos, calc_difference_reward=False):
        new_pos_cell = grid.get(*new_pos)
        if new_pos_cell == None or new_pos_cell.can_overlap():
            cell_status = new_pos_cell
            if not calc_difference_reward:
                self.agents[agent]['pos'] = new_pos
            agent_reset_field = self.toggle_is_colored(grid,
                                                       new_pos_cell, self.agents[agent]['color'], new_pos, old_pos)
            updated_new_pos_cell = grid.get(*new_pos)
            # cell status change needs two conditions since new pos contains agent which could have the same color as prev!
            cell_status_changed = cell_status.color != updated_new_pos_cell.color or cell_status.is_colored != updated_new_pos_cell.is_colored

            return agent_reset_field, cell_status_changed

        return 0, False

    def whole_grid_colored(self, grid):
        return all(grid.encode_grid_objects()[:, :, 1].ravel())

    def grid_colored_percentage(self, grid=None):
        # walkable_cells include agent and floor objects
        walkable_cells = len(self.walkable_cells())
        colored_cells = len(self.colored_cells(grid))
        return colored_cells/walkable_cells

    def walkable_cells(self):
        """ Returns 2d array of all cells containing floors or agents """
        encoded_grid = self.grid.encode_grid_objects()
        floor_cells = encoded_grid[encoded_grid[:, :, 0] == 3]
        agent_cells = encoded_grid[encoded_grid[:, :, 0] == 4]
        return np.concatenate((floor_cells, agent_cells))

    def colored_cells(self, grid=None):
        """ Returns 2d array of all cells containing floor tiles or agents with status is colored """
        encoded_grid = self.grid.encode_grid_objects() if grid is None else grid.encode_grid_objects() 
        colored_floor_cells = encoded_grid[(
            (encoded_grid[:, :, 0] == 3) & (encoded_grid[:, :, 1] == 1))]
        colored_agent_cells = encoded_grid[(
            (encoded_grid[:, :, 0] == 4) & (encoded_grid[:, :, 1] == 1))]
        return np.concatenate((colored_floor_cells, colored_agent_cells))

    def toggle_is_colored(self, grid, obj, color, new_pos, old_pos):
        field_reset = False
        is_colored = 1
        if obj is not None and obj.is_colored and (not self.competitive or (self.competitive and obj.color == color)):
            is_colored = 0
            field_reset = True

        self.put_obj(grid, Agent(is_colored, color), *new_pos)
        if old_pos:
            old_pos_attr = self.grid.get(*old_pos)
            for agent in self.agents:
                if (self.agents[agent]['pos'] == old_pos).all():
                    # in this case the acting agent moves away from a field where another
                    # agent was also standing -> render the staying agent here
                    self.put_obj(grid, Agent(is_colored=old_pos_attr.is_colored,
                                             color=old_pos_attr.color), *old_pos)
            # render the old position without an agent on it
            self.put_obj(grid, Floor(is_colored=old_pos_attr.is_colored,
                         color=old_pos_attr.color), *old_pos)

        return field_reset

    def gen_obs_grid(self, agent):
        """
        Generate the sub-grid observed by the agent.
        This method also outputs a visibility mask telling us which grid
        cells the agent can actually see.
        """

        topX, topY, botX, botY = self.get_view_exts(agent)

        grid = self.grid.slice(
            topX, topY, self.agent_view_size, self.agent_view_size)

        # Process occluders and visibility
        # Note that this incurs some performance cost
        if not self.see_through_walls:
            # self.agents, agent
            vis_mask = grid.process_vis(grid,
                                        agent_pos=(
                                            self.agent_view_size // 2, self.agent_view_size - 1)
                                        )
        else:
            vis_mask = np.ones(shape=(grid.width, grid.height), dtype=np.bool)

        return grid, vis_mask

    def gen_obs(self, agent):
        """
        Generate the agent's view (partially observable, low-resolution encoding)
        """

        grid, vis_mask = self.gen_obs_grid(agent)

        # Encode the partially observable view into a numpy array
        image = grid.encode_grid_objects(vis_mask)

        assert hasattr(
            self, 'mission'), "environments must define a textual mission string"

        # Observations are dictionaries containing:
        # - an image (partially observable view of the environment)
        # - a textual mission string (instructions for the agent)
        obs = {
            'image': image,
            'mission': self.mission
        }

        return obs

    def get_obs_render(self, obs, tile_size=TILE_PIXELS//2):
        """
        Render an global observation for visualization
        """

        grid, _ = Grid.decode(obs)
        # Render the whole grid
        img = grid.render(
            tile_size,
            self.agents
        )

        return img

    def render(self, mode='human', reward=[], close=False, highlight=True, tile_size=TILE_PIXELS):
        """
        Render the whole-grid human view
        """

        if close:
            if self.window:
                self.window.close()
            return

        if mode == 'human' and not self.window:
            import environment.window
            self.window = environment.window.Window('grid coloring')
            self.window.show(block=False)

        # Mask of which cells to highlight
        highlight_mask = np.zeros(
            shape=(self.width, self.height), dtype=np.bool)

        for agent in self.agents:
            # Compute which cells are visible to the agent
            _, vis_mask = self.gen_obs_grid(agent)

            # For each cell in the visibility mask
            for vis_j in range(0, self.agent_view_size):
                for vis_i in range(0, self.agent_view_size):
                    # If this cell is not visible, don't highlight it
                    if not vis_mask[vis_i, vis_j]:
                        continue

                    # Compute the world coordinates of the top cell
                    # from where agent can see
                    topX, topY, _, _ = self.get_view_exts(agent)
                    abs_i, abs_j = np.add((topX, topY), (vis_i, vis_j))

                    if abs_i < 0 or abs_i >= self.width:
                        continue
                    if abs_j < 0 or abs_j >= self.height:
                        continue

                    # Mark this cell to be highlighted
                    highlight_mask[abs_i, abs_j] = True

        # Render the whole grid
        img = self.grid.render(
            tile_size,
            self.agents,
            highlight_mask=highlight_mask if highlight else None
        )

        if mode == 'human':
            coloration_percentage = self.grid_colored_percentage()
            self.window.set_caption(
                self.mission, coloration_percentage, self.step_count, self.max_steps, rewards=reward)
            self.window.show_img(img)

        return img

    def close(self):
        if self.window:
            self.window.close()
        return
