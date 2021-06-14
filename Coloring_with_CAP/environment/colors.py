import numpy as np

# Map of color names to RGB values
COLORS = {
    'black': np.array([0, 0, 0]),
    'white': np.array([255, 255, 255])
}
# Used to map colors to integers
COLOR_TO_IDX = {
    'black': 0,
    'white': 1
}

COLOR_NAMES = []
COLOR_VALUES = []
IDX_TO_COLOR = {}


def update_global_color_variables():
    """Overwrite all global color variables when new colors are added 
    to COLORS and COLOR_TO_IDX"""
    COLOR_NAMES.extend(sorted(list(COLORS.keys())))
    COLOR_VALUES.extend(list(COLORS.values()))
    IDX_TO_COLOR.update(dict(zip(COLOR_TO_IDX.values(), COLOR_TO_IDX.keys())))


def random_rgb_values():
    """pick random rgb values"""
    red = np.random.randint(0, 255)
    green = np.random.randint(0, 255)
    blue = np.random.randint(0, 255)
    return red, green, blue


def generate_colors(amount):
    """Generate Random unique colors based on set amount of agents"""
    for _ in range(amount):
        # check if the color already exists
        while True:
            red, green, blue = random_rgb_values()
            for color in COLORS:
                same_values = COLORS[color] == np.array([red, green, blue])
                if same_values.all():
                    continue
            break
        color_name = "{}.{}.{}".format(red, green, blue)
        # save values
        COLORS[color_name] = np.array([red, green, blue])
        # i+2 since black and white is already defined
        COLOR_TO_IDX[color_name] = len(COLOR_TO_IDX)
    update_global_color_variables()
