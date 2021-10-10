import sys
import numpy as np

# Only ask users to install matplotlib if they actually need it
try:
    import matplotlib.pyplot as plt
except:
    print('To display the environment in a window, please install matplotlib, eg:')
    print('pip install --user matplotlib')
    sys.exit(-1)


class Window:
    """
    Window to draw a gridworld instance using Matplotlib
    """

    def __init__(self, title):
        self.fig = None

        self.imshow_obj = None

        # Create the figure and axes
        self.fig, self.ax = plt.subplots()

        # Show the env name in the window title
        self.fig.canvas.set_window_title(title)

        # Turn off x/y axis numbering/ticks
        self.ax.xaxis.set_ticks_position('none')
        self.ax.yaxis.set_ticks_position('none')
        _ = self.ax.set_xticklabels([])
        _ = self.ax.set_yticklabels([])

        # Flag indicating the window was closed
        self.closed = False

        def close_handler(evt):
            self.closed = True

        self.fig.canvas.mpl_connect('close_event', close_handler)
        self.fig.subplots_adjust(bottom=0.2)

    def show_img(self, img):
        """
        Show an image or update the image being shown
        """

        # Show the first image of the environment
        if self.imshow_obj is None:
            self.imshow_obj = self.ax.imshow(img, interpolation='bilinear')

        self.imshow_obj.set_data(img)
        self.fig.canvas.draw()

        # Let matplotlib process UI events
        # This is needed for interactive mode to work properly
        plt.pause(1)  # (0.001)

    def set_caption(self, mission, grid_coloration_percentage=0, step_count_info=0, max_steps=0, rewards=[]):
        """
        Set/update the caption text below the image
        """
        info = "\ncoloration percentage: " + \
            "{0:0.2f}".format(grid_coloration_percentage) + \
            " steps: " + str(step_count_info) + " of " + str(max_steps) + "\n"
        if rewards:
            reward = "reward:" + str(["{0:0.2f}".format(i) for i in rewards])
            plt.xlabel(mission + info + reward)
        else:
            plt.xlabel(mission + info)

    def reg_key_handler(self, key_handler):
        """
        Register a keyboard event handler
        """

        # Keyboard handler
        self.fig.canvas.mpl_connect('key_press_event', key_handler)

    def show(self, block=True):
        """
        Show the window, and start an event loop
        """

        # If not blocking, trigger interactive mode
        if not block:
            plt.ion()

        # Show the plot
        # In non-interative mode, this enters the matplotlib event loop
        # In interactive mode, this call does not block
        plt.show()

    def close(self):
        """
        Close the window
        """

        plt.close()
        self.closed = True
