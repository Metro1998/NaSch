# author metro(lhq)
# time 2021/10/7

import matplotlib as mpl
import matplotlib.pyplot as plt
import random
import numpy as np


class NaSch(object):
    """ A traffic flow simulation model. """

    def __init__(self, config):
        self.num_of_cells = config.num_of_cells
        self.num_of_cars = config.num_of_cars
        self.max_time = config.max_time
        self.max_speed = config.max_speed
        self.p_slowdown = config.p_slowdown
        self.pause_time = config.pause_time
        self.cell_size = config.cell_size

    def plot_space(self):
        xlabel = np.linspace(-0.5, self.num_of_cells + 0.5, num=self.num_of_cells + 1)
        ylabel = np.tile(np.array([-0.5, 0.5]), (self.num_of_cells + 1, 1))
        plt.plot(xlabel, ylabel)
        plt.axis([-0.5, self.num_of_cells + 0.5, -0.5, 0.5])
        # Disable xticks and yticks
        plt.xticks([])
        plt.yticks([])










