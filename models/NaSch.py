# author metro(lhq)
# time 2021/10/7

import matplotlib as mpl
import matplotlib.pyplot as plt
import random
import numpy as np
import copy


class NaSch(object):
    """ A traffic flow simulation model. """

    def __init__(self, config):
        self.num_of_cells = config.num_of_cells
        self.num_of_cars = config.num_of_cars
        self.max_time_step = config.max_time_step
        self.max_speed = config.max_speed
        self.p_slowdown = config.p_slowdown
        self.pause_time = config.pause_time
        self.cell_size = config.cell_size

        self.link = []
        # The occupation of the road is stored in self.link
        # The elements of the list will be the speed of the car otherwise None
        self.link_index = list(np.arange(self.num_of_cells))

    def plot_space(self):
        """ Plot the initial space. """
        xlabel = np.linspace(-0.5, self.num_of_cells + 0.5, num=self.num_of_cells + 1)
        ylabel = np.tile(np.array([-0.5, 0.5]), (self.num_of_cells + 1, 1))
        plt.plot(xlabel, ylabel)
        plt.axis([-0.5, self.num_of_cells + 0.5, -0.5, 0.5])
        # Disable xticks and yticks
        plt.xticks([])
        plt.yticks([])

    def get_empty_front(self, index_of_cell):
        """ Get the number of empty cells in front of one specific car. """
        link2 = self.link * 2
        # We suppose that cars drive away from the road will come back to road again
        num = 0
        i = 1
        while link2[index_of_cell + i] is None:
            num += 1
            i += 1
        return num

    def initialization(self):
        """ Initialization, we will randomly pick some cells, in which cars will be deployed with random speed. """
        self.plot_space()
        self.link = [None] * self.num_of_cells
        indices = random.sample(self.link_index, self.num_of_cars)
        for i in indices:
            self.link[i] = random.randint(0, self.max_speed)

    def nasch_process(self):
        for t in range(0, self.max_time_step):
            indices = [inx for inx, val in enumerate(self.link) if val is not None]
            for cell in indices:
                # Acceleration
                self.link[cell] = min(self.link[cell] + 1, self.max_speed)
                # Deceleration
                self.link[cell] = min(self.link[cell], self.get_empty_front(index_of_cell=cell))
                # Randomly_slow_down
                if random.random() <= self.p_slowdown:
                    self.link[cell] = max(self.link[cell] - 1, 0)
            link_ = [None] * self.num_of_cells
            for cell in indices:
                index_ = cell + self.link[cell]
                if index_ >= self.num_of_cells:
                    index_ -= self.num_of_cells
                link_[index_] = self.link[cell]
            self.link = copy.deepcopy(link_)

            # Plot the image
            indices = [inx for inx, val in enumerate(self.link) if val is not None]
            plt.plot(indices, [0] * self.num_of_cars, 'sk', markersize = self.cell_size)
            plt.xlabel('time_step' + str(t))
            plt.pause(self.pause_time)
            plt.cla()

















