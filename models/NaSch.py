# author metro(lhq)
# time 2021/10/7

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from scipy.stats import poisson, nbinom
import random
import numpy as np
import copy


class NaSch(object):
    """ A traffic flow simulation model. """

    def __init__(self, config):
        self.num_of_cells = config.num_of_cells
        self.num_of_vehicles = config.num_of_vehicles
        self.max_time_step = config.max_time_step
        self.max_speed = config.max_speed
        self.p_slowdown = config.p_slowdown
        self.pause_time = config.pause_time
        self.cell_size = config.cell_size
        self.conflict_zone = config.conflict_zone
        self.mu = config.mu
        self.peak_period = config.peak_period

        np.random.seed(0)

        self.fig = plt.figure(figsize=(21, 3),
                              dpi=96,
                              )
        self.link = []
        # The occupation of the road is stored in self.link
        # The elements of the list will be the speed of the car otherwise None
        self.link_index = list(np.arange(self.conflict_zone - 10))

    def plot(self, indices, time_step, pedestrian_lock):
        """ Plot the initial space and cells """
        ax = self.fig.add_subplot()
        ax.set(title='Time Step-{}'.format(time_step), xlim=[-5, 105], ylim=[-0.5, 0.5])
        plt.tight_layout()

        x_label = np.linspace(-0.5, self.num_of_cells + 0.5, num=self.num_of_cells + 2)
        y_label = np.tile(np.array([-0.04, 0.04]), (self.num_of_cells + 2, 1))
        ax.plot(x_label, y_label, color='gray')
        for _ in x_label:
            ax.plot([_, _], [-0.04, 0.04], color='gray')

        x_label_ = np.linspace(self.conflict_zone - 0.5, self.conflict_zone + 3.5, num=5)
        y_label_ = np.tile(np.array([-0.05, 0.05]), (5, 1))
        ax.plot(x_label_, y_label_, color='orange', linestyle='dashed')
        for _ in x_label_:
            ax.plot([_, _], [-0.05, 0.05], color='orange', linestyle='dashed')

        ax.plot(indices, [0] * len(indices), 'sk', markersize=self.cell_size)
        if pedestrian_lock == 1:
            ax.plot(self.conflict_zone + 3, [0], 'xr', markersize=self.cell_size)
        # self.ax.tight_layout()
        plt.pause(self.pause_time)
        ax.cla()

    def get_empty_front(self, index_of_cell):
        """ Get the number of empty cells in front of one specific vehicle. """
        num_vehicles_front = 0

        indices = [inx for inx, val in enumerate(self.link) if val is not None]

        # If the vehicle is in front all of the others, return 0.
        if index_of_cell == max(indices):
            num_vehicles_front = 0
        else:
            for _ in indices:
                if index_of_cell == _:
                    num_vehicles_front = indices[indices.index(_) + 1] - (_ + 1)

        return num_vehicles_front

    def initialization(self):  # TODO
        """ Initialization, we will randomly pick some cells, in which cars will be deployed with random speed. """
        self.link = [None] * self.num_of_cells
        indices = random.sample(self.link_index, k=self.num_of_vehicles)
        for i in indices:
            self.link[i] = random.randint(0, self.max_speed)

    def nasch_process(self):
        """

        :return:
        """

        for t in range(0, self.max_time_step):
            for cell in [inx for inx, val in enumerate(self.link) if val is not None]:
                if cell < self.conflict_zone:
                    # Step1 acceleration
                    self.link[cell] = min(self.link[cell] + 1, self.max_speed)
                    # Step2 deceleration
                    self.link[cell] = min(self.link[cell], self.get_empty_front(index_of_cell=cell))
                    # Randomly_slow_down
                    if random.random() <= self.p_slowdown:
                        self.link[cell] = max(self.link[cell] - 1, 0)
                else:
                    # 限制冲突区内最大速度为2
                    self.link[cell] = min(self.link[cell], 2)
                    # 冲突区内不能加速
                    if random.random() <= 0.5:  # TODO
                        self.link[cell] = 1

            link_ = [None] * self.num_of_cells
            for cell in [inx for inx, val in enumerate(self.link) if val is not None]:
                index_ = cell + self.link[cell]
                # 离去规则
                if index_ >= self.num_of_cells:
                    index_ -= self.num_of_cells
                link_[index_] = self.link[cell]
            self.link = copy.deepcopy(link_)

            # Plot the image
            indices = [inx for inx, val in enumerate(self.link) if val is not None]
            print(indices)
            self.plot(indices=indices, time_step=t, pedestrian_lock=1)

    def closed_update(self):
        """
        封闭条件更新规则，指车辆在人行横道前停止，后车开始排队，所有车辆无法驶离

        :return:
        """
        link_ = [None] * self.num_of_cells
        indices = [inx for inx, val in enumerate(self.link) if val is not None]
        if indices[-1] == self.conflict_zone + 2:
            # 最后一辆车停车等待行人通过，速度为0
            self.link[indices[-1]] = 0
        for cell in indices:
            index_ = cell + self.link[cell]
            link_[index_] = self.link[cell]
        self.link = copy.deepcopy(link_)

    def exoteric(self):
        """
        开放条件更新规则，指车辆直接穿过人行横道，并在达到终点时被移除（特指更新后超过了车道的最大值）

        :return:
        """
        link_ = [None] * self.num_of_cells
        indices = [inx for inx, val in enumerate(self.link) if val is not None]
        # 最前面一车辆，考虑其更新后的位置，如果超过了车道最大长度，移除
        if (indices[-1] + self.link[indices[-1]]) > self.num_of_cells:
            indices.pop()  # 移除最后一个元素（车辆）
        for cell in indices:
            index_ = cell + self.link[cell]
            link_[index_] = self.link[cell]
        self.link = copy.deepcopy(link_)

    def vehicle_arrival(self):
        """
        分高峰和平峰时段，用拟合出来的泊松分布进行车辆生成

        :return:
        """
        if self.peak_period:
            prob = poisson.pmf(k=np.arange(0, 3, 1), mu=self.mu)
            prob = [1-prob[1]-prob[2], prob[1], prob[2]]
        else:
            # prob..  # TODO
        num_vehicles = np.random.choice(np.arange(0, 3, 1), p=prob)
        # 判断前num_vehicles是否被占用，如果被占用，则只能在占用位置之后生成；
        # 如果有多车生成，一般直接在一个step内部署完毕
        if num_vehicles == 0:
            pass
        for i in range(num_vehicles):
            for _ in self.link[0:num_vehicles]:
                if _ is not None:
                    # 我们假设生成车辆的初始速度为10m/s
                    self.link[0:self.link.index(_)] = 2

    def pedestrian_arrival(self):
        """
        高峰时段和平峰时段一致，用拟合出来的负二项分布进行行人生成

        :return:
        """
        prob = nbinom.pmf()

















