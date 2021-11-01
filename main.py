from utilities.Config import Config
from models.NaSch import NaSch
from utilities.utilities import cal
import numpy as np
import matplotlib.pyplot as plt


config = Config()
config.num_of_cells = 20
config.num_of_vehicles = 10
config.max_time_step = 120
config.max_speed = 3
config.p_slowdown = 0.3
config.pause_time = 0.01
config.cell_size = 5
config.conflict_zone = 14
config.peak_period = True
config.distribution_parameters['n_peak'] = 340 / 300
config.distribution_parameters['n_flat'] = 335 / 300
config.distribution_parameters['p_peak'] = 0.83
config.distribution_parameters['p_flat'] = 0.9
config.distribution_parameters['mu_peak'] = 16.50 / 300
config.distribution_parameters['mu_flat'] = 17.67 / 300

a = 0.712
b = 0.324
x = 0.561
y = 0.187
config.game_ld_parameters['pedestrian_first'] = a * (1 - b)
config.game_ld_parameters['vehicle_first'] = b * (1 - a)
config.game_ld_parameters['compromise'] = (1 - a) * (1 - b)
config.game_ld_parameters['conflict'] = a * b

config.game_sd_parameters['pedestrian_first'] = x * (1 - y)
config.game_sd_parameters['vehicle_first'] = y * (1 - x)
config.game_sd_parameters['compromise'] = (1 - x) * (1 - y)
config.game_sd_parameters['conflict'] = x * y

z = np.zeros((10, 10), dtype=float)
for i in range(10):
    for j in range(10):
        config.distribution_parameters['p_peak'] = 0.5 + i * 0.05
        config.distribution_parameters['mu_peak'] = 0.25 + i * 0.05
        nasch = NaSch(config=config)
        z[i][j] = nasch.nasch_process()
np.save(arr=z, file='../NaSch/result/z')

fig = plt.figure()
ax = plt.axes(projection='3d')
xx = np.arange(0.5, 1.05, 0.05)
yy = np.arange(0.25, 0.80, 0.05)
X, Y = np.meshgrid(xx, yy)


