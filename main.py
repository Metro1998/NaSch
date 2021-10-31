from utilities.Config import Config
from models.NaSch import NaSch
from utilities.utilities import cal

[x, y, a, b] = cal(V=12, W_v=0.6, O_v1=3, T_v=15.5, U_v=1, P=12, O_p1=2, T_p=0.93, W_p=0.4, O_v2=5, O_p2=4)
print([x, y, a, b])
config = Config()
config.num_of_cells = 15
config.num_of_vehicles = 10
config.max_time_step = 360
config.max_speed = 3
config.p_slowdown = 0.3
config.pause_time = 0.3
config.cell_size = 5
config.conflict_zone = 11
config.peak_period = True
config.distribution_parameters['n_peak'] = 340 / 300
config.distribution_parameters['n_flat'] = 335 / 300
config.distribution_parameters['p_peak'] = 0.83
config.distribution_parameters['p_flat'] = 0.9
config.distribution_parameters['mu_peak'] = 16.50 / 300
config.distribution_parameters['mu_flat'] = 17.67 / 300

config.game_ld_parameters['pedestrian_first'] = a * (1 - b)
config.game_ld_parameters['vehicle_first'] = b * (1 - a)
config.game_ld_parameters['compromise'] = (1 - a) * (1 - b)
config.game_ld_parameters['conflict'] = a * b

config.game_sd_parameters['pedestrian_first'] = x * (1 - y)
config.game_sd_parameters['vehicle_first'] = y * (1 - x)
config.game_sd_parameters['compromise'] = (1 - x) * (1 - y)
config.game_sd_parameters['conflict'] = x * y

if __name__ == '__main__':
    NaSch = NaSch(config=config)
    NaSch.nasch_process()
