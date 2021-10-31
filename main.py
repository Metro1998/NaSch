from utilities.Config import Config
from models.NaSch import NaSch

config = Config()
config.num_of_cells = 50
config.num_of_vehicles = 10
config.max_time_step = 1800
config.max_speed = 3
config.p_slowdown = 0.3
config.pause_time = 0.3
config.cell_size = 5
config.conflict_zone = 40
config.peak_period = True
config.distribution_parameters['n_peak'] = 340 / 300
config.distribution_parameters['n_flat'] = 335 / 300
config.distribution_parameters['p_peak'] = 0.83
config.distribution_parameters['p_flat'] = 0.9
config.distribution_parameters['mu_peak'] = 16.50 / 300
config.distribution_parameters['mu_flat'] = 17.67 / 300

config.game_ld_parameters['pedestrian_first'] = 0.25
config.game_ld_parameters['vehicle_first'] = 0.25
config.game_ld_parameters['compromise'] = 0.25
config.game_ld_parameters['conflict'] = 0.25

config.game_sd_parameters['pedestrian_first'] = 0.25
config.game_sd_parameters['vehicle_first'] = 0.25
config.game_sd_parameters['compromise'] = 0.25
config.game_sd_parameters['conflict'] = 0.25


if __name__ == '__main__':
    NaSch = NaSch(config=config)
    NaSch.nasch_process()
