from utilities import Config
from models.NaSch import NaSch

config = Config
config.num_of_cells = 100
config.num_of_vehicles = 10
config.max_time_step = 100
config.max_speed = 4
config.p_slowdown = 0.3
config.pause_time = 0.3
config.cell_size = 5
config.conflict_zone = 90

if __name__ == '__main__':
    NaSch = NaSch(config=config)
    NaSch.initialization()
    NaSch.nasch_process()
