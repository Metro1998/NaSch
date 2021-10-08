from utilities import Config
from models.NaSch import NaSch

config = Config
config.num_of_cells = 20
config.num_of_cars = 12
config.max_time = 100
config.max_speed = 5
config.p_slowdown = 0.3
config.pause_time = 0.1
config.cell_size = 15

if __name__== '__main__':
    NaSch = NaSch()
    NaSch.initialization()
    NaSch.nasch_process()

