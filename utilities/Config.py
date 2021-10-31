# author metro(lhq)
# time 2021/10/7

class Config(object):
    """ Object to hold the config requirements for a MaSch Model """

    def __init__(self):
        self.num_of_cells = None  # The length of the road
        self.num_of_vehicles = None  # The number of vehicles in space
        self.max_time_step = None  # Maximum time step
        self.max_speed = None  # Maximum speed a car could reach
        self.p_slowdown = None  # Possibility of randomly slowing down
        self.pause_time = None  # Interval of refreshment
        self.cell_size = None  # The size of cell
        self.conflict_zone = None  # The index of the first cell of conflict zone
        self.peak_period = True  # Signature to demonstrate whether is peak period now
        self.distribution_parameters = {}
        self.game_ld_parameters = {}
        self.game_sd_parameters = {}
