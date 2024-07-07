import numpy as np

# Environment parameters
ROAD_LENGTH = 1500
NUM_ACTIONS_SPEED = 3
NUM_STATES_SPEED = 4
VELOCITIES = [0, 1, 5]
MAX_VELOCITY = VELOCITIES[NUM_ACTIONS_SPEED - 1]

NUM_ACTIONS_TRANSITION = 3
NUM_STATES_TRANSITION = 4
CAR_RANGE = 2
CITY_RANGE = [500, 1300]

# Probabilities of pedestrians entering or exiting the road in each step
PEDESTRIAN_ENTER_PROB = 0.01
PEDESTRIAN_EXIT_PROB = 0.5
ENTER_PROB_RANGES = [0.005, 0.01, 0.02, 0.05, 0.1]

# Load Q-tables
Q_TABLE_SPEED = np.load('../data/q_table_speed.npy')
Q_TABLE_TRANSITION = np.load('../data/q_table_transition.npy')
