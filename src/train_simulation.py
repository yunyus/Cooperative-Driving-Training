import numpy as np
import matplotlib.pyplot as plt
from config import ROAD_LENGTH, NUM_ACTIONS_SPEED, NUM_STATES_SPEED, NUM_ACTIONS_TRANSITION, NUM_STATES_TRANSITION
from q_learning import train_q_learning
from config import Q_TABLE_SPEED, Q_TABLE_TRANSITION


def train_simulation():
    num_episodes = 500
    learning_rate = 0.01
    discount_factor = 0.9
    epsilon_decay = 0.9995
    min_epsilon = 0.001

    q_table_speed, q_table_transition = train_q_learning(
        num_episodes, Q_TABLE_SPEED, Q_TABLE_TRANSITION,
        learning_rate, discount_factor, epsilon_decay, min_epsilon
    )

    np.save('data/q_table_speed.npy', q_table_speed)
    np.save('data/q_table_transition.npy', q_table_transition)

    print("Training complete. Q-tables saved.")


if __name__ == "__main__":
    train_simulation()
