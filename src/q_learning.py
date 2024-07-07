import numpy as np
from config import VELOCITIES, ROAD_LENGTH, CAR_RANGE, MAX_VELOCITY, PEDESTRIAN_ENTER_PROB, PEDESTRIAN_EXIT_PROB
from environment import take_city_action, get_speed_state, get_transition_state, update_road


def get_transition_reward(action_speed, action_transition, position, road):
    velocity = VELOCITIES[action_speed]
    correct_position = min(position + velocity, ROAD_LENGTH)

    collision_detect_array = road[correct_position +
                                  1:correct_position + velocity + 1]
    reward = -20 if any(collision_detect_array) else 0

    if action_transition == 0:
        reward += 1
    elif action_transition == 1:
        reward -= 2
    elif action_transition == 2:
        reward -= 25

    return reward


def get_speed_reward(action_speed, position, road):
    velocity = VELOCITIES[action_speed]
    position = min(position + velocity, ROAD_LENGTH)
    collision_detect_array = road[position - velocity + 1:position + 1]

    if any(collision_detect_array):
        reward = -50
    else:
        reward = {0: -2, 1: 5, 2: 15}[action_speed]

    return reward


def q_learning_update(q_table, state, action, reward, learning_rate):
    q_table[state, action] += learning_rate * (reward - q_table[state, action])
    return q_table


def train_q_learning(num_episodes, q_table_speed, q_table_transition, learning_rate, discount_factor, epsilon_decay, min_epsilon):
    for episode in range(1, num_episodes + 1):
        state_speed_current = state_transition_current = position = 0
        road = np.zeros(ROAD_LENGTH + 1, dtype=int)
        done = False
        epsilon = 1
        score_transition = score_speed = 0

        while not done:
            state_transition_current, city_cover = get_transition_state(
                position, road)
            action_transition = np.argmax(q_table_transition[state_transition_current]) if np.random.rand(
            ) >= epsilon else np.random.randint(3)

            data = take_city_action(
                action_transition, city_cover, position, road)
            state_speed_current = get_speed_state(
                city_cover, data, position, road)
            action_speed = np.argmax(q_table_speed[state_speed_current]) if np.random.rand(
            ) >= epsilon else np.random.randint(3)

            transition_reward = get_transition_reward(
                action_speed, action_transition, position, road)
            speed_reward = get_speed_reward(action_speed, position, road)

            additional_reward = 4 * \
                state_speed_current if state_speed_current in [1, 2] else 0

            q_table_transition = q_learning_update(
                q_table_transition, state_transition_current, action_transition, transition_reward + additional_reward, learning_rate)
            q_table_speed = q_learning_update(
                q_table_speed, state_speed_current, action_speed, speed_reward, learning_rate)

            update_road(road, position, PEDESTRIAN_ENTER_PROB,
                        PEDESTRIAN_EXIT_PROB)

            score_speed += speed_reward
            score_transition += transition_reward

            if position >= ROAD_LENGTH:
                done = True

            epsilon = max(min_epsilon, epsilon * epsilon_decay)

        if episode % 20 == 0:
            print(
                f"Episode {episode} completed: Score speed is {score_speed} and score transition is {score_transition}.")

    return q_table_speed, q_table_transition
