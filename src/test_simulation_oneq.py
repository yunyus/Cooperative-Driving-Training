import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from config import ROAD_LENGTH, ENTER_PROB_RANGES, Q_TABLE_SPEED, Q_TABLE_TRANSITION, VELOCITIES, CITY_RANGE, PEDESTRIAN_EXIT_PROB
from environment import take_city_action, get_speed_state, get_transition_state, min_pedestrian_distance, update_road


def run_simulation():
    for enter_prob in ENTER_PROB_RANGES:
        data_list = []
        steps = positions = []
        optimal_policy = optimal_policy_locations = optimal_transition_ped = total_transition_ped = minimums = []
        pedestrian_enter_prob = enter_prob
        position = step = 0
        road = np.zeros(ROAD_LENGTH + 1, dtype=int)
        done = False

        while not done:
            state_transition_current, city_cover = get_transition_state(
                position, road)
            action_transition = np.argmax(
                Q_TABLE_TRANSITION[state_transition_current])
            data = take_city_action(
                action_transition, city_cover, position, road)
            optimal_transition_ped.append(
                0 if data is None else 1 if isinstance(data, int) else len(data))
            total_transition_ped.append(
                road[CITY_RANGE[0]:CITY_RANGE[1] + 1].sum())

            state_speed_current = get_speed_state(
                city_cover, data, position, road)
            action_speed = np.argmax(Q_TABLE_SPEED[state_speed_current])
            velocity = VELOCITIES[action_speed]
            position = min(position + velocity, ROAD_LENGTH)
            step += 1
            minimums.append(min_pedestrian_distance(position, road))

            optimal_policy += {0: 0, 1: 1, 3: 1, 2: 5}[state_speed_current]
            optimal_policy_locations.append(optimal_policy)

            steps.append(step)
            positions.append(position)
            update_road(road, position, pedestrian_enter_prob,
                        PEDESTRIAN_EXIT_PROB)

            if position >= ROAD_LENGTH:
                done = True
                print("Reached the end without hitting pedestrians!")
            elif any(road[position:position + 1]):
                print("Hit a pedestrian!")

            data = {
                'Steps': steps[-1],
                'Optimal Policy Locations': optimal_policy_locations[-1],
                'Modeled Policy Locations': positions[-1],
                'Semantic Transmission': optimal_transition_ped[-1],
                'Full Transmission': total_transition_ped[-1],
                'Minimum Pedestrian Range': minimums[-1]
            }
            data_list.append(data)

        df = pd.DataFrame(data_list)
        df.to_excel(
            f'data/simulation_data/oneq_test_entering_probability_{enter_prob}.xlsx', index=False)

        # Uncomment the following lines to plot the results
        # plot_results(steps, optimal_policy_locations, positions, optimal_transition_ped, total_transition_ped, minimums, enter_prob)


def plot_results(steps, optimal_policy_locations, positions, optimal_transition_ped, total_transition_ped, minimums, enter_prob):
    marker_step = 200
    plt.figure(figsize=(10, 15))

    plt.subplot(3, 1, 1)
    plt.grid(True)
    plt.plot(steps, optimal_policy_locations, label="Optimal")
    plt.plot(steps, positions, label="Model", linestyle="-.")
    plt.plot(range(1, len(steps) + 1), [CITY_RANGE[0]] * len(steps),
             'r--', marker='^', markevery=marker_step, linewidth=2)
    plt.plot(range(1, len(steps) + 1), [CITY_RANGE[1]] * len(steps),
             'r--', marker='v', markevery=marker_step, linewidth=2)
    plt.legend(['Optimal', 'Model', 'SC Obs Starts',
               'SC Obs Ends'], loc='upper right')
    plt.xlabel('Time indices')
    plt.ylabel('Position')
    plt.title(f'Position vs. Time for Enter Probability = {enter_prob}')

    plt.subplot(3, 1, 2)
    plt.plot(steps, optimal_transition_ped,
             label="Modeled Transmission", color='r')
    plt.plot(steps, total_transition_ped,
             label="Non-modeled Transmission", color='b')
    plt.axhline(y=1, color='black', linestyle='--', label="y=1")
    plt.legend(['Modeled Transmission', 'Non-modeled Transmission',
               'y=1'], loc='upper right')
    plt.xlabel('Time indices')
    plt.ylabel('Number of transmission')
    plt.grid(True)
    plt.title(
        f'Number of Transmission vs. Time for Enter Probability = {enter_prob}')

    plt.subplot(3, 1, 3)
    plt.grid(True)
    plt.plot(steps, minimums, label="Min Pedestrian Distance")
    plt.axhline(y=1, color='black', linestyle='--', label="y=1")
    plt.legend(['Min Pedestrian Distance', 'y=1'], loc='upper right')
    plt.xlabel('Time indices')
    plt.ylabel('Minimum distance to pedestrian')
    plt.ylim(0, 10)
    plt.yticks(np.arange(0, 10, 1.0))
    plt.title(
        f'Min distance to pedestrian vs. Time for Enter Probability = {enter_prob}')

    plt.rc('font', size=16)
    plt.tight_layout()
    plt.savefig(f'data/simulation_data/test_{enter_prob}.png')


if __name__ == "__main__":
    run_simulation()
