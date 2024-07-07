import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

print_episode_parameter = 20


class Pedestrian:
    def __init__(self, direction):
        self.direction = direction
        self.speed = ped_velocity


# Define the environment parameters
road_length = 1500
road_width = 40
num_actions_speed = 3
num_states_speed = 4
velocities = [0, 1, 5]
max_velocity = velocities[num_actions_speed - 1]
ped_velocity = 1
danger_zone = 1

num_actions_transition = 3
num_states_transition = 4
car_range = 2
city_range = [500, 1300]

# Probabilities of pedestrians entering or exiting the road in each step
pedestrian_enter_prob = 0.01
pedestrian_exit_prob = 0.5
enter_prob_ranges = [0.005, 0.01, 0.02, 0.05, 0.1]

optimal_policy = 0
optimal_policy_locations = []
positions = []

# Statistical Purposes
optimal_transition_ped = []
total_transition_ped = []
minimums = []

# Initialize the Q-table with zeros
q_table_speed = np.load('q_table_speed.npy')
q_table_transition = np.load('q_table_transition.npy')


# Environment state
road = np.full((road_width, road_length + 1), None, dtype=Pedestrian)
position = [20, 0]


def take_city_action(action_transition, city_cover):

    global position
    temp_road = road[:, position[1]:]
    max_ind = min(road_length - position[1] + 1, len(temp_road[1]))

    if action_transition == 0:
        return None

    elif action_transition == 1:
        if city_cover:
            last_ind = min(max_velocity + 1, max_ind)
        else:
            last_ind = min(car_range + 1, max_ind)

        for i in range(1, last_ind):
            for j in range(-danger_zone, 0):
                if temp_road[position[0] + j][i] is not None and hasattr(temp_road[position[0] + j][i], 'direction') and temp_road[position[0] + j][i].direction == "down":
                    return i
            for j in range(danger_zone, 0):
                if temp_road[position[0] + j][i] is not None and hasattr(temp_road[position[0] + j][i], 'direction') and temp_road[position[0] + j][i].direction == "up":
                    return i
        return -1

    elif action_transition == 2:
        return road[city_range[0]:city_range[1] + 1]


def get_state_speed(city_cover, city_data):
    global position

    temp_road = road[:, position[1]:]
    max_ind = min(road_length - position[1] + 1, road_length)

    if city_data is None:
        state = 3

        for i in range(1, min(car_range + 1, max_ind)):
            for j in range(-danger_zone, 0):
                if temp_road[position[0] + j][i] is not None and hasattr(temp_road[position[0] + j][i], 'direction') and temp_road[position[0] + j][i].direction == "down":
                    return 0
            for j in range(1, 1 + danger_zone):
                if temp_road[position[0] + j][i] is not None and hasattr(temp_road[position[0] + j][i], 'direction') and temp_road[position[0] + j][i].direction == "up":
                    return 0
        return state

    elif type(city_data) == int:
        if city_data == -1:
            if (city_cover):
                return 2
            else:
                return 3
        elif city_data <= car_range:
            return 0
        elif city_data <= max_velocity:
            return 1

    else:
        if city_cover:
            state = 2
            for i in range(1, min(max_velocity + 1, max_ind)):
                for j in range(-danger_zone, 0):
                    if temp_road[position[0] + j][i] is not None and hasattr(temp_road[position[0] + j][i], 'direction') and temp_road[position[0] + j][i].direction == "down":
                        if i <= car_range:
                            return 0
                        elif i <= max_velocity:
                            return 1
                for j in range(1, 1 + danger_zone):
                    if temp_road[position[0] + j][i] is not None and hasattr(temp_road[position[0] + j][i], 'direction') and temp_road[position[0] + j][i].direction == "up":
                        if i <= car_range:
                            return 0
                        elif i <= max_velocity:
                            return 1
            return state
        else:
            state = 3
            for i in range(1, min(car_range + 1, max_ind)):
                for j in range(-danger_zone, 0):
                    if temp_road[position[0] + j][i] is not None and hasattr(temp_road[position[0] + j][i], 'direction') and temp_road[position[0] + j][i].direction == "down":
                        return 0
                for j in range(1, 1 + danger_zone):
                    if temp_road[position[0] + j][i] is not None and hasattr(temp_road[position[0] + j][i], 'direction') and temp_road[position[0] + j][i].direction == "up":
                        return 0
            return state


def get_state_transition():
    global position
    temp_road = road[:, position[1]:]

    # Check for pedestrians in triangular vision
    see_pedestrian = False
    # min(car_range + 1, len(temp_road[1]))
    for i in range(1, min(car_range, len(temp_road[1]))):
        for j in range(-danger_zone, 0):
            if temp_road[position[0] + j][i] is not None and hasattr(temp_road[position[0] + j][i], 'direction') and temp_road[position[0] + j][i].direction == "down":
                see_pedestrian = True
                break
        for j in range(1, 1 + danger_zone):
            if temp_road[position[0] + j][i] is not None and hasattr(temp_road[position[0] + j][i], 'direction') and temp_road[position[0] + j][i].direction == "up":
                see_pedestrian = True
                break
        if see_pedestrian:
            break

    # Check if city is covered by car range and max velocity
    cover_city_car = position[1] + car_range >= city_range[0] and position[1] + \
        max_velocity + 1 <= city_range[1]

    # Determine state based on pedestrian and city coverage
    if see_pedestrian and (not cover_city_car):
        return 0, cover_city_car
    elif see_pedestrian and cover_city_car:
        return 1, cover_city_car
    elif (not see_pedestrian) and (not cover_city_car):
        return 2, cover_city_car
    elif (not see_pedestrian) and cover_city_car:
        return 3, cover_city_car


def min_pedestrian_distance():

    smallest_index = -1

    temp_road = road[position[0], position[1]:]

    for i in range(0, len(temp_road)):
        if temp_road[i] != None:
            smallest_index = i
            break

    if smallest_index != -1:
        return smallest_index
    else:
        return road_length

# Q-learning algorithm


if __name__ == '__main__':

    for i in range(len(enter_prob_ranges)):

        data_list = []

        steps = []
        positions = []
        optimal_policy = 0
        optimal_policy_locations = []
        optimal_transition_ped = []
        total_transition_ped = []
        minimums = []

        pedestrian_enter_prob = enter_prob_ranges[i]

        state_speed_current = 0
        state_transition_current = 0

        step = 0
        road = np.full((road_width, road_length + 1), None, dtype=Pedestrian)
        position = [20, 0]
        done = False

        while not done:
            state_transition_current, city_cover = get_state_transition()
            action_transition = np.argmax(
                q_table_transition[state_transition_current])
            data = take_city_action(action_transition, city_cover)
            if data is None:
                optimal_transition_ped.append(0)
            elif type(data) == int:
                optimal_transition_ped.append(1)
            else:
                optimal_transition_ped.append(len(data))

            temp_road = road[:, city_range[0]:city_range[1] + 1]
            number_of_total_ped = np.count_nonzero(temp_road != None)
            total_transition_ped.append(number_of_total_ped)

            state_speed_current = get_state_speed(city_cover, data)
            action_speed = np.argmax(q_table_speed[state_speed_current])

            for idx in range(road_length):
                for row in range(road_width-1, 0, -ped_velocity):
                    if road[row-ped_velocity][idx] is not None and hasattr(road[row-ped_velocity][idx], 'direction') and road[row-ped_velocity][idx].direction == "down":
                        road[row][idx] = road[row-ped_velocity][idx]
                        road[row-ped_velocity][idx] = None
                    elif road[row-ped_velocity][idx] is not None and hasattr(road[row-ped_velocity][idx], 'direction') and road[row-ped_velocity][idx].direction == "up":
                        road[row][idx] = None
                        row[row-ped_velocity][idx] = None

                if np.random.rand() < pedestrian_enter_prob:
                    random = np.random.randint(1, 3)
                    if random == 1:
                        road[road_width-1][idx] = Pedestrian("up")
                    else:
                        road[0][idx] = Pedestrian("down")

            velocity = velocities[action_speed]
            position[1] = min(position[1] + velocity, road_length)
            step += 1

            minloc = min_pedestrian_distance()
            minimums.append(minloc)

            if state_speed_current == 0:
                optimal_policy += 0
                optimal_policy_locations.append(optimal_policy)
            elif state_speed_current == 1 or state_speed_current == 3:
                optimal_policy += 1
                optimal_policy_locations.append(optimal_policy)
            elif state_speed_current == 2:
                optimal_policy += 5
                optimal_policy_locations.append(optimal_policy)

            steps.append(step)
            positions.append(position[1])

            if position[1] >= road_length:
                done = True

            # Create a dictionary to store the data
            data = {
                'Steps': steps[-1],
                'Optimal Policy Locations': optimal_policy_locations[-1],
                'Modeled Policy Locations': positions[-1],
                'Semantic Transmission': optimal_transition_ped[-1],
                'Full Transmission': total_transition_ped[-1],
                'Minimum Pedestrian Range': minimums[-1]
            }

            # Append the dictionary to the data list
            data_list.append(data)

        # Create a DataFrame from the list of dictionaries
        df = pd.DataFrame(data_list)

        # Export the DataFrame to an Excel file
        df.to_excel(
            f'twoq_test_entering_probability_{enter_prob_ranges[i]}.xlsx', index=False)

        # print("Test")
        # marker_step = 200
        # plt.figure(figsize=(10, 15))
#
        # plt.subplot(3, 1, 1)
        # plt.grid(True)
        # plt.plot(steps, optimal_policy_locations, label="Optimal")
        # plt.plot(steps, positions, label="Model", linestyle="-.")
        # plt.plot(range(1, step + 1), [city_range[0]] * step, 'r--', marker='^',
        #          markevery=marker_step, linewidth=2)
        # plt.plot(range(1, step + 1), [city_range[1]] * step, 'r--', marker='v',
        #          markevery=marker_step, linewidth=2)
        # plt.legend(['Optimal', 'Model', 'SC Obs Starts',
        #            'SC Obs Ends'], loc='upper center')
        # plt.legend(bbox_to_anchor=(0.5, -0.2),
        #            loc='upper center', borderaxespad=0., ncol=4)
        # plt.xlabel('Time indices')
        # plt.ylabel('Position')
#
        # plt.subplot(3, 1, 2)
        # plt.yscale('log')  # Set y-axis to logarithmic scale
        # plt.plot(steps, optimal_transition_ped,
        #          label="Semantic Transmission", color='r')
        # plt.plot(steps, total_transition_ped,
        #          label="Full Transmission", color='b')
        # plt.axhline(y=1, color='black', linestyle='--', label="y=1")
        # plt.legend(['Semantic Transmission', 'Full Transmission',
        #            'y=1'], loc='upper center')
        # plt.legend(bbox_to_anchor=(0.5, -0.2),
        #            loc='upper center', borderaxespad=0., ncol=4)
        # plt.xlabel('Time indices')
        # plt.ylabel('Number of transmission')  # Update y-axis label
        # plt.grid(True)
#
        # plt.subplot(3, 1, 3)
        # plt.grid(True)
        # plt.plot(steps, minimums, label="Min Pedestrian Distance")
        # plt.axhline(y=1, color='black', linestyle='--', label="y=1")
        # plt.legend(['Min Pedestrian Distance', 'y=1'], loc='upper center')
        # plt.legend(bbox_to_anchor=(0.5, -0.2),
        #            loc='upper center', borderaxespad=0., ncol=4)
        # plt.xlabel('Time indices')
        # plt.ylabel('Minimum distance to pedestrian')
        # plt.ylim(0, 10)
        # plt.yticks(np.arange(0, 10, 1.0))
#
        # plt.rc('font', size=16)
#
        # plt.tight_layout()  # To prevent overlapping of labels and titles
#
        # # Save the current plot's data to the list
        # plt.savefig(f'oneq_v13_test_{i}.png')
#
