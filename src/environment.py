import numpy as np
from config import ROAD_LENGTH, CAR_RANGE, MAX_VELOCITY, CITY_RANGE


def take_city_action(action, city_cover, position, road):
    temp_road = road[position:]
    max_ind = min(ROAD_LENGTH - position + 1, len(temp_road))

    if action == 0:
        return None
    elif action == 1:
        last_ind = min(
            MAX_VELOCITY + 1 if city_cover else CAR_RANGE + 1, max_ind)
        for i in range(1, last_ind):
            if temp_road[i] == 1:
                return i
        return -1
    elif action == 2:
        return road[CITY_RANGE[0]:CITY_RANGE[1] + 1]


def get_speed_state(city_cover, data, position, road):
    temp_road = road[position:]
    max_ind = min(ROAD_LENGTH - position + 1, len(temp_road))

    if data is None:
        state = 3
        for i in range(1, min(CAR_RANGE + 1, max_ind)):
            if temp_road[i] == 1:
                return 0
        return state
    elif isinstance(data, int):
        if data == -1:
            return 2 if city_cover else 3
        elif data <= CAR_RANGE:
            return 0
        elif data <= MAX_VELOCITY:
            return 1
    else:
        state = 2 if city_cover else 3
        for i in range(1, min(MAX_VELOCITY + 1 if city_cover else CAR_RANGE + 1, max_ind)):
            if temp_road[i] == 1:
                if i <= CAR_RANGE:
                    return 0
                elif i <= MAX_VELOCITY:
                    return 1
        return state


def get_transition_state(position, road):
    temp_road = road[position:]
    min_data = min(CAR_RANGE + 1, ROAD_LENGTH - position + 1)
    see_pedestrian = any(temp_road[i] == 1 for i in range(1, min_data))
    cover_city_car = position + \
        CAR_RANGE >= CITY_RANGE[0] and position + \
        MAX_VELOCITY + 1 <= CITY_RANGE[1]

    if see_pedestrian and not cover_city_car:
        return 0, cover_city_car
    elif see_pedestrian and cover_city_car:
        return 1, cover_city_car
    elif not see_pedestrian and not cover_city_car:
        return 2, cover_city_car
    else:
        return 3, cover_city_car


def min_pedestrian_distance(position, road):
    smallest_index = next(
        (i for i in range(position+1, len(road)) if road[i] == 1), -1)
    return smallest_index - position if smallest_index != -1 else ROAD_LENGTH


def update_road(road, position, enter_prob, exit_prob):
    for idx in range(len(road)):
        if idx != position:
            if np.random.rand() < enter_prob:
                road[idx] = 1
            elif np.random.rand() < exit_prob:
                road[idx] = 0
