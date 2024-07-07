#!/usr/bin/env python

"""
Welcome, this is Yiğit and Yusuf's CARLA Simulator (Bilkent University)

    W            : throttle
    S            : brake
    A/D          : steer left/right
    Q            : toggle reverse
    Space        : hand-brake
    P            : toggle autopilot
    M            : toggle manual transmission
    ,/.          : gear up/down
    CTRL + W     : toggle constant velocity mode at 60 km/h

    L            : toggle next light type
    SHIFT + L    : toggle high beam
    Z/X          : toggle right/left blinker
    I            : toggle interior light

    TAB          : change sensor position
    ` or N       : next sensor
    [1-9]        : change to sensor [1-9]
    G            : toggle radar visualization
    C            : change weather (Shift+C reverse)
    Backspace    : change vehicle

    V            : Select next map layer (Shift+V reverse)
    B            : Load current selected map layer (Shift+B to unload)

    R            : toggle recording images to disk

    CTRL + R     : toggle recording of simulation (replacing any previous)
    CTRL + P     : start replaying last recorded simulation
    CTRL + +     : increments the start time of the replay by 1 second (+SHIFT = 10 seconds)
    CTRL + -     : decrements the start time of the replay by 1 second (+SHIFT = 10 seconds)

    F1           : toggle HUD
    F2           : toggle left - right arrow buttons
    H/?          : toggle help
    ESC          : quit
"""

from __future__ import print_function
import weakref
import re
import random
import math
import logging
import datetime
import collections
import argparse
from carla import ColorConverter as cc
import carla

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure

# ==============================================================================
# -- find carla module ---------------------------------------------------------
# ==============================================================================

import time
import glob
import os
import sys
import pandas as pd
import queue
from carla import ColorConverter
import math

# ==============================================================================
# -- global -------------------------------------------------------------------
# ==============================================================================

new_cnt = 0
tick_count = 0
cam_list = []
processed = False
MAX_SPEED_PED = 1.0
MIN_SPEED_PED = 0.9

# Create dataframe to store walker data
df = pd.DataFrame(
    columns=['Type', 'ID', 'Tick', 'Location X', 'Location Y', 'Direction X', 'Direction Y', 'Speed', 'Acceleration'])

new_df = pd.DataFrame(
    columns=['Type', 'ID', 'Tick', 'Location X', 'Location Y', 'Direction X', 'Direction Y', 'Speed', 'Acceleration'])

tick_parameters = {
    1: [3, 15],
    100: [2, 10],
    150: [1, 5],
    300: [1, 5],
    # 350: [1, 10],
    400: [2, 10],
}

ped_entring_prob = 0.6
crosswalk_prob = 0.8
speeds = []
transmissions = []
t = []
car_speeds_ind = [0, 4, 12]
current_pos_x = 0

# Create a unique folder for each run
# export_folder = 'lucifer' + str(time.time())
# os.makedirs(export_folder, exist_ok=True)

try:
    sys.path.append(
        glob.glob(os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/carla/dist/carla-*%d.%d-%s.egg' % (
            sys.version_info.major,
            sys.version_info.minor,
            'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

# ==============================================================================
# -- imports -------------------------------------------------------------------
# ==============================================================================


try:
    import pygame
    from pygame import mixer
    from pygame.locals import KMOD_CTRL
    from pygame.locals import KMOD_SHIFT
    from pygame.locals import K_0
    from pygame.locals import K_9
    from pygame.locals import K_BACKQUOTE
    from pygame.locals import K_BACKSPACE
    from pygame.locals import K_COMMA
    from pygame.locals import K_DOWN
    from pygame.locals import K_ESCAPE
    from pygame.locals import K_F1
    from pygame.locals import K_F2
    from pygame.locals import K_LEFT
    from pygame.locals import K_PERIOD
    from pygame.locals import K_RIGHT
    from pygame.locals import K_SLASH
    from pygame.locals import K_SPACE
    from pygame.locals import K_TAB
    from pygame.locals import K_UP
    from pygame.locals import K_a
    from pygame.locals import K_b
    from pygame.locals import K_c
    from pygame.locals import K_d
    from pygame.locals import K_g
    from pygame.locals import K_h
    from pygame.locals import K_i
    from pygame.locals import K_l
    from pygame.locals import K_m
    from pygame.locals import K_n
    from pygame.locals import K_p
    from pygame.locals import K_k
    from pygame.locals import K_j
    from pygame.locals import K_q
    from pygame.locals import K_r
    from pygame.locals import K_s
    from pygame.locals import K_v
    from pygame.locals import K_w
    from pygame.locals import K_x
    from pygame.locals import K_z
    from pygame.locals import K_MINUS
    from pygame.locals import K_EQUALS
except ImportError:
    raise RuntimeError(
        'cannot import pygame, make sure pygame package is installed')

try:
    import numpy as np
except ImportError:
    raise RuntimeError(
        'cannot import numpy, make sure numpy package is installed')

global_client = None
locked = False

# ==============================================================================
# -- Global functions ----------------------------------------------------------
# ==============================================================================
pygame.display.set_caption('Yusuf and Yiğit - Manual Control Car Simulator')


def find_weather_presets():
    rgx = re.compile('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)')
    def name(x): return ' '.join(m.group(0) for m in rgx.finditer(x))
    presets = [x for x in dir(carla.WeatherParameters)
               if re.match('[A-Z].+', x)]
    return [(getattr(carla.WeatherParameters, x), name(x)) for x in presets]


def get_actor_display_name(actor, truncate=250):
    name = ' '.join(actor.type_id.replace('_', '.').title().split('.')[1:])
    return (name[:truncate - 1] + u'\u2026') if len(name) > truncate else name


# ==============================================================================
# -- World ---------------------------------------------------------------------
# ==============================================================================
# Define tag values
TAGS = {15: 'large_vehicle', 14: 'small_vehicle',
        12: 'pedestrian', 18: 'motorcycle', 13: 'bicycle'}

OBJECTS = {
    'motorcycle': ['vehicle.yamaha.yzf',
                   'vehicle.vespa.zx125',
                   'vehicle.kawasaki.ninja',
                   'vehicle.harley-davidson.low_rider'],
    'small_vehicle': ['vehicle.toyota.prius', 'vehicle.tesla.model3', 'vehicle.seat.leon', 'vehicle.nissan.micra',
                      'vehicle.mini.cooper_s_2021', 'vehicle.mini.cooper_s', 'vehicle.micro.microlino',
                      'vehicle.mercedes.coupe', 'vehicle.lincoln.mkz_2020', 'vehicle.lincoln.mkz_2017',
                      'vehicle.ford.mustang', 'vehicle.ford.crown', 'vehicle.dodge.charger_police_2020',
                      'vehicle.dodge.charger_2020', 'vehicle.citroen.c3', 'vehicle.chevrolet.impala',
                      'vehicle.bmw.grandtourer', 'vehicle.audi.tt', 'vehicle.audi.etron', 'vehicle.audi.a2'],
    'large_vehicle': ['vehicle.volkswagen.t2_2021', 'vehicle.volkswagen.t2', 'vehicle.tesla.cybertruck',
                      'vehicle.nissan.patrol_2021', 'vehicle.nissan.patrol', 'vehicle.mercedes.sprinter',
                      'vehicle.mercedes.coupe_2020', 'vehicle.jeep.wrangler_rubicon', 'vehicle.ford.ambulance',
                      'vehicle.carlamotors.firetruck', 'vehicle.carlamotors.carlacola'],
    'bicycle': ['vehicle.gazelle.omafiets', 'vehicle.diamondback.century', 'vehicle.bh.crossbike']
}

FREQS = {'large_vehicle': 0.3, 'small_vehicle': 0.3,
         'motorcycle': 0.2, 'bicycle': 0.2}

# Q Table Parameters
danger_zone = [-198, -192]
city_range = [-20, 100]
car_speeds = [carla.Vector3D(0, 0, 0), carla.Vector3D(
    4, 0, 0), carla.Vector3D(12, 0, 0)]

temp_car_transform = [0, 1, 5]
car_max_speed = 5
car_range = 2
carla_car_offset = 3

q_table_speed = np.load('q_table_speed.npy')

location_offset = 81

optimal_policy = 0
optimal_policy_locations = []
positions = []
data_list = []

# Statistical Purposes
optimal_transition_ped = []
total_transition_ped = []
minimums = []
projected_minimums = []
y_minimums = []


def spawn_pedestrians_general(self, number, isCross):
    for i in range(number):
        isLeft = random.choice([True, False])
        if isLeft:
            self.spawn_pedestrians_left(isCross)
        else:
            self.spawn_pedestrians_right(isCross)

        # Function to process walker data and update the dataframe


# def process_data(self):
#     walkers = self.world.get_actors().filter('walker.*')
#     vehicles = self.world.get_actors().filter('vehicle.*')
#
#     for walker in walkers:
#         # Get walker ID, location, and direction
#         walker_id = walker.id
#         location = walker.get_location()
#         direction = walker.get_control().direction
#         speed = walker.get_control().speed
#
#         # Append data to dataframe
#         df.loc[len(df)] = ["Walker", walker_id, tick_count, location.x,
#                            location.y, direction.x, direction.y, speed, 0]
#
#     for vehicle in vehicles:
#         # Get vehicle ID, location, and direction
#         vehicle_id = vehicle.id
#         location = vehicle.get_location()
#         direction = vehicle.get_transform().rotation.get_forward_vector()
#         speed = vehicle.get_velocity()
#         acceleration = vehicle.get_acceleration()
#
#         # Append data to dataframe
#         df.loc[len(df)] = ["Vehicle", vehicle_id, tick_count,
#                            location.x, location.y, 0, 0, speed, acceleration]
#
#     # Export DataFrame to Excel inside the created folder
#     df.to_excel(os.path.join(export_folder, 'data.xlsx'), index=False)


class World(object):

    def __init__(self, carla_world, hud, args):

        global tick_count
        tick_count = 0

        self.world = carla_world
        self.vehicles = None
        self.actor_role_name = args.rolename
        try:
            self.map = self.world.get_map()
        except RuntimeError as error:
            print('RuntimeError: {}'.format(error))
            print('  The server could not send the OpenDRIVE (.xodr) file:')
            print(
                '  Make sure it exists, has the same name of your town, and is correct.')
            sys.exit(1)
        self.hud = hud
        self.player = None
        self.collision_sensor = None
        self.lane_invasion_sensor = None
        self.gnss_sensor = None
        self.imu_sensor = None
        self.radar_sensor = None
        self.camera_manager = None
        self._weather_presets = find_weather_presets()
        self._weather_index = 0
        self._actor_filter = args.filter
        self._gamma = args.gamma
        self.restart()
        self.world.on_tick(hud.on_world_tick)
        self.recording_enabled = False
        self.recording_start = 0
        self.constant_velocity_enabled = False
        self.current_map_layer = 0
        self.map_layer_names = [
            carla.MapLayer.NONE,
            carla.MapLayer.Buildings,
            carla.MapLayer.Decals,
            carla.MapLayer.Foliage,
            carla.MapLayer.Ground,
            carla.MapLayer.ParkedVehicles,
            carla.MapLayer.Particles,
            carla.MapLayer.Props,
            carla.MapLayer.StreetLights,
            carla.MapLayer.Walls,
            carla.MapLayer.All
        ]

        global cam_list

        # Add Camera
        blueprint_library = self.world.get_blueprint_library()
        camera_pitch = -30
        camera_rgb = blueprint_library.find('sensor.camera.rgb')
        camera_rgb.set_attribute("image_size_x", str(640))
        camera_rgb.set_attribute("image_size_y", str(360))
        camera_rgb.set_attribute('fov', str(120))

        camera_sem = blueprint_library.find(
            'sensor.camera.semantic_segmentation')
        camera_sem.set_attribute("image_size_x", str(640))
        camera_sem.set_attribute("image_size_y", str(360))
        camera_sem.set_attribute('fov', str(120))

        # Pole 1
        camera_transform = carla.Transform(carla.Location(
            x=40, y=-200, z=6), carla.Rotation(camera_pitch, 180, 0))
        # self.camera_pole1_rgb = self.world.spawn_actor(camera_rgb, camera_transform)
        # self.image_queue_pole1_rgb = queue.Queue()
        # self.camera_pole1_rgb.listen(self.image_queue_pole1_rgb.put)
        # cam_list.append(self.camera_pole1_rgb)
#
        # self.camera_pole1_sem = self.world.spawn_actor(camera_sem, camera_transform)
        # self.image_queue_pole1_sem = queue.Queue()
        # self.camera_pole1_sem.listen(self.image_queue_pole1_sem.put)
        # cam_list.append(self.camera_pole1_sem)
#
        # Pole 2
        # camera_transform = carla.Transform(carla.Location(x=40, y=-200, z=6), carla.Rotation(camera_pitch, 0, 0))
        # self.camera_pole2_rgb = self.world.spawn_actor(camera_rgb, camera_transform)
        # self.image_queue_pole2_rgb = queue.Queue()
        # self.camera_pole2_rgb.listen(self.image_queue_pole2_rgb.put)
        # cam_list.append(self.camera_pole2_rgb)
#
        # self.camera_pole2_sem = self.world.spawn_actor(camera_sem, camera_transform)
        # self.image_queue_pole2_sem = queue.Queue()
        # self.camera_pole2_sem.listen(self.image_queue_pole2_sem.put)
        # cam_list.append(self.camera_pole2_sem)

        # Front Camera
        # front_location = carla.Location(2, 0, 1)
        # front_rotation = carla.Rotation(0, 180, 0)
        # front_transform = carla.Transform(front_location, front_rotation)
        # vcls = self.world.get_actors().filter('vehicle.*')
        # vcl = vcls[0]
#
        # self.camera_front_rgb = self.world.spawn_actor(camera_rgb, front_transform,
        #                                                attach_to=vcl, attachment_type=carla.AttachmentType.SpringArm)
        #
        # self.data_queue = queue.Queue()
        #
        # self.image_queue_front_rgb = queue.Queue()
        # self.camera_front_rgb.listen(lambda data: self.new_process(data))
        # cam_list.append(self.camera_front_rgb)

        # self.camera_front_sem = self.world.spawn_actor(camera_sem, front_transform, attach_to=vcl,
        #                                                attachment_type=carla.AttachmentType.SpringArm)
        # self.image_queue_front_sem = queue.Queue()
        # self.camera_front_sem.listen(self.image_queue_front_sem.put)
        # cam_list.append(self.camera_front_sem)

    # def new_process(self, data):
#
#
    #     global new_cnt
    #     new_cnt = new_cnt + 1
#
    #     self.image_queue_front_rgb.put(data)
    #
    #     walkers = self.world.get_actors().filter('walker.*')
    #     vehicles = self.world.get_actors().filter('vehicle.*')
#
    #     for walker in walkers:
    #         # Get walker ID, location, and direction
    #         walker_id = walker.id
    #         location = walker.get_location()
    #         direction = walker.get_control().direction
    #         speed = walker.get_control().speed
#
    #         # Append data to dataframe
    #         # df.loc[len(df)] = ["Walker", walker_id, new_cnt, location.x, location.y, direction.x, direction.y, speed, 0]
    #         self.data_queue.put(["Walker",walker_id, new_cnt, location.x, location.y, direction.x, direction.y, speed, 0])
#
    #     for vehicle in vehicles:
    #         # Get vehicle ID, location, and direction
    #         vehicle_id = vehicle.id
    #         location = vehicle.get_location()
    #         direction = vehicle.get_transform().rotation.get_forward_vector()
    #         speed = vehicle.get_velocity()
    #         acceleration = vehicle.get_acceleration()
#
    #         # Append data to dataframe
    #         # df.loc[len(df)] = ["Vehicle", vehicle_id, tick_count, location.x, location.y, 0, 0, speed, acceleration]
    #         self.data_queue.put(["Vehicle", vehicle_id, new_cnt, location.x, location.y, 0, 0, speed, acceleration])

    def spawn_pedestrians_right(self, isCross):

        blueprints_walkers = self.world.get_blueprint_library().filter("walker.pedestrian.*")
        walker_bp = random.choice(blueprints_walkers)

        for i in range(1):
            walker_bp = random.choice(blueprints_walkers)

            min_x = -60
            max_x = 150
            min_y = -188
            max_y = -183

            if isCross:
                isFirstCross = random.choice([True, False])
                if isFirstCross:
                    min_x = -14
                    max_x = -10.5
                else:
                    min_x = 17
                    max_x = 20.5

            # Randomly select the position for the pedestrian
            x = random.uniform(min_x, max_x)
            y = random.uniform(min_y, max_y)

            spawn_point = carla.Transform(carla.Location(x, y, 2.0))

            while (-10 < spawn_point.location.x < 17) or (70 < spawn_point.location.x < 100):
                x = random.uniform(min_x, max_x)
                y = random.uniform(min_y, max_y)
                spawn_point = carla.Transform(carla.Location(x, y, 2.0))

            if spawn_point:
                npc = self.world.try_spawn_actor(walker_bp, spawn_point)

            if npc is not None:
                ped_control = carla.WalkerControl()
                ped_control.speed = random.uniform(
                    MIN_SPEED_PED, MAX_SPEED_PED)
                ped_control.direction.y = -1
                ped_control.direction.x = 0.15
                npc.apply_control(ped_control)
                npc.set_simulate_physics(True)

    def spawn_pedestrians_left(self, isCross):

        blueprints_walkers = self.world.get_blueprint_library().filter("walker.pedestrian.*")
        walker_bp = random.choice(blueprints_walkers)

        for i in range(1):
            walker_bp = random.choice(blueprints_walkers)
            # spawn_points = self.world.get_map().get_spawn_points()  # Assuming spawn_points is defined elsewhere
            # npc = self.world.try_spawn_actor(walker_bp, random.choice(spawn_points))

            min_x = -50
            max_x = 140
            min_y = -216
            max_y = -210

            if (isCross):
                isFirstCross = random.choice([True, False])
                if isFirstCross:
                    min_x = -14
                    max_x = -10.5
                else:
                    min_x = 17
                    max_x = 20.5

            # Randomly select the position for the pedestrian
            x = random.uniform(min_x, max_x)
            y = random.uniform(min_y, max_y)

            spawn_point = carla.Transform(carla.Location(x, y, 2.0))

            while (-10 < spawn_point.location.x < 17) or (70 < spawn_point.location.x < 100):
                x = random.uniform(min_x, max_x)
                y = random.uniform(min_y, max_y)
                spawn_point = carla.Transform(carla.Location(x, y, 2.0))

            if spawn_point:
                npc = self.world.try_spawn_actor(walker_bp, spawn_point)

            if npc is not None:
                ped_control = carla.WalkerControl()
                ped_control.speed = random.uniform(
                    MIN_SPEED_PED, MAX_SPEED_PED)
                ped_control.direction.y = 1
                ped_control.direction.x = -0.05
                npc.apply_control(ped_control)
                npc.set_simulate_physics(True)

    # def processCameraQueue(self):
    #    global tick_count
    #    global cam_list
#
    #    global processed
    #    processed = True
#
    #    if tick_count != 0:
#
    #        # Destroy the cameras
    #        for cam in cam_list:
    #            cam.stop()
#
    #        cnt = 0
#
    #        while (self.data_queue.empty() == False):
    #            data = self.data_queue.get()
    #            if (data[2] % 10 == 0):
    #                new_df.loc[len(new_df)] = data
#
    #        while (self.image_queue_front_rgb.empty() == False):
    #            cnt = cnt + 1
    #            image_front_rgb = self.image_queue_front_rgb.get()
    #            if (cnt % 10 == 0):
    #                image_front_rgb.save_to_disk(
    #                    export_folder + f'\\front_rgb_{cnt:06d}.png')
#
    #        new_df.to_excel(os.path.join(
    #            export_folder, 'data.xlsx'), index=False)
#
           # cnt = 0
           # while (self.image_queue_front_sem.empty() == False):
           #     cnt = cnt + 1
           #     image_front_sem = self.image_queue_front_sem.get()
           #     if (cnt % 10 == 0):
           #         image_front_sem.save_to_disk(export_folder + f'\\front_sem_{cnt:06d}.png',
           #                                      carla.ColorConverter.CityScapesPalette)

           #     if (cnt >= tick_count):
           #         break

           # cnt = 0
           # while (self.image_queue_pole1_rgb.empty() == False):
           #     cnt = cnt + 1
           #     image_pole1_rgb = self.image_queue_pole1_rgb.get()
           #     if (cnt % 10 == 0):
           #         image_pole1_rgb.save_to_disk(export_folder + f'\\pole1_rgb_{cnt:06d}.png')

           #     if (cnt >= tick_count):
           #         break

           # cnt = 0
           # while (self.image_queue_pole1_sem.empty() == False):
           #     cnt = cnt + 1
           #     image_pole1_sem = self.image_queue_pole1_sem.get()
           #     if (cnt % 10 == 0):
           #         image_pole1_sem.save_to_disk(export_folder + f'\\pole1_sem_{cnt:06d}.png',
           #                                      carla.ColorConverter.CityScapesPalette)

           #     if (cnt >= tick_count):
           #         break

           # cnt = 0
           # while (self.image_queue_pole2_rgb.empty() == False):
           #     cnt = cnt + 1
           #     image_pole2_rgb = self.image_queue_pole2_rgb.get()
           #     if (cnt % 10 == 0):
           #         image_pole2_rgb.save_to_disk(export_folder + f'\\pole2_rgb_{cnt:06d}.png')

           #     if (cnt >= tick_count):
           #         break

           # cnt = 0
           # while (self.image_queue_pole2_sem.empty() == False):
           #     cnt = cnt + 1
           #     image_pole2_sem = self.image_queue_pole2_sem.get()
           #     if (cnt % 10 == 0):
           #         image_pole2_sem.save_to_disk(export_folder + f'\\pole2_sem_{cnt:06d}.png',
           #                                      carla.ColorConverter.CityScapesPalette)

           #     if (cnt >= tick_count):
           #         break

            # for index, row in df.iterrows():
            #    if row['Tick'] % 10 == 0:
            #        new_df.loc[len(new_df)] = row

    def restart(self):

        # self.processCameraQueue()

        df = pd.DataFrame(data_list)
        timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
        df.to_excel(f'threeq_test_carla_{timestamp}.xlsx', index=False)

        global tick_count
        tick_count = 0

        global new_cnt
        new_cnt = 0

        walkers_list = self.world.get_actors().filter('walker.*')
        for actor in walkers_list:
            actor.destroy()

        self.player_max_speed = 1.589
        self.player_max_speed_fast = 3.713
        # Keep same camera config if the camera manager exists.
        cam_index = self.camera_manager.index if self.camera_manager is not None else 0
        cam_pos_index = self.camera_manager.transform_index if self.camera_manager is not None else 0
        # Get a random blueprint.
        blueprint = random.choice(
            self.world.get_blueprint_library().filter("vehicle.ford.mustang"))
        blueprint.set_attribute('role_name', self.actor_role_name)
        if blueprint.has_attribute('color'):
            color = random.choice(
                blueprint.get_attribute('color').recommended_values)
            blueprint.set_attribute('color', color)
        if blueprint.has_attribute('driver_id'):
            driver_id = random.choice(
                blueprint.get_attribute('driver_id').recommended_values)
            blueprint.set_attribute('driver_id', driver_id)
        if blueprint.has_attribute('is_invincible'):
            blueprint.set_attribute('is_invincible', 'false')
        # set the max speed
        if blueprint.has_attribute('speed'):
            self.player_max_speed = float(
                blueprint.get_attribute('speed').recommended_values[1])
            self.player_max_speed_fast = float(
                blueprint.get_attribute('speed').recommended_values[2])
        else:
            print("No recommended values for 'speed' attribute")
        # Spawn the player.
        if self.player is not None:
            if not self.map.get_spawn_points():
                print('There are no spawn points available in your map/town.')
                print('Please add some Vehicle Spawn Point to your UE4 scene.')
                sys.exit(1)
            spawn_points = self.map.get_spawn_points()
            spawn_point = random.choice(
                spawn_points) if spawn_points else carla.Transform()
            spawn_point.location.x = -81.0
            spawn_point.location.y = -195.0
            spawn_point.location.z += 2.0
            spawn_point.rotation.roll = 0.0
            spawn_point.rotation.pitch = 0.0
            spawn_point.rotation.yaw = 0.0
            self.destroy()
            self.player = self.world.try_spawn_actor(blueprint, spawn_point)

        while self.player is None:
            if not self.map.get_spawn_points():
                print('There are no spawn points available in your map/town.')
                print('Please add some Vehicle Spawn Point to your UE4 scene.')
                sys.exit(1)
            spawn_points = self.map.get_spawn_points()
            spawn_point = random.choice(
                spawn_points) if spawn_points else carla.Transform()
            spawn_point.location.x = -81.0
            spawn_point.location.y = -195.0
            spawn_point.location.z += 2.0
            spawn_point.rotation.yaw = 0.0

            self.player = self.world.try_spawn_actor(blueprint, spawn_point)
        # Set up the sensors.
        self.collision_sensor = CollisionSensor(self.player, self.hud)
        self.lane_invasion_sensor = LaneInvasionSensor(self.player, self.hud)
        self.gnss_sensor = GnssSensor(self.player)
        self.imu_sensor = IMUSensor(self.player)
        self.camera_manager = CameraManager(self.player, self.hud, self._gamma)
        self.camera_manager.transform_index = cam_pos_index
        self.camera_manager.set_sensor(cam_index, notify=False)
        actor_type = get_actor_display_name(self.player)
        self.hud.notification(actor_type)

        vehicles = self.world.get_actors().filter('vehicle.*')
        current_vehicle = vehicles[0]
        current_vehicle.set_target_velocity(car_speeds[0])
        current_vehicle.set_simulate_physics(False)

        global location_offset
        location_offset = current_vehicle.get_location().x
        location_offset = location_offset * -1

    def next_weather(self, reverse=False):
        self._weather_index += -1 if reverse else 1
        self._weather_index %= len(self._weather_presets)
        preset = self._weather_presets[self._weather_index]
        self.hud.notification('Weather: %s' % preset[1])
        self.player.get_world().set_weather(preset[0])

    def next_map_layer(self, reverse=False):
        self.current_map_layer += -1 if reverse else 1
        self.current_map_layer %= len(self.map_layer_names)
        selected = self.map_layer_names[self.current_map_layer]
        self.hud.notification('LayerMap selected: %s' % selected)

    def load_map_layer(self, unload=False):
        selected = self.map_layer_names[self.current_map_layer]
        if unload:
            self.hud.notification('Unloading map layer: %s' % selected)
            self.world.unload_map_layer(selected)
        else:
            self.hud.notification('Loading map layer: %s' % selected)
            self.world.load_map_layer(selected)

    def toggle_radar(self):
        if self.radar_sensor is None:
            self.radar_sensor = RadarSensor(self.player)
        elif self.radar_sensor.sensor is not None:
            self.radar_sensor.sensor.destroy()
            self.radar_sensor = None

    def takeAction(self):

        walkers = self.world.get_actors().filter('walker.*')

        vehicles = self.world.get_actors().filter('vehicle.*')
        current_vehicle = vehicles[0]
        current_location = current_vehicle.get_location()

        walkers = [x for x in walkers
                   if x.get_location().y > danger_zone[0] and x.get_location().y < danger_zone[1]

                   and not ((x.get_location().y > current_location.y and x.get_control().direction.y == 1) or
                            (x.get_location().y < current_location.y and x.get_control().direction.y == -1))

                   and carla_car_offset < x.get_location().x - current_location.x < 10
                   ]

        global current_pos_x
        current_pos_x = current_location.x

        min_pedestrian = None
        min_distance = 250  # road_length
        action_state = 0

        global transmissions

        for walker in walkers:
            walker_location = walker.get_location()

            # if walker_location.x <= current_location.x + 2:
            # sed_control = carla.WalkerControl()
            # ped_control.speed = 0
            # ped_control.direction.y = 0
            # ped_control.direction.x = 0
            # walker.apply_control(ped_control)
            #    continue

            distance = walker_location.x - current_location.x
            if distance < min_distance:
                min_distance = distance
                min_pedestrian = walker

        if min_pedestrian is not None:

            if min_distance <= car_range + carla_car_offset:
                transmissions.append(0)
                action_state = 0

            elif city_range[0] - carla_car_offset <= current_location.x <= city_range[1] - carla_car_offset:
                if min_distance <= car_max_speed + carla_car_offset:
                    transmissions.append(1)
                    action_state = 1
                else:
                    transmissions.append(1)
                    action_state = 2
            else:
                transmissions.append(0)
                action_state = 3

        if min_pedestrian is None:

            if city_range[0] - carla_car_offset <= current_location.x <= city_range[1] - carla_car_offset:
                transmissions.append(1)
                action_state = 2
            else:
                transmissions.append(0)
                action_state = 3

        walker_dist = []

        if min_pedestrian is None:

            dist_temp_walkers = self.world.get_actors().filter('walker.*')
            if len(dist_temp_walkers) == 0:
                minimums.append(250)
                projected_minimums.append(250)
                y_minimums.append(250)
            else:
                dist_temp_walkers = [x for x in dist_temp_walkers if
                                     ((x.get_location().y) < current_location.y and x.get_control().direction.y >= 0) or
                                     ((x.get_location().y) >
                                         current_location.y and x.get_control().direction.y <= 0)]

                dist_temp_walkers = [
                    x for x in dist_temp_walkers if x.get_location().x >= current_location.x + carla_car_offset]

                if len(dist_temp_walkers) == 0:
                    minimums.append(250)
                    projected_minimums.append(250)
                    y_minimums.append(250)
                else:
                    for i in dist_temp_walkers:
                        x = abs(i.get_location().x - current_location.x)
                        y = abs(i.get_location().y - current_location.y)
                        distance = math.sqrt(x*x + y*y)
                        walker_dist.append(
                            {'distance': distance, 'x': x, 'y': y})

                    # Find minimum distance and corresponding x, y values
                    min_walker = min(walker_dist, key=lambda x: x['distance'])

                    # Store minimum values
                    minimums.append(min_walker['distance'])
                    projected_minimums.append(min_walker['x'])
                    y_minimums.append(abs(min_walker['y']))

        else:
            x = min_pedestrian.get_location().x - current_location.x
            y = min_pedestrian.get_location().y - current_location.y
            minimums.append((x**2 + y**2)**0.5)
            projected_minimums.append(x)
            y_minimums.append(y)

        dist_walker_speed_alignment = self.world.get_actors().filter('walker.*')
        dist_walker_speed_alignment = [x for x in dist_walker_speed_alignment if
                                       ((x.get_location().y) < current_location.y and x.get_control().direction.y >= 0) or
                                       ((x.get_location().y) >
                                        current_location.y and x.get_control().direction.y <= 0)]

        if action_state == 0 and min_pedestrian is not None:
            for walker in dist_walker_speed_alignment:
                walker_location = walker.get_location()
                walker_control = walker.get_control()

                if (current_location.x - carla_car_offset < walker_location.x < min_pedestrian.get_location().x):
                    walker_control.speed = 0
                    walker.apply_control(walker_control)

        elif action_state != 0:

            for walker in dist_walker_speed_alignment:
                walker_location = walker.get_location()
                walker_control = walker.get_control()

                if walker_location.x < current_location.x - carla_car_offset and walker.get_control().speed == 0:
                    walker_control.speed = random.uniform(
                        MIN_SPEED_PED, MAX_SPEED_PED)
                    walker.apply_control(walker_control)

        isZeroWalker = self.world.get_actors().filter('walker.*')
        isZeroWalker = [x for x in isZeroWalker if x.get_control(
        ).speed == 0 and x.get_location().x < current_location.x - 4]
        print(len(isZeroWalker))

        optimal_transition_ped.append(transmissions[-1])

        # number of pedestrians in city range
        city_range_walkers_data = self.world.get_actors().filter('walker.*')
        in_range_walkers = [x for x in city_range_walkers_data if city_range[0]
                            <= x.get_location().x <= city_range[1]]
        total_transition_ped.append(len(in_range_walkers))

        action_speed = np.argmax(q_table_speed[action_state])
        # current_vehicle.set_target_velocity(car_speeds[action_speed])
        current_vehicle.set_transform(carla.Transform(carla.Location(
            x=current_location.x + temp_car_transform[action_speed], y=current_location.y, z=0)))

        global speeds
        # speeds.append(car_speeds_ind[action_speed])
        speeds.append(temp_car_transform[action_speed])
        positions.append((current_location.x + location_offset))
        optimal_policy_locations.append((current_location.x + location_offset))
        t.append(tick_count)

        # Create a dictionary to store the data
        data = {
            'Steps_1': t[-1],
            'Optimal Policy Locations': optimal_policy_locations[-1],
            'Modeled Policy Locations': positions[-1],
            ' ': ' ',
            'Steps_2': t[-1],
            'Semantic Transmission': optimal_transition_ped[-1],
            'Full Transmission': total_transition_ped[-1],
            '   ': '   ',
            'Steps_3': t[-1],
            'Minimum Pedestrian Range': minimums[-1],
            'Projected Minimum Pedestrian Range': projected_minimums[-1]
        }

        data_list.append(data)

    def tick(self, clock):
        self.hud.tick(self, clock)

        global tick_count
        tick_count = tick_count + 1

        # if tick_count % 10 == 0:
        #     process_data(self)

        # if tick_count % 5 == 0:
        #     self.takeAction()

        self.takeAction()

        walkers_list = self.world.get_actors().filter('walker.*')
        # Get the player's location
        for actor in walkers_list:
            player_transform = actor.get_transform()
            player_direction = actor.get_control().direction
            player_location = player_transform.location

            if (player_location.y < -210 and player_direction.y < 0):
                print("destroyed from left")
                actor.destroy()

            elif (player_location.y > -191 and player_direction.y > 0):
                print("destroyed from right")
                actor.destroy()

        # if tick_count in tick_parameters:
        #     param = tick_parameters[tick_count][0]
        #     spawn_pedestrians_general(self, param, False)

        # if tick_count in tick_parameters:
        #     param2 = tick_parameters[tick_count][1]
        #     spawn_pedestrians_general(self, param2, True)

        if random.random() < ped_entring_prob:

            if random.random() < crosswalk_prob:
                spawn_pedestrians_general(self, 1, True)

            else:
                spawn_pedestrians_general(self, 1, False)

        # Get the player's transform
        player_transform = self.player.get_transform()

        # Get the player's location
        player_location = player_transform.location

        if player_location.x > 169.0:
            self.restart()

    def render(self, display):
        self.camera_manager.render(display)
        self.hud.render(display)

    def destroy_sensors(self):
        self.camera_manager.sensor.destroy()
        self.camera_manager.sensor = None
        self.camera_manager.index = None

    def destroy(self):
        if self.radar_sensor is not None:
            self.toggle_radar()
        sensors = [
            self.camera_manager.sensor,
            self.collision_sensor.sensor,
            self.lane_invasion_sensor.sensor,
            self.gnss_sensor.sensor,
            self.imu_sensor.sensor]
        for sensor in sensors:
            if sensor is not None:
                sensor.stop()
                sensor.destroy()
        if self.player is not None:
            self.player.destroy()


# ==============================================================================
# -- KeyboardControl -----------------------------------------------------------
# ==============================================================================


class KeyboardControl(object):
    """Class that handles keyboard input."""

    def __init__(self, world, start_in_autopilot):
        self._carsim_enabled = False
        self._carsim_road = False
        self._autopilot_enabled = start_in_autopilot
        if isinstance(world.player, carla.Vehicle):
            self._control = carla.VehicleControl()
            self._lights = carla.VehicleLightState.NONE
            world.player.set_autopilot(self._autopilot_enabled)
            world.player.set_light_state(self._lights)
        elif isinstance(world.player, carla.Walker):
            self._control = carla.WalkerControl()
            self._autopilot_enabled = False
            self._rotation = world.player.get_transform().rotation
        else:
            raise NotImplementedError("Actor type not supported")
        self._steer_cache = 0.0
        world.hud.notification("Press 'H' or '?' for help.", seconds=4.0)

    def parse_events(self, client, world, clock):

        if isinstance(self._control, carla.VehicleControl):
            current_lights = self._lights
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return True
            elif event.type == pygame.KEYUP:
                if self._is_quit_shortcut(event.key):
                    return True
                elif event.key == K_BACKSPACE:
                    if self._autopilot_enabled:
                        world.player.set_autopilot(False)
                        world.restart()
                        world.player.set_autopilot(True)
                    else:
                        world.restart()
                elif event.key == K_F1:
                    world.hud.toggle_info()
                elif event.key == K_F2:
                    global locked
                    locked = not locked
                elif event.key == K_v and pygame.key.get_mods() & KMOD_SHIFT:
                    world.next_map_layer(reverse=True)
                elif event.key == K_v:
                    world.next_map_layer()
                elif event.key == K_b and pygame.key.get_mods() & KMOD_SHIFT:
                    world.load_map_layer(unload=True)
                elif event.key == K_b:
                    world.load_map_layer()
                elif event.key == K_h or (event.key == K_SLASH and pygame.key.get_mods() & KMOD_SHIFT):
                    world.hud.help.toggle()
                elif event.key == K_TAB:
                    world.camera_manager.toggle_camera()
                elif event.key == K_c and pygame.key.get_mods() & KMOD_SHIFT:
                    world.next_weather(reverse=True)
                elif event.key == K_c:
                    world.next_weather()
                elif event.key == K_g:
                    world.toggle_radar()
                elif event.key == K_BACKQUOTE:
                    world.camera_manager.next_sensor()
                elif event.key == K_n:
                    world.camera_manager.next_sensor()
                elif event.key == K_w and (pygame.key.get_mods() & KMOD_CTRL):
                    if world.constant_velocity_enabled:
                        world.player.disable_constant_velocity()
                        world.constant_velocity_enabled = False
                        world.hud.notification(
                            "Disabled Constant Velocity Mode")
                    else:
                        world.player.enable_constant_velocity(
                            carla.Vector3D(17, 0, 0))
                        world.constant_velocity_enabled = True
                        world.hud.notification(
                            "Enabled Constant Velocity Mode at 60 km/h")
                elif event.key > K_0 and event.key <= K_9:
                    world.camera_manager.set_sensor(event.key - 1 - K_0)
                elif event.key == K_r and not (pygame.key.get_mods() & KMOD_CTRL):
                    world.camera_manager.toggle_recording()
                elif event.key == K_r and (pygame.key.get_mods() & KMOD_CTRL):
                    if (world.recording_enabled):
                        client.stop_recorder()
                        world.recording_enabled = False
                        world.hud.notification("Recorder is OFF")
                    else:
                        # client.start_recorder("manual_recording.rec")
                        client.start_recorder(
                            "C:\\Users\\anony\\Desktop\\Recordings\\manual_recording.log")
                        world.recording_enabled = True
                        world.hud.notification("Recorder is ON")
                elif event.key == K_p and (pygame.key.get_mods() & KMOD_CTRL):
                    # stop recorder
                    client.stop_recorder()
                    world.recording_enabled = False
                    # work around to fix camera at start of replaying
                    current_index = world.camera_manager.index
                    world.destroy_sensors()
                    # disable autopilot
                    self._autopilot_enabled = False
                    world.player.set_autopilot(self._autopilot_enabled)
                    world.hud.notification(
                        "Replaying file 'manual_recording.log'")
                    # replayer
                    client.replay_file("C:\\Users\\anony\\Desktop\\Recordings\\manual_recording.log",
                                       world.recording_start, 0, 0)
                    world.camera_manager.set_sensor(current_index)
                elif event.key == K_k and (pygame.key.get_mods() & KMOD_CTRL):
                    print("k pressed")
                    world.player.enable_carsim(
                        "d:/CVC/carsim/DataUE4/ue4simfile.sim")
                elif event.key == K_j and (pygame.key.get_mods() & KMOD_CTRL):
                    self._carsim_road = not self._carsim_road
                    world.player.use_carsim_road(self._carsim_road)
                    print("j pressed, using carsim road =", self._carsim_road)
                # elif event.key == K_i and (pygame.key.get_mods() & KMOD_CTRL):
                #     print("i pressed")
                #     imp = carla.Location(z=50000)
                #     world.player.add_impulse(imp)
                elif event.key == K_MINUS and (pygame.key.get_mods() & KMOD_CTRL):
                    if pygame.key.get_mods() & KMOD_SHIFT:
                        world.recording_start -= 10
                    else:
                        world.recording_start -= 1
                    world.hud.notification(
                        "Recording start time is %d" % (world.recording_start))
                elif event.key == K_EQUALS and (pygame.key.get_mods() & KMOD_CTRL):
                    if pygame.key.get_mods() & KMOD_SHIFT:
                        world.recording_start += 10
                    else:
                        world.recording_start += 1
                    world.hud.notification(
                        "Recording start time is %d" % (world.recording_start))
                if isinstance(self._control, carla.VehicleControl):
                    if event.key == K_q:
                        self._control.gear = 1 if self._control.reverse else -1
                    elif event.key == K_m:
                        self._control.manual_gear_shift = not self._control.manual_gear_shift
                        self._control.gear = world.player.get_control().gear
                        world.hud.notification('%s Transmission' %
                                               ('Manual' if self._control.manual_gear_shift else 'Automatic'))
                    elif self._control.manual_gear_shift and event.key == K_COMMA:
                        self._control.gear = max(-1, self._control.gear - 1)
                    elif self._control.manual_gear_shift and event.key == K_PERIOD:
                        self._control.gear = self._control.gear + 1
                    elif event.key == K_p and not pygame.key.get_mods() & KMOD_CTRL:
                        self._autopilot_enabled = not self._autopilot_enabled
                        world.player.set_autopilot(self._autopilot_enabled)
                        world.hud.notification(
                            'Autopilot %s' % ('On' if self._autopilot_enabled else 'Off'))
                    elif event.key == K_l and pygame.key.get_mods() & KMOD_CTRL:
                        current_lights ^= carla.VehicleLightState.Special1
                    elif event.key == K_l and pygame.key.get_mods() & KMOD_SHIFT:
                        current_lights ^= carla.VehicleLightState.HighBeam
                    elif event.key == K_l:
                        # Use 'L' key to switch between lights:
                        # closed -> position -> low beam -> fog
                        if not self._lights & carla.VehicleLightState.Position:
                            world.hud.notification("Position lights")
                            current_lights |= carla.VehicleLightState.Position
                        else:
                            world.hud.notification("Low beam lights")
                            current_lights |= carla.VehicleLightState.LowBeam
                        if self._lights & carla.VehicleLightState.LowBeam:
                            world.hud.notification("Fog lights")
                            current_lights |= carla.VehicleLightState.Fog
                        if self._lights & carla.VehicleLightState.Fog:
                            world.hud.notification("Lights off")
                            current_lights ^= carla.VehicleLightState.Position
                            current_lights ^= carla.VehicleLightState.LowBeam
                            current_lights ^= carla.VehicleLightState.Fog
                    elif event.key == K_i:
                        current_lights ^= carla.VehicleLightState.Interior
                    elif event.key == K_z:
                        current_lights ^= carla.VehicleLightState.LeftBlinker
                    elif event.key == K_x:
                        current_lights ^= carla.VehicleLightState.RightBlinker

        if not self._autopilot_enabled:
            if isinstance(self._control, carla.VehicleControl):
                self._parse_vehicle_keys(
                    pygame.key.get_pressed(), clock.get_time())
                self._control.reverse = self._control.gear < 0
                # Set automatic control-related vehicle lights
                if self._control.brake:
                    current_lights |= carla.VehicleLightState.Brake
                else:  # Remove the Brake flag
                    current_lights &= ~carla.VehicleLightState.Brake
                if self._control.reverse:
                    current_lights |= carla.VehicleLightState.Reverse
                else:  # Remove the Reverse flag
                    current_lights &= ~carla.VehicleLightState.Reverse
                if current_lights != self._lights:  # Change the light state only if necessary
                    self._lights = current_lights
                    world.player.set_light_state(
                        carla.VehicleLightState(self._lights))
            elif isinstance(self._control, carla.WalkerControl):
                self._parse_walker_keys(
                    pygame.key.get_pressed(), clock.get_time(), world)
            world.player.apply_control(self._control)

    def _parse_vehicle_keys(self, keys, milliseconds):

        global locked

        if keys[K_UP] or keys[K_w]:
            self._control.throttle = min(self._control.throttle + 0.01, 1)
        else:
            self._control.throttle = 0.0

        if keys[K_DOWN] or keys[K_s]:
            self._control.brake = min(self._control.brake + 0.2, 1)
        else:
            self._control.brake = 0

        steer_increment = 5e-4 * milliseconds
        if (keys[K_LEFT] or keys[K_a]) and (not locked):
            if self._steer_cache > 0:
                self._steer_cache = 0
            else:
                self._steer_cache -= steer_increment
        elif (keys[K_RIGHT] or keys[K_d]) and (not locked):
            if self._steer_cache < 0:
                self._steer_cache = 0
            else:
                self._steer_cache += steer_increment
        else:
            self._steer_cache = 0.0
        self._steer_cache = min(0.7, max(-0.7, self._steer_cache))
        self._control.steer = round(self._steer_cache, 1)
        self._control.hand_brake = keys[K_SPACE]

    def _parse_walker_keys(self, keys, milliseconds, world):
        self._control.speed = 0.0
        if keys[K_DOWN] or keys[K_s]:
            self._control.speed = 0.0
        if keys[K_LEFT] or keys[K_a]:
            self._control.speed = .01
            self._rotation.yaw -= 0.08 * milliseconds
        if keys[K_RIGHT] or keys[K_d]:
            self._control.speed = .01
            self._rotation.yaw += 0.08 * milliseconds
        if keys[K_UP] or keys[K_w]:
            self._control.speed = world.player_max_speed_fast if pygame.key.get_mods(
            ) & KMOD_SHIFT else world.player_max_speed
        self._control.jump = keys[K_SPACE]
        self._rotation.yaw = round(self._rotation.yaw, 1)
        self._control.direction = self._rotation.get_forward_vector()

    @staticmethod
    def _is_quit_shortcut(key):
        return (key == K_ESCAPE) or (key == K_q and pygame.key.get_mods() & KMOD_CTRL)


# ==============================================================================
# -- HUD -----------------------------------------------------------------------
# ==============================================================================


class HUD(object):
    def __init__(self, width, height):
        self.dim = (width, height)
        font = pygame.font.Font(pygame.font.get_default_font(), 20)
        font_name = 'courier' if os.name == 'nt' else 'mono'
        fonts = [x for x in pygame.font.get_fonts() if font_name in x]
        default_font = 'ubuntumono'
        mono = default_font if default_font in fonts else fonts[0]
        mono = pygame.font.match_font(mono)
        self._font_mono = pygame.font.Font(mono, 12 if os.name == 'nt' else 14)
        self._notifications = FadingText(font, (width, 40), (0, height - 40))
        self.help = HelpText(pygame.font.Font(mono, 16), width, height)
        self.server_fps = 0
        self.frame = 0
        self.simulation_time = 0
        self._show_info = True
        self._info_text = []
        self._server_clock = pygame.time.Clock()

    def on_world_tick(self, timestamp):
        self._server_clock.tick()
        self.server_fps = self._server_clock.get_fps()
        self.frame = timestamp.frame
        self.simulation_time = timestamp.elapsed_seconds

    def tick(self, world, clock):
        self._notifications.tick(world, clock)
        if not self._show_info:
            return
        t = world.player.get_transform()
        v = world.player.get_velocity()
        c = world.player.get_control()
        compass = world.imu_sensor.compass
        heading = 'N' if compass > 270.5 or compass < 89.5 else ''
        heading += 'S' if 90.5 < compass < 269.5 else ''
        heading += 'E' if 0.5 < compass < 179.5 else ''
        heading += 'W' if 180.5 < compass < 359.5 else ''
        colhist = world.collision_sensor.get_collision_history()
        collision = [colhist[x + self.frame - 200] for x in range(0, 200)]
        max_col = max(1.0, max(collision))
        collision = [x / max_col for x in collision]
        vehicles = world.world.get_actors().filter('vehicle.*')
        self._info_text = [
            'Server:  % 16.0f FPS' % self.server_fps,
            'Client:  % 16.0f FPS' % clock.get_fps(),
            '',
            'Vehicle: % 20s' % get_actor_display_name(
                world.player, truncate=20),
            'Map:     % 20s' % world.map.name.split('/')[-1],
            'Simulation time: % 12s' % datetime.timedelta(
                seconds=int(self.simulation_time)),
            '',
            'Speed:   % 15.0f km/h' % (3.6 *
                                       math.sqrt(v.x ** 2 + v.y ** 2 + v.z ** 2)),
            u'Compass:% 17.0f\N{DEGREE SIGN} % 2s' % (compass, heading),
            'Accelero: (%5.1f,%5.1f,%5.1f)' % (world.imu_sensor.accelerometer),
            'Gyroscop: (%5.1f,%5.1f,%5.1f)' % (world.imu_sensor.gyroscope),
            'Location:% 20s' % ('(% 5.1f, % 5.1f)' %
                                (t.location.x, t.location.y)),
            'GNSS:% 24s' % ('(% 2.6f, % 3.6f)' %
                            (world.gnss_sensor.lat, world.gnss_sensor.lon)),
            'Height:  % 18.0f m' % t.location.z,
            '']

        if isinstance(c, carla.VehicleControl):
            self._info_text += [
                ('Acceleration:', c.throttle, 0.0, 1.0),
                ('Steer Angle:', c.steer, -1.0, 1.0),
                ('Deceleration:', c.brake, 0.0, 1.0),
                ('Reverse:', c.reverse),
                ('Hand brake:', c.hand_brake),
                ('Manual:', c.manual_gear_shift),
                'Gear:        %s' % {-1: 'R', 0: 'N'}.get(c.gear, c.gear)]
        elif isinstance(c, carla.WalkerControl):
            self._info_text += [
                ('Speed:', c.speed, 0.0, 5.556),
                ('Jump:', c.jump)]
        self._info_text += [
            '',
            'Collision:',
            collision,
            '',
            'Number of vehicles: % 8d' % len(vehicles)]
        if len(vehicles) > 1:
            self._info_text += ['Nearby vehicles:']
            def distance(l): return math.sqrt(
                (l.x - t.location.x) ** 2 + (l.y - t.location.y) ** 2 + (l.z - t.location.z) ** 2)
            vehicles = [(distance(x.get_location()), x)
                        for x in vehicles if x.id != world.player.id]
            for d, vehicle in sorted(vehicles, key=lambda vehicles: vehicles[0]):
                if d > 200.0:
                    break
                vehicle_type = get_actor_display_name(vehicle, truncate=22)
                self._info_text.append('% 4dm %s' % (d, vehicle_type))

        if city_range[0] <= current_pos_x <= city_range[1]:
            self._info_text += ['City Range!']
        else:
            self._info_text += ['Not in City Range!']

    def toggle_info(self):
        self._show_info = not self._show_info

    def notification(self, text, seconds=2.0):
        self._notifications.set_text(text, seconds=seconds)

    def error(self, text):
        self._notifications.set_text('Error: %s' % text, (255, 0, 0))

    def render(self, display):
        if self._show_info:
            info_surface = pygame.Surface((220, self.dim[1]))
            info_surface.set_alpha(100)
            display.blit(info_surface, (0, 0))
            v_offset = 4
            bar_h_offset = 100
            bar_width = 106
            for item in self._info_text:
                if v_offset + 18 > self.dim[1]:
                    break
                if isinstance(item, list):
                    if len(item) > 1:
                        points = [(x + 8, v_offset + 8 + (1.0 - y) * 30)
                                  for x, y in enumerate(item)]
                        pygame.draw.lines(
                            display, (255, 136, 0), False, points, 2)
                    item = None
                    v_offset += 18
                elif isinstance(item, tuple):
                    if isinstance(item[1], bool):
                        rect = pygame.Rect(
                            (bar_h_offset, v_offset + 8), (6, 6))
                        pygame.draw.rect(display, (255, 255, 255),
                                         rect, 0 if item[1] else 1)
                    else:
                        rect_border = pygame.Rect(
                            (bar_h_offset, v_offset + 8), (bar_width, 6))
                        pygame.draw.rect(
                            display, (255, 255, 255), rect_border, 1)
                        f = (item[1] - item[2]) / (item[3] - item[2])
                        if item[2] < 0.0:
                            rect = pygame.Rect(
                                (bar_h_offset + f * (bar_width - 6), v_offset + 8), (6, 6))
                        else:
                            rect = pygame.Rect(
                                (bar_h_offset, v_offset + 8), (f * bar_width, 6))
                        pygame.draw.rect(display, (255, 255, 255), rect)
                    item = item[0]
                if item:  # At this point has to be a str.
                    surface = self._font_mono.render(
                        item, True, (255, 255, 255))
                    display.blit(surface, (8, v_offset))
                v_offset += 18

            graph_width = 300
            graph_height = 200
            graph_spacing = 20

            graph_surface = pygame.Surface(
                (graph_width, graph_height), pygame.SRCALPHA)
            graph_surface.fill((0, 0, 0, 0))  # Make it transparent
            self.render_graph(graph_surface, graph_width,
                              graph_height, "speed")
            display.blit(
                graph_surface, (self.dim[0] - graph_width - graph_spacing, graph_spacing))

            # Render the second graph on the HUD
            graph2_surface = pygame.Surface(
                (graph_width, graph_height), pygame.SRCALPHA)
            graph2_surface.fill((0, 0, 0, 0))  # Make it transparent
            self.render_graph(graph2_surface, graph_width,
                              graph_height, "transmission")
            display.blit(graph2_surface, (self.dim[0] - graph_width -
                         graph_spacing, graph_spacing + graph_height + graph_spacing))

        self._notifications.render(display)
        self.help.render(display)

    def render_graph(self, surface, graph_width, graph_height, type):

        figure_width, figure_height = surface.get_size()

        x_values = []
        y_values = []

        if type == "speed":
            x_values = t
            y_values = speeds
        elif type == "transmission":
            x_values = t
            y_values = transmissions

        # Plot the graph
        fig, ax = plt.subplots(figsize=(figure_width / 80, figure_height / 80))
        ax.plot(x_values, y_values)

        if type == "speed":
            ax.set_xlabel('Time (Ticks)')
            ax.set_ylabel('Speed (m/s)')
            ax.set_title('Speed vs Time Graph')
            ax.set_yticks([0, 1, 5])
        elif type == "transmission":
            ax.set_xlabel('Time (Ticks)')
            ax.set_ylabel('Transmission')
            ax.set_title('Transmission vs Time Graph')
            ax.set_yticks([0, 1, 5])
        canvas = FigureCanvas(fig)
        canvas.draw()

        # Convert the rendered graph to a Pygame surface
        renderer = canvas.get_renderer()
        raw_data = renderer.tostring_rgb()
        size = canvas.get_width_height()
        graph_surface = pygame.image.fromstring(raw_data, size, "RGB")

        graph_surface = pygame.transform.scale(
            graph_surface, (graph_width, graph_height))
        surface.blit(graph_surface, (0, 0))

# ==============================================================================
# -- FadingText ----------------------------------------------------------------
# ==============================================================================


class FadingText(object):
    def __init__(self, font, dim, pos):
        self.font = font
        self.dim = dim
        self.pos = pos
        self.seconds_left = 0
        self.surface = pygame.Surface(self.dim)

    def set_text(self, text, color=(255, 255, 255), seconds=2.0):
        text_texture = self.font.render(text, True, color)
        self.surface = pygame.Surface(self.dim)
        self.seconds_left = seconds
        self.surface.fill((0, 0, 0, 0))
        self.surface.blit(text_texture, (10, 11))

    def tick(self, _, clock):
        delta_seconds = 1e-3 * clock.get_time()
        self.seconds_left = max(0.0, self.seconds_left - delta_seconds)
        self.surface.set_alpha(500.0 * self.seconds_left)

    def render(self, display):
        display.blit(self.surface, self.pos)


# ==============================================================================
# -- HelpText ------------------------------------------------------------------
# ==============================================================================


class HelpText(object):
    """Helper class to handle text output using pygame"""

    def __init__(self, font, width, height):
        lines = __doc__.split('\n')
        self.font = font
        self.line_space = 18
        self.dim = (780, len(lines) * self.line_space + 12)
        self.pos = (0.5 * width - 0.5 *
                    self.dim[0], 0.5 * height - 0.5 * self.dim[1])
        self.seconds_left = 0
        self.surface = pygame.Surface(self.dim)
        self.surface.fill((0, 0, 0, 0))
        for n, line in enumerate(lines):
            text_texture = self.font.render(line, True, (255, 255, 255))
            self.surface.blit(text_texture, (22, n * self.line_space))
            self._render = False
        self.surface.set_alpha(220)

    def toggle(self):
        self._render = not self._render

    def render(self, display):
        if self._render:
            display.blit(self.surface, self.pos)


# ==============================================================================
# -- CollisionSensor -----------------------------------------------------------
# ==============================================================================


class CollisionSensor(object):
    def __init__(self, parent_actor, hud):
        self.sensor = None
        self.history = []
        self._parent = parent_actor
        self.hud = hud
        world = self._parent.get_world()
        bp = world.get_blueprint_library().find('sensor.other.collision')
        self.sensor = world.spawn_actor(
            bp, carla.Transform(), attach_to=self._parent)
        # We need to pass the lambda a weak reference to self to avoid circular
        # reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(
            lambda event: CollisionSensor._on_collision(weak_self, event))

    def get_collision_history(self):
        history = collections.defaultdict(int)
        for frame, intensity in self.history:
            history[frame] += intensity
        return history

    @staticmethod
    def _on_collision(weak_self, event):
        self = weak_self()
        if not self:
            return
        actor_type = get_actor_display_name(event.other_actor)
        self.hud.notification('Collision with %r' % actor_type)
        impulse = event.normal_impulse
        intensity = math.sqrt(impulse.x ** 2 + impulse.y ** 2 + impulse.z ** 2)
        self.history.append((event.frame, intensity))
        if len(self.history) > 4000:
            self.history.pop(0)


# ==============================================================================
# -- LaneInvasionSensor --------------------------------------------------------
# ==============================================================================


class LaneInvasionSensor(object):
    def __init__(self, parent_actor, hud):
        self.sensor = None
        self._parent = parent_actor
        self.hud = hud
        world = self._parent.get_world()
        bp = world.get_blueprint_library().find('sensor.other.lane_invasion')
        self.sensor = world.spawn_actor(
            bp, carla.Transform(), attach_to=self._parent)
        # We need to pass the lambda a weak reference to self to avoid circular
        # reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(
            lambda event: LaneInvasionSensor._on_invasion(weak_self, event))

    @staticmethod
    def _on_invasion(weak_self, event):
        self = weak_self()
        if not self:
            return
        lane_types = set(x.type for x in event.crossed_lane_markings)
        text = ['%r' % str(x).split()[-1] for x in lane_types]
        self.hud.notification('Crossed line %s' % ' and '.join(text))


# ==============================================================================
# -- GnssSensor ----------------------------------------------------------------
# ==============================================================================


class GnssSensor(object):
    def __init__(self, parent_actor):
        self.sensor = None
        self._parent = parent_actor
        self.lat = 0.0
        self.lon = 0.0
        world = self._parent.get_world()
        bp = world.get_blueprint_library().find('sensor.other.gnss')
        self.sensor = world.spawn_actor(bp, carla.Transform(
            carla.Location(x=1.0, z=2.8)), attach_to=self._parent)
        # We need to pass the lambda a weak reference to self to avoid circular
        # reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(
            lambda event: GnssSensor._on_gnss_event(weak_self, event))

    @staticmethod
    def _on_gnss_event(weak_self, event):
        self = weak_self()
        if not self:
            return
        self.lat = event.latitude
        self.lon = event.longitude


# ==============================================================================
# -- IMUSensor -----------------------------------------------------------------
# ==============================================================================


class IMUSensor(object):
    def __init__(self, parent_actor):
        self.sensor = None
        self._parent = parent_actor
        self.accelerometer = (0.0, 0.0, 0.0)
        self.gyroscope = (0.0, 0.0, 0.0)
        self.compass = 0.0
        world = self._parent.get_world()
        bp = world.get_blueprint_library().find('sensor.other.imu')
        self.sensor = world.spawn_actor(
            bp, carla.Transform(), attach_to=self._parent)
        # We need to pass the lambda a weak reference to self to avoid circular
        # reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(
            lambda sensor_data: IMUSensor._IMU_callback(weak_self, sensor_data))

    @staticmethod
    def _IMU_callback(weak_self, sensor_data):
        self = weak_self()
        if not self:
            return
        limits = (-99.9, 99.9)
        self.accelerometer = (
            max(limits[0], min(limits[1], sensor_data.accelerometer.x)),
            max(limits[0], min(limits[1], sensor_data.accelerometer.y)),
            max(limits[0], min(limits[1], sensor_data.accelerometer.z)))
        self.gyroscope = (
            max(limits[0], min(limits[1], math.degrees(
                sensor_data.gyroscope.x))),
            max(limits[0], min(limits[1], math.degrees(
                sensor_data.gyroscope.y))),
            max(limits[0], min(limits[1], math.degrees(sensor_data.gyroscope.z))))
        self.compass = math.degrees(sensor_data.compass)


# ==============================================================================
# -- RadarSensor ---------------------------------------------------------------
# ==============================================================================


class RadarSensor(object):
    def __init__(self, parent_actor):
        self.sensor = None
        self._parent = parent_actor
        self.velocity_range = 7.5  # m/s
        world = self._parent.get_world()
        self.debug = world.debug
        bp = world.get_blueprint_library().find('sensor.other.radar')
        bp.set_attribute('horizontal_fov', str(35))
        bp.set_attribute('vertical_fov', str(20))
        self.sensor = world.spawn_actor(
            bp,
            carla.Transform(
                carla.Location(x=2.8, z=1.0),
                carla.Rotation(pitch=5)),
            attach_to=self._parent)
        # We need a weak reference to self to avoid circular reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(
            lambda radar_data: RadarSensor._Radar_callback(weak_self, radar_data))

    @staticmethod
    def _Radar_callback(weak_self, radar_data):
        self = weak_self()
        if not self:
            return
        # To get a numpy [[vel, altitude, azimuth, depth],...[,,,]]:
        # points = np.frombuffer(radar_data.raw_data, dtype=np.dtype('f4'))
        # points = np.reshape(points, (len(radar_data), 4))

        current_rot = radar_data.transform.rotation
        for detect in radar_data:
            azi = math.degrees(detect.azimuth)
            alt = math.degrees(detect.altitude)
            # The 0.25 adjusts a bit the distance so the dots can
            # be properly seen
            fw_vec = carla.Vector3D(x=detect.depth - 0.25)
            carla.Transform(
                carla.Location(),
                carla.Rotation(
                    pitch=current_rot.pitch + alt,
                    yaw=current_rot.yaw + azi,
                    roll=current_rot.roll)).transform(fw_vec)

            def clamp(min_v, max_v, value):
                return max(min_v, min(value, max_v))

            norm_velocity = detect.velocity / \
                self.velocity_range  # range [-1, 1]
            r = int(clamp(0.0, 1.0, 1.0 - norm_velocity) * 255.0)
            g = int(clamp(0.0, 1.0, 1.0 - abs(norm_velocity)) * 255.0)
            b = int(abs(clamp(- 1.0, 0.0, - 1.0 - norm_velocity)) * 255.0)
            self.debug.draw_point(
                radar_data.transform.location + fw_vec,
                size=0.075,
                life_time=0.06,
                persistent_lines=False,
                color=carla.Color(r, g, b))


# ==============================================================================
# -- CameraManager -------------------------------------------------------------
# ==============================================================================


class CameraManager(object):
    def __init__(self, parent_actor, hud, gamma_correction):
        self.sensor = None
        self.surface = None
        self._parent = parent_actor
        self.hud = hud
        self.recording = False
        bound_y = 0.5 + self._parent.bounding_box.extent.y
        Attachment = carla.AttachmentType
        self._camera_transforms = [
            (carla.Transform(carla.Location(x=-5.5, z=2.5),
             carla.Rotation(pitch=8.0)), Attachment.SpringArmGhost),
            (carla.Transform(carla.Location(x=1.6, z=1.7)), Attachment.Rigid),
            (carla.Transform(carla.Location(x=5.5, y=1.5, z=1.5)),
             Attachment.SpringArmGhost),
            (carla.Transform(carla.Location(x=-8.0, z=6.0),
             carla.Rotation(pitch=6.0)), Attachment.SpringArmGhost),
            (carla.Transform(carla.Location(x=-1, y=-bound_y, z=0.5)), Attachment.Rigid)]
        self.transform_index = 1
        self.sensors = [
            ['sensor.camera.rgb', cc.Raw, 'Camera RGB', {}],
            ['sensor.camera.depth', cc.Raw, 'Camera Depth (Raw)', {}],
            ['sensor.camera.depth', cc.Depth, 'Camera Depth (Gray Scale)', {}],
            ['sensor.camera.depth', cc.LogarithmicDepth,
                'Camera Depth (Logarithmic Gray Scale)', {}],
            ['sensor.camera.semantic_segmentation', cc.Raw,
                'Camera Semantic Segmentation (Raw)', {}],
            ['sensor.camera.semantic_segmentation', cc.CityScapesPalette,
             'Camera Semantic Segmentation (CityScapes Palette)', {}],
            ['sensor.lidar.ray_cast', None,
                'Lidar (Ray-Cast)', {'range': '50'}],
            ['sensor.camera.dvs', cc.Raw, 'Dynamic Vision Sensor', {}],
            ['sensor.camera.rgb', cc.Raw, 'Camera RGB Distorted',
             {'lens_circle_multiplier': '3.0',
              'lens_circle_falloff': '3.0',
              'chromatic_aberration_intensity': '0.5',
              'chromatic_aberration_offset': '0'}]]
        world = self._parent.get_world()
        bp_library = world.get_blueprint_library()
        for item in self.sensors:
            bp = bp_library.find(item[0])
            if item[0].startswith('sensor.camera'):
                bp.set_attribute('image_size_x', str(hud.dim[0]))
                bp.set_attribute('image_size_y', str(hud.dim[1]))
                if bp.has_attribute('gamma'):
                    bp.set_attribute('gamma', str(gamma_correction))
                for attr_name, attr_value in item[3].items():
                    bp.set_attribute(attr_name, attr_value)
            elif item[0].startswith('sensor.lidar'):
                self.lidar_range = 50

                for attr_name, attr_value in item[3].items():
                    bp.set_attribute(attr_name, attr_value)
                    if attr_name == 'range':
                        self.lidar_range = float(attr_value)

            item.append(bp)
        self.index = None

    def toggle_camera(self):
        self.transform_index = (self.transform_index +
                                1) % len(self._camera_transforms)
        self.set_sensor(self.index, notify=False, force_respawn=True)

    def set_sensor(self, index, notify=True, force_respawn=False):
        index = index % len(self.sensors)
        needs_respawn = True if self.index is None else \
            (force_respawn or (self.sensors[index]
             [2] != self.sensors[self.index][2]))
        if needs_respawn:
            if self.sensor is not None:
                self.sensor.destroy()
                self.surface = None
            self.sensor = self._parent.get_world().spawn_actor(
                self.sensors[index][-1],
                self._camera_transforms[self.transform_index][0],
                attach_to=self._parent,
                attachment_type=self._camera_transforms[self.transform_index][1])
            # We need to pass the lambda a weak reference to self to avoid
            # circular reference.
            weak_self = weakref.ref(self)
            self.sensor.listen(
                lambda image: CameraManager._parse_image(weak_self, image))
        if notify:
            self.hud.notification(self.sensors[index][2])
        self.index = index

    def next_sensor(self):
        self.set_sensor(self.index + 1)

    def toggle_recording(self):
        self.recording = not self.recording
        self.hud.notification('Recording %s' %
                              ('On' if self.recording else 'Off'))

    def render(self, display):
        if self.surface is not None:
            display.blit(self.surface, (0, 0))

    @staticmethod
    def _parse_image(weak_self, image):
        self = weak_self()
        if not self:
            return
        if self.sensors[self.index][0].startswith('sensor.lidar'):
            points = np.frombuffer(image.raw_data, dtype=np.dtype('f4'))
            points = np.reshape(points, (int(points.shape[0] / 4), 4))
            lidar_data = np.array(points[:, :2])
            lidar_data *= min(self.hud.dim) / (2.0 * self.lidar_range)
            lidar_data += (0.5 * self.hud.dim[0], 0.5 * self.hud.dim[1])
            lidar_data = np.fabs(lidar_data)  # pylint: disable=E1111
            lidar_data = lidar_data.astype(np.int32)
            lidar_data = np.reshape(lidar_data, (-1, 2))
            lidar_img_size = (self.hud.dim[0], self.hud.dim[1], 3)
            lidar_img = np.zeros((lidar_img_size), dtype=np.uint8)
            lidar_img[tuple(lidar_data.T)] = (255, 255, 255)
            self.surface = pygame.surfarray.make_surface(lidar_img)
        elif self.sensors[self.index][0].startswith('sensor.camera.dvs'):
            # Example of converting the raw_data from a carla.DVSEventArray
            # sensor into a NumPy array and using it as an image
            dvs_events = np.frombuffer(image.raw_data, dtype=np.dtype([
                ('x', np.uint16), ('y', np.uint16), ('t', np.int64), ('pol', np.bool)]))
            dvs_img = np.zeros((image.height, image.width, 3), dtype=np.uint8)
            # Blue is positive, red is negative
            dvs_img[dvs_events[:]['y'], dvs_events[:]
                    ['x'], dvs_events[:]['pol'] * 2] = 255
            self.surface = pygame.surfarray.make_surface(
                dvs_img.swapaxes(0, 1))
        else:
            image.convert(self.sensors[self.index][1])
            array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
            array = np.reshape(array, (image.height, image.width, 4))
            array = array[:, :, :3]
            array = array[:, :, ::-1]
            self.surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
        if self.recording:
            image.save_to_disk('_out/%08d' % image.frame)


# ==============================================================================
# -- game_loop() ---------------------------------------------------------------
# ==============================================================================

def game_loop(args):
    window_size = (1440, 810)

    # Create a Pygame display surface with the desired size
    pygame.init()

    world = None

    try:
        client = carla.Client(args.host, args.port)
        client.set_timeout(20.0)

        global global_client
        global_client = client

        display = pygame.display.set_mode(
            (args.width, args.height),
            pygame.HWSURFACE | pygame.DOUBLEBUF)

        world = client.load_world('Town03')

        hud = HUD(args.width, args.height)

        world = World(client.get_world(), hud, args)

        controller = KeyboardControl(world, args.autopilot)
        clock = pygame.time.Clock()

        while True:
            clock.tick_busy_loop(60)
            if controller.parse_events(client, world, clock):
                return

            world.tick(clock)
            world.render(display)
            pygame.display.flip()

    finally:

        if world and world.recording_enabled:
            client.stop_recorder()

        if world is not None:
            world.destroy()

        pygame.quit()


# ==============================================================================
# -- main() --------------------------------------------------------------------
# ==============================================================================


def main():
    argparser = argparse.ArgumentParser(
        description='CARLA Manual Control Client')
    argparser.add_argument(
        '-v', '--verbose',
        action='store_true',
        dest='debug',
        help='print debug information')
    argparser.add_argument(
        '--host',
        metavar='H',
        default='127.0.0.1',
        help='IP of the host server (default: 127.0.0.1)')
    argparser.add_argument(
        '-p', '--port',
        metavar='P',
        default=2000,
        type=int,
        help='TCP port to listen to (default: 2000)')
    argparser.add_argument(
        '-a', '--autopilot',
        action='store_true',
        help='enable autopilot')
    argparser.add_argument(
        '--res',
        metavar='WIDTHxHEIGHT',
        default='1280x720',
        help='window resolution (default: 1280x720)')
    argparser.add_argument(
        '--filterw',
        metavar='PATTERN',
        default='walker.pedestrian.*',
        help='Filter pedestrian type (default: "walker.pedestrian.*")')
    argparser.add_argument(
        '--filter',
        metavar='PATTERN',
        default='vehicle.*',
        help='actor filter (default: "vehicle.*")')
    argparser.add_argument(
        '--rolename',
        metavar='NAME',
        default='hero',
        help='actor role name (default: "hero")')
    argparser.add_argument(
        '--generationw',
        metavar='G',
        default='2',
        help='restrict to certain pedestrian generation (values: "1","2","All" - default: "2")')
    argparser.add_argument(
        '--gamma',
        default=2.2,
        type=float,
        help='Gamma correction of the camera (default: 2.2)')
    args = argparser.parse_args()

    args.width, args.height = [int(x) for x in args.res.split('x')]

    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(format='%(levelname)s: %(message)s', level=log_level)

    logging.info('listening to server %s:%s', args.host, args.port)
    print(__doc__)

    try:
        game_loop(args)

    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')


if __name__ == '__main__':
    main()
