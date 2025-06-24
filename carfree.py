#!/usr/bin/env python

# Copyright (c) 2019 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
CarFree: Complete Pipeline for CARLA Simulation

This script combines the functionality of spawn.py.py, weather.py.py, extract.py.py, and generate.py.py
into a single pipeline that:
1. Spawns vehicles and pedestrians
2. Changes weather conditions
3. Extracts data from the simulation
4. Processes the extracted data

Run this script to execute the entire pipeline in one go.
"""

import glob
import os
import sys
import time
import argparse
import logging
import random
import copy
import re
import weakref
import textwrap
import math

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla
from carla import ColorConverter as cc

try:
    import pygame
    from pygame.locals import K_ESCAPE
    from pygame.locals import K_SPACE
    from pygame.locals import KMOD_SHIFT
    from pygame.locals import K_a
    from pygame.locals import K_d
    from pygame.locals import K_s
    from pygame.locals import K_w
    from pygame.locals import K_TAB
    from pygame.locals import K_BACKQUOTE
    from pygame.locals import K_p
    from pygame.locals import K_c
    from pygame.locals import K_l
except ImportError:
    raise RuntimeError('cannot import pygame, make sure pygame package is installed')

try:
    import numpy as np
except ImportError:
    raise RuntimeError('cannot import numpy, make sure numpy package is installed')

try:
    import cv2
except ImportError:
    raise RuntimeError('cannot import cv2, make sure opencv-python package is installed')

# Global variables and constants
VIEW_WIDTH = 1920//2
VIEW_HEIGHT = 1080//2
VIEW_FOV = 90

VBB_COLOR = (0, 0, 255)
WBB_COLOR = (255, 0, 0)

Vehicle_COLOR = (142, 0, 0)
Walker_COLOR = (60, 20, 220)

rgb_info = np.zeros((VIEW_HEIGHT, VIEW_WIDTH, 3), dtype="i")
seg_info = np.zeros((VIEW_HEIGHT, VIEW_WIDTH, 3), dtype="i")
rgb_info_left = np.zeros((VIEW_HEIGHT, VIEW_WIDTH, 3), dtype="i")
seg_info_left = np.zeros((VIEW_HEIGHT, VIEW_WIDTH, 3), dtype="i")
rgb_info_front_left = np.zeros((VIEW_HEIGHT, VIEW_WIDTH, 3), dtype="i")
seg_info_front_left = np.zeros((VIEW_HEIGHT, VIEW_WIDTH, 3), dtype="i")
rgb_info_right = np.zeros((VIEW_HEIGHT, VIEW_WIDTH, 3), dtype="i")
seg_info_right = np.zeros((VIEW_HEIGHT, VIEW_WIDTH, 3), dtype="i")
rgb_info_front_right = np.zeros((VIEW_HEIGHT, VIEW_WIDTH, 3), dtype="i")
seg_info_front_right = np.zeros((VIEW_HEIGHT, VIEW_WIDTH, 3), dtype="i")
area_info = np.zeros(shape=[VIEW_HEIGHT, VIEW_WIDTH, 3], dtype=np.uint8)
index_count = 0

vehicle_bbox_record = False
pedestrian_bbox_record = False
count = 0

# Create directories
dir_my = 'my_data/'
dir_custom = 'custom_data/'
dir_draw = 'draw_bounding_box/'
dir_vehicle_bbox = 'VehicleBBox/'
dir_pedestrian_bbox = 'PedestrianBBox/'
dir_segmentation = 'SegmentationImage/'

if not os.path.exists(dir_my):
    os.makedirs(dir_my)
if not os.path.exists(dir_custom):
    os.makedirs(dir_custom)
if not os.path.exists(dir_draw):
    os.makedirs(dir_draw)
if not os.path.exists(dir_vehicle_bbox):
    os.makedirs(dir_vehicle_bbox)
if not os.path.exists(dir_pedestrian_bbox):
    os.makedirs(dir_pedestrian_bbox)
if not os.path.exists(dir_segmentation):
    os.makedirs(dir_segmentation)

# Helper functions
def clamp(value, minimum=0.0, maximum=100.0):
    return max(minimum, min(value, maximum))

# Classes from weather.py.py
class Sun(object):
    def __init__(self, azimuth, altitude):
        self.azimuth = azimuth
        self.altitude = altitude
        self._t = 0.0

    def tick(self, delta_seconds):
        self._t += 0.008 * delta_seconds
        self._t %= 2.0 * math.pi
        self.azimuth += 0.25 * delta_seconds
        self.azimuth %= 360.0
        self.altitude = 35.0 * (math.sin(self._t) + 1.0) 
        if self.altitude < -80:
            self.altitude = -60 
        if self.altitude > 40:
            self.altitude = 0

    def __str__(self):
        return 'Sun(%.2f, %.2f)' % (self.azimuth, self.altitude)

class Storm(object):
    def __init__(self, precipitation):
        self._t = precipitation if precipitation > 0.0 else -50.0
        self._increasing = True
        self.clouds = 0.0
        self.rain = 0.0
        self.puddles = 0.0
        self.wind = 0.0

    def tick(self, delta_seconds):
        delta = (1.3 if self._increasing else -1.3) * delta_seconds
        self._t = clamp(delta + self._t, -250.0, 100.0)
        self.clouds = clamp(self._t + 40.0, 0.0, 90.0) + 70
        self.rain = clamp(self._t, 30.0, 100.0) + 50
        delay = -10.0 if self._increasing else 90.0
        self.puddles = clamp(self._t + delay, 0.0, 100.0)
        self.wind = clamp(self._t - delay, 0.0, 100.0)
        if self._t == -250.0:
            self._increasing = True
        if self._t == 100.0:
            self._increasing = False

    def __str__(self):
        return 'Storm(clouds=%d%%, rain=%d%%, wind=%d%%)' % (self.clouds, self.rain, self.wind)

class Weather(object):
    def __init__(self, weather):
        self.weather = weather
        self._sun = Sun(weather.sun_azimuth_angle, weather.sun_altitude_angle)
        self._storm = Storm(weather.precipitation)

    def tick(self, delta_seconds):
        self._sun.tick(delta_seconds)
        self._storm.tick(delta_seconds)
        self.weather.cloudyness = self._storm.clouds
        self.weather.precipitation = self._storm.rain
        self.weather.precipitation_deposits = self._storm.puddles
        self.weather.wind_intensity = self._storm.wind
        self.weather.sun_azimuth_angle = self._sun.azimuth
        self.weather.sun_altitude_angle = self._sun.altitude

    def __str__(self):
        return '%s %s' % (self._sun, self._storm)

# Classes from extract.py.py
class PedestrianBoundingBoxes(object):
    @staticmethod
    def get_bounding_boxes(pedestrians, camera):
        """
        Returns 3D bounding boxes for all pedestrians in the supplied list
        """
        bounding_boxes = [PedestrianBoundingBoxes.get_bounding_box(pedestrian, camera) for pedestrian in pedestrians]
        # Filter objects behind camera
        bounding_boxes = [bb for bb in bounding_boxes if all(bb[:, 2] > 0)]
        return bounding_boxes

    @staticmethod
    def draw_bounding_boxes(display, bounding_boxes):
        """
        Draws bounding boxes on pygame display.
        """
        global pedestrian_bbox_record
        global count

        bb_surface = pygame.Surface((VIEW_WIDTH, VIEW_HEIGHT))
        bb_surface.set_colorkey((0, 0, 0))

        # Ensure PedestrianBBox directory exists
        pedestrian_bbox_dir = 'PedestrianBBox'
        if not os.path.exists(pedestrian_bbox_dir):
            os.makedirs(pedestrian_bbox_dir)
            print(f"Created directory: {pedestrian_bbox_dir}")

        if pedestrian_bbox_record:
            f = open(os.path.join(pedestrian_bbox_dir, 'bbox'+ str(count)), 'w')

        for bbox in bounding_boxes:
            points = [(int(bbox[i, 0]), int(bbox[i, 1])) for i in range(8)]
            # draw lines
            # base
            pygame.draw.line(bb_surface, WBB_COLOR, points[0], points[1])
            pygame.draw.line(bb_surface, WBB_COLOR, points[1], points[2])
            pygame.draw.line(bb_surface, WBB_COLOR, points[2], points[3])
            pygame.draw.line(bb_surface, WBB_COLOR, points[3], points[0])
            # top
            pygame.draw.line(bb_surface, WBB_COLOR, points[4], points[5])
            pygame.draw.line(bb_surface, WBB_COLOR, points[5], points[6])
            pygame.draw.line(bb_surface, WBB_COLOR, points[6], points[7])
            pygame.draw.line(bb_surface, WBB_COLOR, points[7], points[4])
            # base-top
            pygame.draw.line(bb_surface, WBB_COLOR, points[0], points[4])
            pygame.draw.line(bb_surface, WBB_COLOR, points[1], points[5])
            pygame.draw.line(bb_surface, WBB_COLOR, points[2], points[6])
            pygame.draw.line(bb_surface, WBB_COLOR, points[3], points[7])

            if pedestrian_bbox_record:
                for i in range(8):
                    f.write(str(points[i][0]) + ',' + str(points[i][1]) + ' ')
                f.write('\n')

        if pedestrian_bbox_record:
            f.close()
            # Don't set pedestrian_bbox_record to False to allow continuous recording

        display.blit(bb_surface, (0, 0))

    @staticmethod
    def get_bounding_box(pedestrian, camera):
        """
        Returns 3D bounding box for a pedestrian based on camera view.
        """
        bb_cords = PedestrianBoundingBoxes._create_bb_points(pedestrian)
        cords_x_y_z = PedestrianBoundingBoxes._pedestrian_to_sensor(bb_cords, pedestrian, camera)[:3, :]
        cords_y_minus_z_x = np.concatenate([cords_x_y_z[1, :], -cords_x_y_z[2, :], cords_x_y_z[0, :]])
        bbox = np.transpose(np.dot(camera.calibration, cords_y_minus_z_x))
        camera_bbox = np.concatenate([bbox[:, 0] / bbox[:, 2], bbox[:, 1] / bbox[:, 2], bbox[:, 2]], axis=1)
        return camera_bbox

    @staticmethod
    def _create_bb_points(pedestrian):
        """
        Returns 3D bounding box for a pedestrian.
        """
        cords = np.zeros((8, 4))
        extent = pedestrian.bounding_box.extent
        cords[0, :] = np.array([extent.x, extent.y, -extent.z, 1])
        cords[1, :] = np.array([-extent.x, extent.y, -extent.z, 1])
        cords[2, :] = np.array([-extent.x, -extent.y, -extent.z, 1])
        cords[3, :] = np.array([extent.x, -extent.y, -extent.z, 1])
        cords[4, :] = np.array([extent.x, extent.y, extent.z, 1])
        cords[5, :] = np.array([-extent.x, extent.y, extent.z, 1])
        cords[6, :] = np.array([-extent.x, -extent.y, extent.z, 1])
        cords[7, :] = np.array([extent.x, -extent.y, extent.z, 1])
        return cords

    @staticmethod
    def _pedestrian_to_sensor(cords, pedestrian, sensor):
        """
        Transforms coordinates of a pedestrian bounding box to sensor.
        """
        world_cord = PedestrianBoundingBoxes._pedestrian_to_world(cords, pedestrian)
        sensor_cord = PedestrianBoundingBoxes._world_to_sensor(world_cord, sensor)
        return sensor_cord

    @staticmethod
    def _pedestrian_to_world(cords, pedestrian):
        """
        Transforms coordinates of a pedestrian bounding box to world.
        """
        bb_transform = carla.Transform(pedestrian.bounding_box.location)
        bb_pedestrian_matrix = PedestrianBoundingBoxes.get_matrix(bb_transform)
        pedestrian_world_matrix = PedestrianBoundingBoxes.get_matrix(pedestrian.get_transform())
        bb_world_matrix = np.dot(pedestrian_world_matrix, bb_pedestrian_matrix)
        world_cords = np.dot(bb_world_matrix, np.transpose(cords))
        return world_cords

    @staticmethod
    def _world_to_sensor(cords, sensor):
        """
        Transforms world coordinates to sensor.
        """
        sensor_world_matrix = PedestrianBoundingBoxes.get_matrix(sensor.get_transform())
        world_sensor_matrix = np.linalg.inv(sensor_world_matrix)
        sensor_cords = np.dot(world_sensor_matrix, cords)
        return sensor_cords

    @staticmethod
    def get_matrix(transform):
        """
        Creates matrix from carla transform.
        """
        rotation = transform.rotation
        location = transform.location
        c_y = np.cos(np.radians(rotation.yaw))
        s_y = np.sin(np.radians(rotation.yaw))
        c_r = np.cos(np.radians(rotation.roll))
        s_r = np.sin(np.radians(rotation.roll))
        c_p = np.cos(np.radians(rotation.pitch))
        s_p = np.sin(np.radians(rotation.pitch))
        matrix = np.matrix(np.identity(4))
        matrix[0, 3] = location.x
        matrix[1, 3] = location.y
        matrix[2, 3] = location.z
        matrix[0, 0] = c_p * c_y
        matrix[0, 1] = c_y * s_p * s_r - s_y * c_r
        matrix[0, 2] = -c_y * s_p * c_r - s_y * s_r
        matrix[1, 0] = s_y * c_p
        matrix[1, 1] = s_y * s_p * s_r + c_y * c_r
        matrix[1, 2] = -s_y * s_p * c_r + c_y * s_r
        matrix[2, 0] = s_p
        matrix[2, 1] = -c_p * s_r
        matrix[2, 2] = c_p * c_r
        return matrix

class VehicleBoundingBoxes(object):
    @staticmethod
    def get_bounding_boxes(vehicles, camera):
        """
        Returns 3D bounding boxes for all vehicles in the supplied list
        """
        bounding_boxes = [VehicleBoundingBoxes.get_bounding_box(vehicle, camera) for vehicle in vehicles]
        # Filter objects behind camera
        bounding_boxes = [bb for bb in bounding_boxes if all(bb[:, 2] > 0)]
        return bounding_boxes

    @staticmethod
    def draw_bounding_boxes(display, bounding_boxes):
        """
        Draws bounding boxes on pygame display.
        """
        global vehicle_bbox_record
        global count

        bb_surface = pygame.Surface((VIEW_WIDTH, VIEW_HEIGHT))
        bb_surface.set_colorkey((0, 0, 0))

        # Ensure VehicleBBox directory exists
        vehicle_bbox_dir = 'VehicleBBox'
        if not os.path.exists(vehicle_bbox_dir):
            os.makedirs(vehicle_bbox_dir)
            print(f"Created directory: {vehicle_bbox_dir}")

        if vehicle_bbox_record:
            f = open(os.path.join(vehicle_bbox_dir, 'bbox'+ str(count)), 'w')

        for bbox in bounding_boxes:
            points = [(int(bbox[i, 0]), int(bbox[i, 1])) for i in range(8)]
            # draw lines
            # base
            pygame.draw.line(bb_surface, VBB_COLOR, points[0], points[1])
            pygame.draw.line(bb_surface, VBB_COLOR, points[1], points[2])
            pygame.draw.line(bb_surface, VBB_COLOR, points[2], points[3])
            pygame.draw.line(bb_surface, VBB_COLOR, points[3], points[0])
            # top
            pygame.draw.line(bb_surface, VBB_COLOR, points[4], points[5])
            pygame.draw.line(bb_surface, VBB_COLOR, points[5], points[6])
            pygame.draw.line(bb_surface, VBB_COLOR, points[6], points[7])
            pygame.draw.line(bb_surface, VBB_COLOR, points[7], points[4])
            # base-top
            pygame.draw.line(bb_surface, VBB_COLOR, points[0], points[4])
            pygame.draw.line(bb_surface, VBB_COLOR, points[1], points[5])
            pygame.draw.line(bb_surface, VBB_COLOR, points[2], points[6])
            pygame.draw.line(bb_surface, VBB_COLOR, points[3], points[7])

            if vehicle_bbox_record:
                for i in range(8):
                    f.write(str(points[i][0]) + ',' + str(points[i][1]) + ' ')
                f.write('\n')

        if vehicle_bbox_record:
            f.close()
            # Don't set vehicle_bbox_record to False to allow continuous recording

        display.blit(bb_surface, (0, 0))

    @staticmethod
    def get_bounding_box(vehicle, camera):
        """
        Returns 3D bounding box for a vehicle based on camera view.
        """
        bb_cords = VehicleBoundingBoxes._create_bb_points(vehicle)
        cords_x_y_z = VehicleBoundingBoxes._vehicle_to_sensor(bb_cords, vehicle, camera)[:3, :]
        cords_y_minus_z_x = np.concatenate([cords_x_y_z[1, :], -cords_x_y_z[2, :], cords_x_y_z[0, :]])
        bbox = np.transpose(np.dot(camera.calibration, cords_y_minus_z_x))
        camera_bbox = np.concatenate([bbox[:, 0] / bbox[:, 2], bbox[:, 1] / bbox[:, 2], bbox[:, 2]], axis=1)
        return camera_bbox

    @staticmethod
    def _create_bb_points(vehicle):
        """
        Returns 3D bounding box for a vehicle.
        """
        cords = np.zeros((8, 4))
        extent = vehicle.bounding_box.extent
        cords[0, :] = np.array([extent.x, extent.y, -extent.z, 1])
        cords[1, :] = np.array([-extent.x, extent.y, -extent.z, 1])
        cords[2, :] = np.array([-extent.x, -extent.y, -extent.z, 1])
        cords[3, :] = np.array([extent.x, -extent.y, -extent.z, 1])
        cords[4, :] = np.array([extent.x, extent.y, extent.z, 1])
        cords[5, :] = np.array([-extent.x, extent.y, extent.z, 1])
        cords[6, :] = np.array([-extent.x, -extent.y, extent.z, 1])
        cords[7, :] = np.array([extent.x, -extent.y, extent.z, 1])
        return cords

    @staticmethod
    def _vehicle_to_sensor(cords, vehicle, sensor):
        """
        Transforms coordinates of a vehicle bounding box to sensor.
        """
        world_cord = VehicleBoundingBoxes._vehicle_to_world(cords, vehicle)
        sensor_cord = VehicleBoundingBoxes._world_to_sensor(world_cord, sensor)
        return sensor_cord

    @staticmethod
    def _vehicle_to_world(cords, vehicle):
        """
        Transforms coordinates of a vehicle bounding box to world.
        """
        bb_transform = carla.Transform(vehicle.bounding_box.location)
        bb_vehicle_matrix = VehicleBoundingBoxes.get_matrix(bb_transform)
        vehicle_world_matrix = VehicleBoundingBoxes.get_matrix(vehicle.get_transform())
        bb_world_matrix = np.dot(vehicle_world_matrix, bb_vehicle_matrix)
        world_cords = np.dot(bb_world_matrix, np.transpose(cords))
        return world_cords

    @staticmethod
    def _world_to_sensor(cords, sensor):
        """
        Transforms world coordinates to sensor.
        """
        sensor_world_matrix = VehicleBoundingBoxes.get_matrix(sensor.get_transform())
        world_sensor_matrix = np.linalg.inv(sensor_world_matrix)
        sensor_cords = np.dot(world_sensor_matrix, cords)
        return sensor_cords

    @staticmethod
    def get_matrix(transform):
        """
        Creates matrix from carla transform.
        """
        rotation = transform.rotation
        location = transform.location
        c_y = np.cos(np.radians(rotation.yaw))
        s_y = np.sin(np.radians(rotation.yaw))
        c_r = np.cos(np.radians(rotation.roll))
        s_r = np.sin(np.radians(rotation.roll))
        c_p = np.cos(np.radians(rotation.pitch))
        s_p = np.sin(np.radians(rotation.pitch))
        matrix = np.matrix(np.identity(4))
        matrix[0, 3] = location.x
        matrix[1, 3] = location.y
        matrix[2, 3] = location.z
        matrix[0, 0] = c_p * c_y
        matrix[0, 1] = c_y * s_p * s_r - s_y * c_r
        matrix[0, 2] = -c_y * s_p * c_r - s_y * s_r
        matrix[1, 0] = s_y * c_p
        matrix[1, 1] = s_y * s_p * s_r + c_y * c_r
        matrix[1, 2] = -s_y * s_p * c_r + c_y * s_r
        matrix[2, 0] = s_p
        matrix[2, 1] = -c_p * s_r
        matrix[2, 2] = c_p * c_r
        return matrix

class BasicSynchronousClient(object):
    def __init__(self, client=None, world=None):
        self.client = client
        self.world = world
        self.camera = None
        self.camera_segmentation = None
        self.camera_left = None
        self.camera_front_left = None
        self.camera_right = None
        self.camera_front_right = None
        self.camera_segmentation_left = None
        self.camera_segmentation_front_left = None
        self.camera_segmentation_right = None
        self.camera_segmentation_front_right = None
        self.car = None

        self.display = None
        self.image = None
        self.segmentation = None
        self.image_left = None
        self.image_front_left = None
        self.image_right = None
        self.image_front_right = None
        self.segmentation_left = None
        self.segmentation_front_left = None
        self.segmentation_right = None
        self.segmentation_front_right = None
        self.capture = True
        self.rgb_record = False
        self.seg_record = False
        self.screen_capture = 0
        self.loop_state = True
        self.image_count = 0
        self.time_interval = 0

    def camera_blueprint(self, filter):
        """
        Returns camera blueprint.
        """
        camera_bp = self.world.get_blueprint_library().find(filter)
        camera_bp.set_attribute('image_size_x', str(VIEW_WIDTH))
        camera_bp.set_attribute('image_size_y', str(VIEW_HEIGHT))
        camera_bp.set_attribute('fov', str(VIEW_FOV))
        return camera_bp

    def set_synchronous_mode(self, synchronous_mode):
        """
        Sets synchronous mode.
        """
        settings = self.world.get_settings()
        settings.synchronous_mode = synchronous_mode
        self.world.apply_settings(settings)

    def setup_car(self):
        """
        Spawns actor-vehicle to be used to as camera platform.
        """
        # Try different vehicle models in case the specified one is not found
        vehicle_models = [
            'vehicle.lincoln.mkz2017',
            'vehicle.tesla.model3',
            'vehicle.audi.a2',
            'vehicle.audi.tt',
            'vehicle.bmw.grandtourer',
            'vehicle.chevrolet.impala',
            'vehicle.citroen.c3',
            'vehicle.dodge.charger',
            'vehicle.ford.mustang',
            'vehicle.jeep.wrangler_rubicon',
            'vehicle.mercedes.coupe',
            'vehicle.mini.cooper_s',
            'vehicle.nissan.micra',
            'vehicle.nissan.patrol',
            'vehicle.seat.leon',
            'vehicle.toyota.prius',
            'vehicle.volkswagen.t2'
        ]

        car_bp = None
        for model in vehicle_models:
            try:
                car_bp = self.world.get_blueprint_library().find(model)
                if car_bp is not None:
                    print(f"Using vehicle model: {model}")
                    break
            except:
                continue

        if car_bp is None:
            # If no specific model is found, try to get any vehicle
            try:
                blueprints = self.world.get_blueprint_library().filter('vehicle.*')
                if blueprints:
                    car_bp = random.choice(blueprints)
                    print(f"Using random vehicle model: {car_bp.id}")
            except Exception as e:
                print(f"Error finding vehicle blueprint: {e}")
                raise

        car_bp.set_attribute('role_name', 'hero')
        location = random.choice(self.world.get_map().get_spawn_points())
        self.car = self.world.spawn_actor(car_bp, location)

        # Set the vehicle to autopilot mode
        self.car.set_autopilot(True)
        print("Vehicle set to autonomous driving mode")

    def setup_camera(self):
        """
        Spawns actor-camera to be used to render view and sets up the spectator to follow the vehicle.
        """
        # Set up the spectator to follow the vehicle
        self.spectator = self.world.get_spectator()

        # We still need the cameras for data collection
        camera_transform = carla.Transform(carla.Location(x=-5.5, z=2.8), carla.Rotation(pitch=-15))
        self.camera = self.world.spawn_actor(self.camera_blueprint('sensor.camera.rgb'), camera_transform, attach_to=self.car)
        weak_self = weakref.ref(self)
        self.camera.listen(lambda image: BasicSynchronousClient.set_image(weak_self, image))

        camera_segmentation_transform = carla.Transform(carla.Location(x=-5.5, z=2.8), carla.Rotation(pitch=-15))
        self.camera_segmentation = self.world.spawn_actor(self.camera_blueprint('sensor.camera.semantic_segmentation'), camera_segmentation_transform, attach_to=self.car)
        weak_self = weakref.ref(self)
        self.camera_segmentation.listen(lambda image: BasicSynchronousClient.set_segmentation(weak_self, image))

        # Add new cameras on top of the vehicle
        # Left camera - facing left (90 degrees to the left)
        camera_left_transform = carla.Transform(carla.Location(x=0, y=0, z=2.8), carla.Rotation(pitch=0, yaw=90))
        self.camera_left = self.world.spawn_actor(self.camera_blueprint('sensor.camera.rgb'), camera_left_transform, attach_to=self.car)
        weak_self = weakref.ref(self)
        self.camera_left.listen(lambda image: BasicSynchronousClient.set_image_left(weak_self, image))

        # Left segmentation camera
        self.camera_segmentation_left = self.world.spawn_actor(self.camera_blueprint('sensor.camera.semantic_segmentation'), camera_left_transform, attach_to=self.car)
        weak_self = weakref.ref(self)
        self.camera_segmentation_left.listen(lambda image: BasicSynchronousClient.set_segmentation_left(weak_self, image))

        # Front left camera - facing front left (45 degrees to the left)
        camera_front_left_transform = carla.Transform(carla.Location(x=0, y=0, z=2.8), carla.Rotation(pitch=0, yaw=45))
        self.camera_front_left = self.world.spawn_actor(self.camera_blueprint('sensor.camera.rgb'), camera_front_left_transform, attach_to=self.car)
        weak_self = weakref.ref(self)
        self.camera_front_left.listen(lambda image: BasicSynchronousClient.set_image_front_left(weak_self, image))

        # Front left segmentation camera
        self.camera_segmentation_front_left = self.world.spawn_actor(self.camera_blueprint('sensor.camera.semantic_segmentation'), camera_front_left_transform, attach_to=self.car)
        weak_self = weakref.ref(self)
        self.camera_segmentation_front_left.listen(lambda image: BasicSynchronousClient.set_segmentation_front_left(weak_self, image))

        # Right camera - facing right (90 degrees to the right)
        camera_right_transform = carla.Transform(carla.Location(x=0, y=0, z=2.8), carla.Rotation(pitch=0, yaw=-90))
        self.camera_right = self.world.spawn_actor(self.camera_blueprint('sensor.camera.rgb'), camera_right_transform, attach_to=self.car)
        weak_self = weakref.ref(self)
        self.camera_right.listen(lambda image: BasicSynchronousClient.set_image_right(weak_self, image))

        # Right segmentation camera
        self.camera_segmentation_right = self.world.spawn_actor(self.camera_blueprint('sensor.camera.semantic_segmentation'), camera_right_transform, attach_to=self.car)
        weak_self = weakref.ref(self)
        self.camera_segmentation_right.listen(lambda image: BasicSynchronousClient.set_segmentation_right(weak_self, image))

        # Front right camera - facing front right (45 degrees to the right)
        camera_front_right_transform = carla.Transform(carla.Location(x=0, y=0, z=2.8), carla.Rotation(pitch=0, yaw=-45))
        self.camera_front_right = self.world.spawn_actor(self.camera_blueprint('sensor.camera.rgb'), camera_front_right_transform, attach_to=self.car)
        weak_self = weakref.ref(self)
        self.camera_front_right.listen(lambda image: BasicSynchronousClient.set_image_front_right(weak_self, image))

        # Front right segmentation camera
        self.camera_segmentation_front_right = self.world.spawn_actor(self.camera_blueprint('sensor.camera.semantic_segmentation'), camera_front_right_transform, attach_to=self.car)
        weak_self = weakref.ref(self)
        self.camera_segmentation_front_right.listen(lambda image: BasicSynchronousClient.set_segmentation_front_right(weak_self, image))

        calibration = np.identity(3)
        calibration[0, 2] = VIEW_WIDTH / 2.0
        calibration[1, 2] = VIEW_HEIGHT / 2.0
        calibration[0, 0] = calibration[1, 1] = VIEW_WIDTH / (2.0 * np.tan(VIEW_FOV * np.pi / 360.0))
        self.camera.calibration = calibration
        self.camera_segmentation.calibration = calibration
        self.camera_left.calibration = calibration
        self.camera_front_left.calibration = calibration
        self.camera_right.calibration = calibration
        self.camera_front_right.calibration = calibration
        self.camera_segmentation_left.calibration = calibration
        self.camera_segmentation_front_left.calibration = calibration
        self.camera_segmentation_right.calibration = calibration
        self.camera_segmentation_front_right.calibration = calibration

    def control(self, car):
        """
        Applies control to main car based on pygame pressed keys.
        Will return True If ESCAPE is hit, otherwise False to end this control
        """
        keys = pygame.key.get_pressed()
        if keys[K_ESCAPE]:
            return True

        control = car.get_control()
        control.throttle = 0
        if keys[K_w]:
            control.throttle = 1
            control.reverse = False
        elif keys[K_s]:
            control.throttle = 1
            control.reverse = True
        if keys[K_a]:
            control.steer = max(-1., min(control.steer - 0.05, 0))
        elif keys[K_d]:
            control.steer = min(1., max(control.steer + 0.05, 0))
        else:
            control.steer = 0
        control.hand_brake = keys[K_SPACE]

        if keys[K_TAB]:
            self.loop_state = not self.loop_state
            print("Loop State : %s" %self.loop_state)
            time.sleep(0.5)

        if keys[K_BACKQUOTE]:
            self.screen_capture = 1
            time.sleep(0.1)
        else:
            self.screen_capture = 0

        if keys[K_p]:
            self.rgb_record = True
            time.sleep(0.1)

        if keys[K_c]:
            self.seg_record = True
            time.sleep(0.1)

        car.apply_control(control)
        return False

    @staticmethod
    def set_image(weak_self, img):
        """
        Sets image coming from camera sensor.
        """
        self = weak_self()
        if self.capture:
            self.image = img
            if self.rgb_record:
                # Ensure the directory exists
                custom_dir = 'custom_data'
                if not os.path.exists(custom_dir):
                    os.makedirs(custom_dir)
                img.save_to_disk(os.path.join(custom_dir, 'image%d.png' % count))
                print("RGB Image saved")

                # Process the image data
                global rgb_info
                array = np.frombuffer(img.raw_data, dtype=np.dtype("uint8"))
                array = np.reshape(array, (img.height, img.width, 4))
                array = array[:, :, :3]
                array = array[:, :, ::-1]
                rgb_info = array

    @staticmethod
    def set_image_left(weak_self, img):
        """
        Sets image coming from left camera sensor.
        """
        self = weak_self()
        if self.capture:
            self.image_left = img
            if self.rgb_record:
                # Ensure the directory exists
                custom_dir = 'custom_data'
                if not os.path.exists(custom_dir):
                    os.makedirs(custom_dir)
                img.save_to_disk(os.path.join(custom_dir, 'image_left%d.png' % count))
                print("Left RGB Image saved")

                # Process the image the same way as the main camera
                global rgb_info_left
                array = np.frombuffer(img.raw_data, dtype=np.dtype("uint8"))
                array = np.reshape(array, (img.height, img.width, 4))
                array = array[:, :, :3]
                array = array[:, :, ::-1]
                rgb_info_left = array

    @staticmethod
    def set_image_front_left(weak_self, img):
        """
        Sets image coming from front left camera sensor.
        """
        self = weak_self()
        if self.capture:
            self.image_front_left = img
            if self.rgb_record:
                # Ensure the directory exists
                custom_dir = 'custom_data'
                if not os.path.exists(custom_dir):
                    os.makedirs(custom_dir)
                img.save_to_disk(os.path.join(custom_dir, 'image_front_left%d.png' % count))
                print("Front Left RGB Image saved")

                # Process the image the same way as the main camera
                global rgb_info_front_left
                array = np.frombuffer(img.raw_data, dtype=np.dtype("uint8"))
                array = np.reshape(array, (img.height, img.width, 4))
                array = array[:, :, :3]
                array = array[:, :, ::-1]
                rgb_info_front_left = array

    @staticmethod
    def set_image_right(weak_self, img):
        """
        Sets image coming from right camera sensor.
        """
        self = weak_self()
        if self.capture:
            self.image_right = img
            if self.rgb_record:
                # Ensure the directory exists
                custom_dir = 'custom_data'
                if not os.path.exists(custom_dir):
                    os.makedirs(custom_dir)
                img.save_to_disk(os.path.join(custom_dir, 'image_right%d.png' % count))
                print("Right RGB Image saved")

                # Process the image the same way as the main camera
                global rgb_info_right
                array = np.frombuffer(img.raw_data, dtype=np.dtype("uint8"))
                array = np.reshape(array, (img.height, img.width, 4))
                array = array[:, :, :3]
                array = array[:, :, ::-1]
                rgb_info_right = array

    @staticmethod
    def set_image_front_right(weak_self, img):
        """
        Sets image coming from front right camera sensor.
        """
        self = weak_self()
        if self.capture:
            self.image_front_right = img
            if self.rgb_record:
                # Ensure the directory exists
                custom_dir = 'custom_data'
                if not os.path.exists(custom_dir):
                    os.makedirs(custom_dir)
                img.save_to_disk(os.path.join(custom_dir, 'image_front_right%d.png' % count))
                print("Front Right RGB Image saved")

                # Process the image the same way as the main camera
                global rgb_info_front_right
                array = np.frombuffer(img.raw_data, dtype=np.dtype("uint8"))
                array = np.reshape(array, (img.height, img.width, 4))
                array = array[:, :, :3]
                array = array[:, :, ::-1]
                rgb_info_front_right = array

    @staticmethod
    def set_segmentation(weak_self, img):
        """
        Sets segmentation image coming from camera sensor.
        """
        self = weak_self()
        if self.capture:
            self.segmentation = img
            if self.seg_record:
                # Ensure the directory exists
                seg_dir = 'SegmentationImage'
                if not os.path.exists(seg_dir):
                    os.makedirs(seg_dir)
                img.save_to_disk(os.path.join(seg_dir, 'seg%d.png' % count), cc.CityScapesPalette)
                print("Segmentation Image saved")

                global seg_info
                array = np.frombuffer(img.raw_data, dtype=np.dtype("uint8"))
                array = np.reshape(array, (img.height, img.width, 4))
                array = array[:, :, :3]
                array = array[:, :, ::-1]
                seg_info = array

    @staticmethod
    def set_segmentation_left(weak_self, img):
        """
        Sets segmentation image coming from left camera sensor.
        """
        self = weak_self()
        if self.capture:
            self.segmentation_left = img
            if self.seg_record:
                # Ensure the directory exists
                seg_dir = 'SegmentationImage'
                if not os.path.exists(seg_dir):
                    os.makedirs(seg_dir)
                img.save_to_disk(os.path.join(seg_dir, 'seg_left%d.png' % count), cc.CityScapesPalette)
                print("Left Segmentation Image saved")

                global seg_info_left
                array = np.frombuffer(img.raw_data, dtype=np.dtype("uint8"))
                array = np.reshape(array, (img.height, img.width, 4))
                array = array[:, :, :3]
                array = array[:, :, ::-1]
                seg_info_left = array

    @staticmethod
    def set_segmentation_front_left(weak_self, img):
        """
        Sets segmentation image coming from front left camera sensor.
        """
        self = weak_self()
        if self.capture:
            self.segmentation_front_left = img
            if self.seg_record:
                # Ensure the directory exists
                seg_dir = 'SegmentationImage'
                if not os.path.exists(seg_dir):
                    os.makedirs(seg_dir)
                img.save_to_disk(os.path.join(seg_dir, 'seg_front_left%d.png' % count), cc.CityScapesPalette)
                print("Front Left Segmentation Image saved")

                global seg_info_front_left
                array = np.frombuffer(img.raw_data, dtype=np.dtype("uint8"))
                array = np.reshape(array, (img.height, img.width, 4))
                array = array[:, :, :3]
                array = array[:, :, ::-1]
                seg_info_front_left = array

    @staticmethod
    def set_segmentation_right(weak_self, img):
        """
        Sets segmentation image coming from right camera sensor.
        """
        self = weak_self()
        if self.capture:
            self.segmentation_right = img
            if self.seg_record:
                # Ensure the directory exists
                seg_dir = 'SegmentationImage'
                if not os.path.exists(seg_dir):
                    os.makedirs(seg_dir)
                img.save_to_disk(os.path.join(seg_dir, 'seg_right%d.png' % count), cc.CityScapesPalette)
                print("Right Segmentation Image saved")

                global seg_info_right
                array = np.frombuffer(img.raw_data, dtype=np.dtype("uint8"))
                array = np.reshape(array, (img.height, img.width, 4))
                array = array[:, :, :3]
                array = array[:, :, ::-1]
                seg_info_right = array

    @staticmethod
    def set_segmentation_front_right(weak_self, img):
        """
        Sets segmentation image coming from front right camera sensor.
        """
        self = weak_self()
        if self.capture:
            self.segmentation_front_right = img
            if self.seg_record:
                # Ensure the directory exists
                seg_dir = 'SegmentationImage'
                if not os.path.exists(seg_dir):
                    os.makedirs(seg_dir)
                img.save_to_disk(os.path.join(seg_dir, 'seg_front_right%d.png' % count), cc.CityScapesPalette)
                print("Front Right Segmentation Image saved")

                global seg_info_front_right
                array = np.frombuffer(img.raw_data, dtype=np.dtype("uint8"))
                array = np.reshape(array, (img.height, img.width, 4))
                array = array[:, :, :3]
                array = array[:, :, ::-1]
                seg_info_front_right = array

    def render(self, display):
        """
        Renders image coming from camera sensor.
        """
        if self.image is not None:
            array = np.frombuffer(self.image.raw_data, dtype=np.dtype("uint8"))
            array = np.reshape(array, (self.image.height, self.image.width, 4))
            array = array[:, :, :3]
            array = array[:, :, ::-1]

            global rgb_info
            rgb_info = array

            surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
            display.blit(surface, (0, 0))

    def game_loop(self, args):
        """
        Main program loop.
        """
        try:
            pygame.init()

            # Use the provided client and world if available, otherwise create new ones
            if self.client is None:
                self.client = carla.Client('127.0.0.1', 2000)
                self.client.set_timeout(20.0)  # Increased timeout for processing multiple cameras
            if self.world is None:
                self.world = self.client.get_world()

            self.setup_car()
            self.setup_camera()

            # Create a hidden pygame display for event handling
            # We're not using it for visualization anymore
            self.display = pygame.Surface((VIEW_WIDTH, VIEW_HEIGHT))
            pygame_clock = pygame.time.Clock()

            self.set_synchronous_mode(True)
            vehicles = self.world.get_actors().filter('vehicle.*')
            pedestrians = self.world.get_actors().filter('walker.pedestrian.*')

            self.image_count = 0
            self.time_interval = 0

            global vehicle_bbox_record
            global pedestrian_bbox_record
            global count
            global rgb_info
            global rgb_info_left
            global rgb_info_front_left
            global rgb_info_right
            global rgb_info_front_right
            global seg_info
            global seg_info_left
            global seg_info_front_left
            global seg_info_right
            global seg_info_front_right

            # For continuous data processing
            last_processed_count = -1

            # Set start time for timeout
            start_time = time.time()
            timeout_seconds = 180  # Run for 20 seconds then exit

            # Ensure directories exist
            my_data_dir = 'my_data'
            custom_dir = 'custom_data'
            draw_dir = 'draw_bounding_box'
            vehicle_bbox_dir = 'VehicleBBox'
            pedestrian_bbox_dir = 'PedestrianBBox'

            for directory in [my_data_dir, custom_dir, draw_dir, vehicle_bbox_dir, pedestrian_bbox_dir]:
                if not os.path.exists(directory):
                    os.makedirs(directory)
                    print(f"Created directory: {directory}")

            # Initialize train.txt file
            with open(os.path.join(my_data_dir, "train.txt"), 'w') as train:
                train.write("# Training data for continuous collection\n")

            while True:
                # Check if timeout has been reached
                current_time = time.time()
                elapsed_time = current_time - start_time
                if elapsed_time >= timeout_seconds:
                    print(f"Timeout reached after {elapsed_time:.2f} seconds.")
                    print("Processing final data and exiting...")

                    # Process any remaining data before exiting
                    vehicle_bbox_file = os.path.join('VehicleBBox', f'bbox{count}')
                    pedestrian_bbox_file = os.path.join('PedestrianBBox', f'bbox{count}')

                    if os.path.exists(vehicle_bbox_file) and os.path.exists(pedestrian_bbox_file):
                        try:
                            data = reading_data(count)
                            if data != False:
                                v_four_points = converting(data[0], data[1])
                                w_four_points = converting(data[2], data[3])
                                # Process main camera data
                                if 'seg_info' in globals() and seg_info is not None:
                                    processing(rgb_info, v_four_points, w_four_points, count, seg_info)
                                    with open(os.path.join(my_data_dir, "train.txt"), 'a') as train:
                                        train.write(os.path.join(custom_dir, f'image{count}.png') + '\n')

                                # Process left camera data
                                if 'rgb_info_left' in globals() and rgb_info_left is not None and 'seg_info_left' in globals() and seg_info_left is not None:
                                    processing(rgb_info_left, v_four_points, w_four_points, f"{count}_left", seg_info_left)
                                    with open(os.path.join(my_data_dir, "train.txt"), 'a') as train:
                                        train.write(os.path.join(custom_dir, f'image_left{count}.png') + '\n')

                                # Process front left camera data
                                if 'rgb_info_front_left' in globals() and rgb_info_front_left is not None and 'seg_info_front_left' in globals() and seg_info_front_left is not None:
                                    processing(rgb_info_front_left, v_four_points, w_four_points, f"{count}_front_left", seg_info_front_left)
                                    with open(os.path.join(my_data_dir, "train.txt"), 'a') as train:
                                        train.write(os.path.join(custom_dir, f'image_front_left{count}.png') + '\n')

                                # Process right camera data
                                if 'rgb_info_right' in globals() and rgb_info_right is not None and 'seg_info_right' in globals() and seg_info_right is not None:
                                    processing(rgb_info_right, v_four_points, w_four_points, f"{count}_right", seg_info_right)
                                    with open(os.path.join(my_data_dir, "train.txt"), 'a') as train:
                                        train.write(os.path.join(custom_dir, f'image_right{count}.png') + '\n')

                                # Process front right camera data
                                if 'rgb_info_front_right' in globals() and rgb_info_front_right is not None and 'seg_info_front_right' in globals() and seg_info_front_right is not None:
                                    processing(rgb_info_front_right, v_four_points, w_four_points, f"{count}_front_right", seg_info_front_right)
                                    with open(os.path.join(my_data_dir, "train.txt"), 'a') as train:
                                        train.write(os.path.join(custom_dir, f'image_front_right{count}.png') + '\n')

                                print(f"Processed final data for frame {count}")
                        except Exception as e:
                            print(f"Error processing final data: {e}")

                    print("Exiting script...")
                    break

                self.world.tick()

                # Update spectator to follow the vehicle
                if self.car is not None and self.spectator is not None:
                    # Get the car's transform
                    car_transform = self.car.get_transform()

                    # Calculate a position behind and above the car
                    spectator_transform = carla.Transform(
                        car_transform.location + carla.Location(
                            x=-5.5 * math.cos(math.radians(car_transform.rotation.yaw)),
                            y=-5.5 * math.sin(math.radians(car_transform.rotation.yaw)),
                            z=2.8
                        ),
                        carla.Rotation(
                            pitch=-15,
                            yaw=car_transform.rotation.yaw
                        )
                    )

                    # Set the spectator's transform
                    self.spectator.set_transform(spectator_transform)

                self.capture = True
                pygame_clock.tick_busy_loop(60)

                # We still need to render to the hidden surface for data processing
                self.render(self.display)

                # Continuously collect data in every frame
                self.image_count = self.image_count + 1
                self.rgb_record = True
                self.seg_record = True
                vehicle_bbox_record = True
                pedestrian_bbox_record = True
                count = self.image_count

                if self.image_count % 10 == 0:  # Print status every 10 frames to avoid console spam
                    print("-------------------------------------------------")
                    print("ImageCount - %d" % self.image_count)

                bounding_boxes = VehicleBoundingBoxes.get_bounding_boxes(vehicles, self.camera)
                pedestrian_bounding_boxes = PedestrianBoundingBoxes.get_bounding_boxes(pedestrians, self.camera)

                VehicleBoundingBoxes.draw_bounding_boxes(self.display, bounding_boxes)
                PedestrianBoundingBoxes.draw_bounding_boxes(self.display, pedestrian_bounding_boxes)

                # Process the data on the fly
                if last_processed_count != self.image_count:
                    try:
                        # Process the latest data
                        vehicle_bbox_file = os.path.join('VehicleBBox', f'bbox{count}')
                        pedestrian_bbox_file = os.path.join('PedestrianBBox', f'bbox{count}')

                        # Define custom_dir here to ensure it's in scope
                        custom_dir = 'custom_data'

                        if os.path.exists(vehicle_bbox_file) and os.path.exists(pedestrian_bbox_file):
                            data = reading_data(count)
                            if data != False:
                                v_four_points = converting(data[0], data[1])
                                w_four_points = converting(data[2], data[3])
                                # Process main camera data
                                if 'seg_info' in globals() and seg_info is not None:
                                    processing(rgb_info, v_four_points, w_four_points, count, seg_info)
                                    with open(os.path.join(my_data_dir, "train.txt"), 'a') as train:
                                        train.write(os.path.join(custom_dir, f'image{count}.png') + '\n')

                                # Process left camera data
                                if 'rgb_info_left' in globals() and rgb_info_left is not None and 'seg_info_left' in globals() and seg_info_left is not None:
                                    processing(rgb_info_left, v_four_points, w_four_points, f"{count}_left", seg_info_left)
                                    with open(os.path.join(my_data_dir, "train.txt"), 'a') as train:
                                        train.write(os.path.join(custom_dir, f'image_left{count}.png') + '\n')

                                # Process front left camera data
                                if 'rgb_info_front_left' in globals() and rgb_info_front_left is not None and 'seg_info_front_left' in globals() and seg_info_front_left is not None:
                                    processing(rgb_info_front_left, v_four_points, w_four_points, f"{count}_front_left", seg_info_front_left)
                                    with open(os.path.join(my_data_dir, "train.txt"), 'a') as train:
                                        train.write(os.path.join(custom_dir, f'image_front_left{count}.png') + '\n')

                                # Process right camera data
                                if 'rgb_info_right' in globals() and rgb_info_right is not None and 'seg_info_right' in globals() and seg_info_right is not None:
                                    processing(rgb_info_right, v_four_points, w_four_points, f"{count}_right", seg_info_right)
                                    with open(os.path.join(my_data_dir, "train.txt"), 'a') as train:
                                        train.write(os.path.join(custom_dir, f'image_right{count}.png') + '\n')

                                # Process front right camera data
                                if 'rgb_info_front_right' in globals() and rgb_info_front_right is not None and 'seg_info_front_right' in globals() and seg_info_front_right is not None:
                                    processing(rgb_info_front_right, v_four_points, w_four_points, f"{count}_front_right", seg_info_front_right)
                                    with open(os.path.join(my_data_dir, "train.txt"), 'a') as train:
                                        train.write(os.path.join(custom_dir, f'image_front_right{count}.png') + '\n')

                                print(f"Processed data for frame {count}")
                                last_processed_count = self.image_count
                    except Exception as e:
                        print(f"Error processing data: {e}")

                # Add a small delay to maintain consistent frame rate
                time.sleep(0.03)

                # No need to flip the display since we're not showing it
                # But we still need to handle pygame events
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        return
                    elif event.type == pygame.KEYDOWN:
                        if event.key == K_ESCAPE:
                            return

        finally:
            try:
                self.set_synchronous_mode(False)
                if self.camera is not None:
                    self.camera.destroy()
                if self.camera_segmentation is not None:
                    self.camera_segmentation.destroy()
                if self.camera_left is not None:
                    self.camera_left.destroy()
                if self.camera_front_left is not None:
                    self.camera_front_left.destroy()
                if self.camera_right is not None:
                    self.camera_right.destroy()
                if self.camera_front_right is not None:
                    self.camera_front_right.destroy()
                if self.camera_segmentation_left is not None:
                    self.camera_segmentation_left.destroy()
                if self.camera_segmentation_front_left is not None:
                    self.camera_segmentation_front_left.destroy()
                if self.camera_segmentation_right is not None:
                    self.camera_segmentation_right.destroy()
                if self.camera_segmentation_front_right is not None:
                    self.camera_segmentation_front_right.destroy()
                if self.car is not None:
                    self.car.destroy()
                # No need to destroy the spectator as it's a built-in CARLA object
                pygame.quit()
            except Exception as e:
                print(f"Error during cleanup in game_loop: {e}")

# Functions from generate.py.py
def reading_data(index):
    global rgb_info, seg_info, seg_info_left, seg_info_front_left, seg_info_right, seg_info_front_right
    v_data = []
    w_data = []
    k = 0
    w = 0

    rgb_path = 'custom_data/image'+ str(index)+ '.png'
    seg_path = 'SegmentationImage/seg'+ str(index)+ '.png'

    # Paths for the new camera segmentation images
    seg_left_path = 'SegmentationImage/seg_left'+ str(index)+ '.png'
    seg_front_left_path = 'SegmentationImage/seg_front_left'+ str(index)+ '.png'
    seg_right_path = 'SegmentationImage/seg_right'+ str(index)+ '.png'
    seg_front_right_path = 'SegmentationImage/seg_front_right'+ str(index)+ '.png'

    # Check if files exist before trying to read them
    if not os.path.exists(rgb_path):
        print(f"RGB Image file not found: {rgb_path}")
        return False
    if not os.path.exists(seg_path):
        print(f"Segmentation Image file not found: {seg_path}")
        return False

    rgb_img = cv2.imread(rgb_path, cv2.IMREAD_COLOR)
    seg_img = cv2.imread(seg_path, cv2.IMREAD_COLOR)

    # Update the global seg_info variable with the segmentation image data
    seg_info = seg_img
    print(f"Updated seg_info with segmentation data from: {seg_path}")

    # Read and update segmentation data for the new cameras if available
    if os.path.exists(seg_left_path):
        seg_left_img = cv2.imread(seg_left_path, cv2.IMREAD_COLOR)
        seg_info_left = seg_left_img
        print(f"Updated seg_info_left with segmentation data from: {seg_left_path}")

    if os.path.exists(seg_front_left_path):
        seg_front_left_img = cv2.imread(seg_front_left_path, cv2.IMREAD_COLOR)
        seg_info_front_left = seg_front_left_img
        print(f"Updated seg_info_front_left with segmentation data from: {seg_front_left_path}")

    if os.path.exists(seg_right_path):
        seg_right_img = cv2.imread(seg_right_path, cv2.IMREAD_COLOR)
        seg_info_right = seg_right_img
        print(f"Updated seg_info_right with segmentation data from: {seg_right_path}")

    if os.path.exists(seg_front_right_path):
        seg_front_right_img = cv2.imread(seg_front_right_path, cv2.IMREAD_COLOR)
        seg_info_front_right = seg_front_right_img
        print(f"Updated seg_info_front_right with segmentation data from: {seg_front_right_path}")

    if str(rgb_img) != "None" and str(seg_img) != "None":
        # Check if VehicleBBox directory exists
        vehicle_bbox_dir = 'VehicleBBox'
        if not os.path.exists(vehicle_bbox_dir):
            print(f"Vehicle bounding box directory not found: {vehicle_bbox_dir}")
            return False

        vehicle_bbox_file = os.path.join(vehicle_bbox_dir, 'bbox' + str(index))
        if not os.path.exists(vehicle_bbox_file):
            print(f"Vehicle bounding box file not found: {vehicle_bbox_file}")
            return False

        # Check if PedestrianBBox directory exists
        pedestrian_bbox_dir = 'PedestrianBBox'
        if not os.path.exists(pedestrian_bbox_dir):
            print(f"Pedestrian bounding box directory not found: {pedestrian_bbox_dir}")
            return False

        pedestrian_bbox_file = os.path.join(pedestrian_bbox_dir, 'bbox' + str(index))
        if not os.path.exists(pedestrian_bbox_file):
            print(f"Pedestrian bounding box file not found: {pedestrian_bbox_file}")
            return False

        # Vehicle
        with open(vehicle_bbox_file, 'r') as fin:
            v_bounding_box_rawdata = fin.read()

        v_bounding_box_rawdata = v_bounding_box_rawdata.split('\n')
        v_bounding_box_rawdata = list(filter(None, v_bounding_box_rawdata))

        for i in range(len(v_bounding_box_rawdata)):
            v_data.append([])
            v_bounding_box_rawdata[i] = v_bounding_box_rawdata[i].split(' ')
            v_bounding_box_rawdata[i] = list(filter(None, v_bounding_box_rawdata[i]))

            for j in range(len(v_bounding_box_rawdata[i])):
                v_data[k].append(v_bounding_box_rawdata[i][j].split(','))
                v_data[k][j][0] = int(v_data[k][j][0])
                v_data[k][j][1] = int(v_data[k][j][1])
            k = k + 1

        # Walker (Pedestrian)
        with open(pedestrian_bbox_file, 'r') as fin:
            w_bounding_box_rawdata = fin.read()

        w_bounding_box_rawdata = w_bounding_box_rawdata.split('\n')
        w_bounding_box_rawdata = list(filter(None, w_bounding_box_rawdata))

        for i in range(len(w_bounding_box_rawdata)):
            w_data.append([])
            w_bounding_box_rawdata[i] = w_bounding_box_rawdata[i].split(' ')
            w_bounding_box_rawdata[i] = list(filter(None, w_bounding_box_rawdata[i]))

            for j in range(len(w_bounding_box_rawdata[i])):
                w_data[w].append(w_bounding_box_rawdata[i][j].split(','))
                w_data[w][j][0] = int(w_data[w][j][0])
                w_data[w][j][1] = int(w_data[w][j][1])
            w = w + 1

        return v_data, len(v_bounding_box_rawdata), w_data, len(w_bounding_box_rawdata)
    else:
        return False

def converting(bounding_boxes, line_length):
    points_array = []
    bb_4data = [[0 for col in range(4)] for row in range(line_length)]
    k = 0
    for i in range(line_length):
        points_array_x = []
        points_array_y = []      
        for j in range(8):
            points_array_x.append(bounding_boxes[i][j][0])
            points_array_y.append(bounding_boxes[i][j][1])

        max_x = max(points_array_x)
        min_x = min(points_array_x)
        max_y = max(points_array_y)
        min_y = min(points_array_y)           

        points_array.append(tuple((min_x, min_y)))
        points_array.append(tuple((max_x, min_y)))
        points_array.append(tuple((max_x, max_y)))
        points_array.append(tuple((min_x, max_y)))

    for i in range(line_length):
        for j in range(len(points_array)//line_length):
            bb_4data[i][j] = points_array[k]
            k += 1  

    return bb_4data

def object_area(data, segmentation_data):
    global area_info

    # Initialize area_info with zeros
    area_info = np.zeros(shape=[VIEW_HEIGHT, VIEW_WIDTH, 3], dtype=np.uint8)

    # Print debug information
    print(f"object_area: segmentation_data shape = {segmentation_data.shape}, area_info shape = {area_info.shape}")

    # Copy segmentation_data to area_info for the entire image
    # This ensures area_info is fully populated with segmentation data
    area_info[:] = segmentation_data[:]

    print("object_area: Copied segmentation data to area_info")

    for bbox in data:
        array_x = []
        array_y = []

        for i in range(4):
           array_x.append(bbox[i][0])
        for j in range(4):
           array_y.append(bbox[j][1])

        for i in range(4):
            if array_x[i] <= 0:
                array_x[i] = 1
            elif array_x[i] >= VIEW_WIDTH - 1:
                array_x[i] = VIEW_WIDTH - 2
        for j in range(4):
            if array_y[j] <= 0:
                array_y[j] = 1
            elif array_y[j] >= VIEW_HEIGHT - 1:
                array_y[j] = VIEW_HEIGHT - 2

        min_x = min(array_x) 
        max_x = max(array_x) 
        min_y = min(array_y) 
        max_y = max(array_y) 

        print(f"object_area: Processing bounding box with coordinates: min_x={min_x}, max_x={max_x}, min_y={min_y}, max_y={max_y}")

def fitting_x(x1, x2, range_min, range_max, color):
    global area_info

    if x1 <= 0:
        x1 = 1
    elif x1 >= VIEW_WIDTH - 1:
        x1 = VIEW_WIDTH - 2

    if range_min <= 0:
        range_min = 1
    elif range_min >= VIEW_HEIGHT - 1:
        range_min = VIEW_HEIGHT - 2

    if range_max <= 0:
        range_max = 1
    elif range_max >= VIEW_HEIGHT - 1:
        range_max = VIEW_HEIGHT - 2

    if x1 < x2:
        for i in range(range_min, range_max):
            for j in range(x1, VIEW_WIDTH - 1):
                if (area_info[i][j][0] == color[0] and area_info[i][j][1] == color[1] and area_info[i][j][2] == color[2]):
                    return j
        return x1
    else:
        for i in range(range_min, range_max):
            for j in range(x1, 0, -1):
                if (area_info[i][j][0] == color[0] and area_info[i][j][1] == color[1] and area_info[i][j][2] == color[2]):
                    return j
        return x1

def fitting_y(y1, y2, range_min, range_max, color):
    global area_info

    if y1 <= 0:
        y1 = 1
    elif y1 >= VIEW_HEIGHT - 1:
        y1 = VIEW_HEIGHT - 2

    if range_min <= 0:
        range_min = 1
    elif range_min >= VIEW_WIDTH - 1:
        range_min = VIEW_WIDTH - 2

    if range_max <= 0:
        range_max = 1
    elif range_max >= VIEW_WIDTH - 1:
        range_max = VIEW_WIDTH - 2

    if y1 < y2:
        for i in range(range_min, range_max):
            for j in range(y1, VIEW_HEIGHT - 1):
                if (area_info[j][i][0] == color[0] and area_info[j][i][1] == color[1] and area_info[j][i][2] == color[2]):
                    return j
        return y1
    else:
        for i in range(range_min, range_max):
            for j in range(y1, 0, -1):
                if (area_info[j][i][0] == color[0] and area_info[j][i][1] == color[1] and area_info[j][i][2] == color[2]):
                    return j
        return y1

def small_objects_excluded(array, bb_min):
    if (array[1] - array[0]) * (array[3] - array[2]) < bb_min * bb_min:
        return False
    else:
        return True

def post_occluded_objects_excluded(array, color):
    global area_info

    center_x = (array[0] + array[1])//2
    center_y = (array[2] + array[3])//2

    if (area_info[center_y][center_x][0] == color[0] and area_info[center_y][center_x][1] == color[1] and area_info[center_y][center_x][2] == color[2]):
        return True
    else:
        return False

def pre_occluded_objects_excluded(array, area_image, color):
    center_x = (array[0] + array[1])//2
    center_y = (array[2] + array[3])//2

    if (area_image[center_y][center_x][0] == color[0] and area_image[center_y][center_x][1] == color[1] and area_image[center_y][center_x][2] == color[2]):
        return True
    else:
        return False

def filtering(array, color, segmentation_data):
    center_x = (array[0] + array[1])//2
    center_y = (array[2] + array[3])//2

    if (segmentation_data[center_y][center_x][0] == color[0] and segmentation_data[center_y][center_x][1] == color[1] and segmentation_data[center_y][center_x][2] == color[2]):
        return True
    else:
        return False

def processing(img, v_data, w_data, index, segmentation_data):
    global area_info

    vehicle_class = 0
    walker_class = 1

    # Ensure the custom_data directory exists
    custom_dir = 'custom_data'
    if not os.path.exists(custom_dir):
        os.makedirs(custom_dir)

    # Create a copy of the image for drawing bounding boxes
    draw_img = img.copy()

    # Open the text file for writing
    txt_file_path = os.path.join(custom_dir, 'image'+str(index)+'.txt')
    f = open(txt_file_path, 'w')
    print(f"Writing bounding box data to: {txt_file_path}")

    # Vehicle
    object_area(v_data, segmentation_data)

    for v_bbox in v_data:
        array_x = []
        array_y = []

        for i in range(4):
           array_x.append(v_bbox[i][0])
        for j in range(4):
           array_y.append(v_bbox[j][1])

        for i in range(4):
            if array_x[i] <= 0:
                array_x[i] = 1
            elif array_x[i] >= VIEW_WIDTH - 1:
                array_x[i] = VIEW_WIDTH - 2
        for j in range(4):
            if array_y[j] <= 0:
                array_y[j] = 1
            elif array_y[j] >= VIEW_HEIGHT - 1:
                array_y[j] = VIEW_HEIGHT - 2

        min_x = min(array_x) 
        max_x = max(array_x) 
        min_y = min(array_y) 
        max_y = max(array_y) 
        v_bb_array = [min_x, max_x, min_y, max_y]
        center_x = (min_x + max_x)//2
        center_y = (min_y + max_y)//2

        if filtering(v_bb_array, Vehicle_COLOR, segmentation_data) and pre_occluded_objects_excluded(v_bb_array, area_info, Vehicle_COLOR): 
            cali_min_x = fitting_x(min_x, max_x, min_y, max_y, Vehicle_COLOR)
            cali_max_x = fitting_x(max_x, min_x, min_y, max_y, Vehicle_COLOR)
            cali_min_y = fitting_y(min_y, max_y, min_x, max_x, Vehicle_COLOR)
            cali_max_y = fitting_y(max_y, min_y, min_x, max_x, Vehicle_COLOR)
            v_cali_array = [cali_min_x, cali_max_x, cali_min_y, cali_max_y]

            if small_objects_excluded(v_cali_array, 10) and post_occluded_objects_excluded(v_cali_array, Vehicle_COLOR):
                darknet_x = float((cali_min_x + cali_max_x) // 2) / float(VIEW_WIDTH)
                darknet_y = float((cali_min_y + cali_max_y) // 2) / float(VIEW_HEIGHT)
                darknet_width = float(cali_max_x - cali_min_x) / float(VIEW_WIDTH)
                darknet_height= float(cali_max_y - cali_min_y) / float(VIEW_HEIGHT)

                # Write to the text file
                f.write(str(vehicle_class) + ' ' + str("%0.6f" % darknet_x) + ' ' + str("%0.6f" % darknet_y) + ' ' + 
                str("%0.6f" % darknet_width) + ' ' + str("%0.6f" % darknet_height) + "\n")

                # Draw the bounding box on the image
                cv2.line(draw_img, (cali_min_x, cali_min_y), (cali_max_x, cali_min_y), VBB_COLOR, 2)
                cv2.line(draw_img, (cali_max_x, cali_min_y), (cali_max_x, cali_max_y), VBB_COLOR, 2)
                cv2.line(draw_img, (cali_max_x, cali_max_y), (cali_min_x, cali_max_y), VBB_COLOR, 2)
                cv2.line(draw_img, (cali_min_x, cali_max_y), (cali_min_x, cali_min_y), VBB_COLOR, 2)

    # Walker (Pedestrian)
    object_area(w_data, segmentation_data)

    for wbbox in w_data:
        array_x = []
        array_y = []

        for i in range(4):
           array_x.append(wbbox[i][0])
        for j in range(4):
           array_y.append(wbbox[j][1])

        for i in range(4):
            if array_x[i] <= 0:
                array_x[i] = 1
            elif array_x[i] >= VIEW_WIDTH - 1:
                array_x[i] = VIEW_WIDTH - 2
        for j in range(4):
            if array_y[j] <= 0:
                array_y[j] = 1
            elif array_y[j] >= VIEW_HEIGHT - 1:
                array_y[j] = VIEW_HEIGHT - 2

        min_x = min(array_x) 
        max_x = max(array_x) 
        min_y = min(array_y) 
        max_y = max(array_y)
        w_bb_array = [min_x, max_x, min_y, max_y]
        if filtering(w_bb_array, Walker_COLOR, segmentation_data) and pre_occluded_objects_excluded(w_bb_array, area_info, Walker_COLOR): 
            cali_min_x = fitting_x(min_x, max_x, min_y, max_y, Walker_COLOR)
            cali_max_x = fitting_x(max_x, min_x, min_y, max_y, Walker_COLOR)
            cali_min_y = fitting_y(min_y, max_y, min_x, max_x, Walker_COLOR)
            cali_max_y = fitting_y(max_y, min_y, min_x, max_x, Walker_COLOR)
            w_cali_array = [cali_min_x, cali_max_x, cali_min_y, cali_max_y]

            if small_objects_excluded(w_cali_array, 7) and post_occluded_objects_excluded(w_cali_array, Walker_COLOR):
                darknet_x = float((cali_min_x + cali_max_x) // 2) / float(VIEW_WIDTH)
                darknet_y = float((cali_min_y + cali_max_y) // 2) / float(VIEW_HEIGHT)
                darknet_width = float(cali_max_x - cali_min_x) / float(VIEW_WIDTH)
                darknet_height= float(cali_max_y - cali_min_y) / float(VIEW_HEIGHT)

                # Write to the text file
                f.write(str(walker_class) + ' ' + str("%0.6f" % darknet_x) + ' ' + str("%0.6f" % darknet_y) + ' ' + 
                str("%0.6f" % darknet_width) + ' ' + str("%0.6f" % darknet_height) + "\n")

                # Draw the bounding box on the image
                cv2.line(draw_img, (cali_min_x, cali_min_y), (cali_max_x, cali_min_y), WBB_COLOR, 2)
                cv2.line(draw_img, (cali_max_x, cali_min_y), (cali_max_x, cali_max_y), WBB_COLOR, 2)
                cv2.line(draw_img, (cali_max_x, cali_max_y), (cali_min_x, cali_max_y), WBB_COLOR, 2)
                cv2.line(draw_img, (cali_min_x, cali_max_y), (cali_min_x, cali_min_y), WBB_COLOR, 2)

    # Close the text file
    f.close()
    print(f"Bounding box data written to: {txt_file_path}")

    # Ensure the draw_bounding_box directory exists before writing the file
    draw_dir = 'draw_bounding_box'
    if not os.path.exists(draw_dir):
        os.makedirs(draw_dir)

    # Save the image with bounding boxes
    draw_img_path = os.path.join(draw_dir, 'image'+str(index)+'.png')
    cv2.imwrite(draw_img_path, draw_img)
    print(f"Image with bounding boxes saved to: {draw_img_path}")

    # Also save the original image to custom_data
    img_path = os.path.join(custom_dir, 'image'+str(index)+'.png')
    cv2.imwrite(img_path, img)
    print(f"Original image saved to: {img_path}")

def run_generate():
    global rgb_info
    global index_count

    dataEA = len(next(os.walk('VehicleBBox/'))[2])
    train = open("my_data/train.txt", 'w')

    for i in range(dataEA + 1):
        data = reading_data(i)
        if data != False:
            v_four_points = converting(data[0], data[1])
            w_four_points = converting(data[2], data[3])
            if 'seg_info' in globals() and seg_info is not None:
                processing(rgb_info, v_four_points, w_four_points, i, seg_info)
                train.write(str('custom_data/image'+str(i) + '.png') + "\n")
            index_count = index_count + 1
            print("Processed data for frame", i)
    train.close()
    print("-------------------------------------------------")
    print("ImageCount -", index_count)

# Functions from spawn.py.py
def spawn_vehicles_and_walkers(args):
    """
    Spawns vehicles and walkers in the CARLA world
    """
    vehicles_list = []
    walkers_list = []
    all_id = []
    client = carla.Client(args.host, args.port)
    client.set_timeout(20.0)  # Increased timeout for processing multiple cameras

    try:
        world = client.get_world()
        blueprints = world.get_blueprint_library().filter(args.filterv)
        blueprintsWalkers = world.get_blueprint_library().filter(args.filterw)

        if args.safe:
            blueprints = [x for x in blueprints if int(x.get_attribute('number_of_wheels')) == 4]
            blueprints = [x for x in blueprints if not x.id.endswith('isetta')]
            blueprints = [x for x in blueprints if not x.id.endswith('carlacola')]

        spawn_points = world.get_map().get_spawn_points()
        number_of_spawn_points = len(spawn_points)

        if args.number_of_vehicles < number_of_spawn_points:
            random.shuffle(spawn_points)
        elif args.number_of_vehicles > number_of_spawn_points:
            msg = 'requested %d vehicles, but could only find %d spawn points'
            logging.warning(msg, args.number_of_vehicles, number_of_spawn_points)
            args.number_of_vehicles = number_of_spawn_points

        # @todo cannot import these directly.
        SpawnActor = carla.command.SpawnActor
        SetAutopilot = carla.command.SetAutopilot
        FutureActor = carla.command.FutureActor

        # --------------
        # Spawn vehicles
        # --------------
        batch = []
        for n, transform in enumerate(spawn_points):
            if n >= args.number_of_vehicles:
                break
            blueprint = random.choice(blueprints)
            if blueprint.tags[0] != "crossbike" and blueprint.tags[0] != "low rider" and blueprint.tags[0] != "ninja" and blueprint.tags[0] != "yzf" and blueprint.tags[0] != "century" and blueprint.tags[0] != "omafiets" and blueprint.tags[0] != "diamondback" and blueprint.tags[0] != "carlacola":
                if blueprint.has_attribute('color'):
                    color = random.choice(blueprint.get_attribute('color').recommended_values)
                    blueprint.set_attribute('color', color)
                if blueprint.has_attribute('driver_id'):
                    driver_id = random.choice(blueprint.get_attribute('driver_id').recommended_values)
                    blueprint.set_attribute('driver_id', driver_id)
                blueprint.set_attribute('role_name', 'autopilot')
                batch.append(SpawnActor(blueprint, transform).then(SetAutopilot(FutureActor, True)))

        for response in client.apply_batch_sync(batch):
            if response.error:
                logging.error(response.error)
            else:
                vehicles_list.append(response.actor_id)

        # -------------
        # Spawn Walkers
        # -------------
        # some settings
        percentagePedestriansRunning = 0.0      # how many pedestrians will run
        percentagePedestriansCrossing = 0.0     # how many pedestrians will walk through the road
        # 1. take all the random locations to spawn
        spawn_points = []
        for i in range(args.number_of_walkers):
            spawn_point = carla.Transform()
            loc = world.get_random_location_from_navigation()
            if (loc != None):
                spawn_point.location = loc
                spawn_points.append(spawn_point)
        # 2. we spawn the walker object
        batch = []
        walker_speed = []
        for spawn_point in spawn_points:
            walker_bp = random.choice(blueprintsWalkers)
            # set as not invincible
            if walker_bp.has_attribute('is_invincible'):
                walker_bp.set_attribute('is_invincible', 'false')
            # set the max speed
            if walker_bp.has_attribute('speed'):
                if (random.random() > percentagePedestriansRunning):
                    # walking
                    walker_speed.append(walker_bp.get_attribute('speed').recommended_values[1])
                else:
                    # running
                    walker_speed.append(walker_bp.get_attribute('speed').recommended_values[2])
            else:
                print("Walker has no speed")
                walker_speed.append(0.0)
            batch.append(SpawnActor(walker_bp, spawn_point))
        results = client.apply_batch_sync(batch, True)
        walker_speed2 = []
        for i in range(len(results)):
            if results[i].error:
                logging.error(results[i].error)
            else:
                walkers_list.append({"id": results[i].actor_id})
                walker_speed2.append(walker_speed[i])
        walker_speed = walker_speed2
        # 3. we spawn the walker controller
        batch = []
        walker_controller_bp = world.get_blueprint_library().find('controller.ai.walker')
        for i in range(len(walkers_list)):
            batch.append(SpawnActor(walker_controller_bp, carla.Transform(), walkers_list[i]["id"]))
        results = client.apply_batch_sync(batch, True)
        for i in range(len(results)):
            if results[i].error:
                logging.error(results[i].error)
            else:
                walkers_list[i]["con"] = results[i].actor_id
        # 4. we put altogether the walkers and controllers id to get the objects from their id
        for i in range(len(walkers_list)):
            all_id.append(walkers_list[i]["con"])
            all_id.append(walkers_list[i]["id"])
        all_actors = world.get_actors(all_id)

        # wait for a tick to ensure client receives the last transform of the walkers we have just created
        world.wait_for_tick()

        # 5. initialize each controller and set target to walk to (list is [controller, actor, controller, actor ...])
        # set how many pedestrians can cross the road
        world.set_pedestrians_cross_factor(percentagePedestriansCrossing)
        for i in range(0, len(all_id), 2):
            # start walker
            all_actors[i].start()
            # set walk to random point
            all_actors[i].go_to_location(world.get_random_location_from_navigation())
            # max speed
            all_actors[i].set_max_speed(float(walker_speed[int(i/2)]))

        print('spawned %d vehicles and %d walkers, press Ctrl+C to exit.' % (len(vehicles_list), len(walkers_list)))

        # example of how to use parameters
        traffic_manager = client.get_trafficmanager()
        traffic_manager.global_percentage_speed_difference(30.0)

        return client, world, vehicles_list, all_id, walkers_list

    except KeyboardInterrupt:
        print('\nCancelled by user. Cleaning up...')
        client.apply_batch([carla.command.DestroyActor(x) for x in vehicles_list])
        client.apply_batch([carla.command.DestroyActor(x) for x in all_id])
        time.sleep(0.5)
        return None, None, [], [], []

# Function to change weather
def change_weather(client, world, speed_factor=1.0, update_freq=0.1):
    """
    Changes weather conditions in the CARLA world
    """
    weather = Weather(world.get_weather())
    elapsed_time = 0.0

    # Update weather for a short time to see changes
    for _ in range(10):  # Update weather 10 times
        timestamp = world.wait_for_tick(seconds=30.0).timestamp
        elapsed_time += timestamp.delta_seconds
        if elapsed_time > update_freq:
            weather.tick(speed_factor * elapsed_time)
            world.set_weather(weather.weather)
            sys.stdout.write('\r' + str(weather) + 12 * ' ')
            sys.stdout.flush()
            elapsed_time = 0.0
            time.sleep(0.1)  # Short delay between updates

    print("\nWeather changed successfully")

# Main function that runs the complete pipeline
def main():
    """
    Main function that runs the complete pipeline:
    1. Spawn vehicles and pedestrians
    2. Change weather conditions
    3. Extract data from the simulation
    4. Process the extracted data
    """
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument('--host', metavar='H', default='127.0.0.1', help='IP of the host server (default: 127.0.0.1)')
    argparser.add_argument('-p', '--port', metavar='P', default=2000, type=int, help='TCP port to listen to (default: 2000)')
    argparser.add_argument('-n', '--number-of-vehicles', metavar='N', default=10, type=int, help='number of vehicles (default: 10)')
    argparser.add_argument('-w', '--number-of-walkers', metavar='W', default=50, type=int, help='number of walkers (default: 50)')
    argparser.add_argument('--safe', action='store_true', help='avoid spawning vehicles prone to accidents')
    argparser.add_argument('--filterv', metavar='PATTERN', default='vehicle.*', help='vehicles filter (default: "vehicle.*")')
    argparser.add_argument('--filterw', metavar='PATTERN', default='walker.pedestrian.*', help='pedestrians filter (default: "walker.pedestrian.*")')
    argparser.add_argument('-s', '--speed', metavar='FACTOR', default=1.0, type=float, help='rate at which the weather changes (default: 1.0)')
    argparser.add_argument('-l', '--CaptureLoop', metavar='N', default=100, type=int, help='set Capture Cycle settings, Recommand : above 100')

    args = argparser.parse_args()

    logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)

    try:
        # Step 1: Spawn vehicles and pedestrians
        print("Step 1: Spawning vehicles and pedestrians...")
        client, world, vehicles_list, all_id, walkers_list = spawn_vehicles_and_walkers(args)
        if client is None:
            return

        # Step 2: Change weather conditions
        print("\nStep 2: Changing weather conditions...")
        change_weather(client, world, args.speed)

        # Step 3 & 4: Extract data and process it on the fly
        print("\nStep 3 & 4: Extracting and processing data...")
        sync_client = None
        try:
            # Create a BasicSynchronousClient with the existing client and world
            sync_client = BasicSynchronousClient(client, world)
            sync_client.game_loop(args)

            # Data is already processed on the fly in game_loop, no need to call run_generate

        finally:
            print('Data extraction and processing completed')

        print("\nPipeline completed successfully!")

    except KeyboardInterrupt:
        print('\nCancelled by user. Cleaning up...')
    except Exception as e:
        print(f'Error occurred: {e}')
    finally:
        # Clean up
        print("\nCleaning up...")
        try:
            # Use the client from sync_client if available, otherwise use the original client
            cleanup_client = sync_client.client if sync_client is not None and hasattr(sync_client, 'client') and sync_client.client is not None else client
            if cleanup_client is not None:
                if 'vehicles_list' in locals() and vehicles_list:
                    cleanup_client.apply_batch([carla.command.DestroyActor(x) for x in vehicles_list])
                if 'all_id' in locals() and all_id:
                    cleanup_client.apply_batch([carla.command.DestroyActor(x) for x in all_id])
                time.sleep(0.5)
            else:
                print("Warning: Client is None, skipping actor cleanup")
        except Exception as e:
            print(f"Error during cleanup: {e}")
        print('EXIT')

if __name__ == '__main__':
    main()
