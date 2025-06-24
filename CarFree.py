#!/usr/bin/env python

"""
An example of client-side bounding boxes with basic car controls, data extraction and processing.
The main CARLA simulator window is used to view the vehicle.

Controls:
Welcome to CARLA for Getting Bounding Box Data and Processing.
Use WASD keys for control.
    W            : throttle
    S            : brake
    AD           : steer
    Space        : hand-brake
    P            : autopilot mode
    C            : Capture Data
    l            : Loop Capture Start
    L            : Loop Capture End
    G            : Generate and process data
    ESC          : quit
"""

# ==============================================================================
# -- find carla module ---------------------------------------------------------
# ==============================================================================


import glob
import os
import sys

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass


# ==============================================================================
# -- imports -------------------------------------------------------------------
# ==============================================================================

import carla

from carla import ColorConverter as cc

import weakref
import random
import cv2
import time
import argparse
import textwrap
import copy
import re
import os.path
import math

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
    from pygame.locals import K_g
except ImportError:
    raise RuntimeError('cannot import pygame, make sure pygame package is installed')

try:
    import numpy as np
except ImportError:
    raise RuntimeError('cannot import numpy, make sure numpy package is installed')

# ==============================================================================
# -- Global Variables and Constants --------------------------------------------
# ==============================================================================

VIEW_WIDTH = 1920//2
VIEW_HEIGHT = 1080//2
VIEW_FOV = 90

BB_COLOR = (248, 64, 24)
VBB_COLOR = (0, 0, 255)
WBB_COLOR = (255, 0, 0)
Vehicle_COLOR = (142, 0, 0)
Walker_COLOR = (60, 20, 220)

vehicle_bbox_record = False
pedestrian_bbox_record = False
count = 0

rgb_info = np.zeros((VIEW_HEIGHT, VIEW_WIDTH, 3), dtype="i")
seg_info = np.zeros((VIEW_HEIGHT, VIEW_WIDTH, 3), dtype="i")
area_info = np.zeros(shape=[VIEW_HEIGHT, VIEW_WIDTH, 3], dtype=np.uint8)
index_count = 0

# Creates Directory
dir_rgb = 'custom_data/'
dir_seg = 'SegmentationImage/'
dir_pbbox = 'PedestrianBBox/'
dir_vbbox = 'VehicleBBox/'
dir_my = 'my_data/'
dir_draw = 'draw_bounding_box/'

if not os.path.exists(dir_rgb):
    os.makedirs(dir_rgb)
if not os.path.exists(dir_seg):
    os.makedirs(dir_seg)
if not os.path.exists(dir_pbbox):
    os.makedirs(dir_pbbox)
if not os.path.exists(dir_vbbox):
    os.makedirs(dir_vbbox)
if not os.path.exists(dir_my):
    os.makedirs(dir_my)
if not os.path.exists(dir_draw):
    os.makedirs(dir_draw)

# ==============================================================================
# -- PedestrianBoundingBoxes ---------------------------------------------------
# ==============================================================================

class PedestrianBoundingBoxes(object):
    """
    This is a module responsible for creating 3D bounding boxes and drawing them
    client-side on pygame surface.
    """

    @staticmethod
    def get_bounding_boxes(pedestrians, camera):
        """
        Creates 3D bounding boxes based on carla Pedestrian list and camera.
        """

        bounding_boxes = [PedestrianBoundingBoxes.get_bounding_box(pedestrian, camera) for pedestrian in pedestrians]
        # filter objects behind camera
        bounding_boxes = [bb for bb in bounding_boxes if all(bb[:, 2] > 0)]
        return bounding_boxes

    @staticmethod
    def draw_bounding_boxes(bounding_boxes):
        """
        Records bounding boxes data to file.
        """
        global pedestrian_bbox_record
        global count

        if pedestrian_bbox_record == True:
            f = open("PedestrianBBox/bbox"+str(count), 'w')
            print("PedestrianBoundingBox")
            for bbox in bounding_boxes:
                points = [(int(bbox[i, 0]), int(bbox[i, 1])) for i in range(8)]
                f.write(str(points)+"\n")
            f.close()
            pedestrian_bbox_record = False

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




# ==============================================================================
# -- VehicleBoundingBoxes ---------------------------------------------------
# ==============================================================================


class VehicleBoundingBoxes(object):
    """
    This is a module responsible for creating 3D bounding boxes and drawing them
    client-side on pygame surface.
    """

    @staticmethod
    def get_bounding_boxes(vehicles, camera):
        """
        Creates 3D bounding boxes based on carla vehicle list and camera.
        """

        bounding_boxes = [VehicleBoundingBoxes.get_bounding_box(vehicle, camera) for vehicle in vehicles]
        # filter objects behind camera
        bounding_boxes = [bb for bb in bounding_boxes if all(bb[:, 2] > 0)]
        return bounding_boxes

    @staticmethod
    def draw_bounding_boxes(bounding_boxes):
        """
        Records bounding boxes data to file.
        """
        global vehicle_bbox_record
        global count

        if vehicle_bbox_record == True:
            f = open("VehicleBBox/bbox"+str(count), 'w')
            print("VehicleBoundingBox")
            for bbox in bounding_boxes:
                points = [(int(bbox[i, 0]), int(bbox[i, 1])) for i in range(8)]
                f.write(str(points)+"\n")
            f.close()
            vehicle_bbox_record = False

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


# ==============================================================================
# -- BasicSynchronousClient ----------------------------------------------------
# ==============================================================================


class BasicSynchronousClient(object):
    """
    Basic implementation of a synchronous client.
    """

    def __init__(self):
        self.client = None
        self.world = None
        self.camera = None
        self.camera_segmentation = None
        self.car = None

        self.display = None
        self.image = None
        self.segmentation_image = None

        self.capture = True
        self.capture_segmentation = True

        self.record = True
        self.seg_record = False
        self.rgb_record = False

        self.screen_capture = 0 
        self.loop_state = True

        self.generate_data = False

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
        Spawns actor-vehicle to be controled.
        """

        car_bp = self.world.get_blueprint_library().filter('vehicle.*')[0]
        location = random.choice(self.world.get_map().get_spawn_points())
        self.car = self.world.spawn_actor(car_bp, location)
        self.car.set_autopilot(True)

    def setup_camera(self):
        """
        Spawns actor-camera to be used to render view.
        Sets calibration for client-side boxes rendering.
        Sets up the spectator to follow the vehicle in the main simulator window.
        """
        # Set up the spectator to follow the vehicle
        self.spectator = self.world.get_spectator()

        seg_transform = carla.Transform(carla.Location(x=1.6, z=1.7), carla.Rotation(pitch=-15))
        self.camera_segmentation = self.world.spawn_actor(self.camera_blueprint('sensor.camera.semantic_segmentation'), seg_transform, attach_to=self.car)
        weak_self = weakref.ref(self)
        self.camera_segmentation.listen(lambda image_seg: weak_self().set_segmentation(weak_self, image_seg))

        #camera_transform = carla.Transform(carla.Location(x=1.5, z=2.8), carla.Rotation(pitch=-15))
        camera_transform = carla.Transform(carla.Location(x=1.6, z=1.7), carla.Rotation(pitch=-15))
        self.camera = self.world.spawn_actor(self.camera_blueprint('sensor.camera.rgb'), camera_transform, attach_to=self.car)
        weak_self = weakref.ref(self)
        self.camera.listen(lambda image: weak_self().set_image(weak_self, image))

        calibration = np.identity(3)
        calibration[0, 2] = VIEW_WIDTH / 2.0
        calibration[1, 2] = VIEW_HEIGHT / 2.0
        calibration[0, 0] = calibration[1, 1] = VIEW_WIDTH / (2.0 * np.tan(VIEW_FOV * np.pi / 360.0))
        self.camera.calibration = calibration
        self.camera_segmentation.calibration = calibration

    def control(self, car):
        """
        Applies control to main car based on pygame pressed keys.
        Will return True If ESCAPE is hit, otherwise False to end main loop.
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
        if keys[K_p]:
            car.set_autopilot(True)       
        if keys[K_c]:
            self.screen_capture = self.screen_capture + 1
        else:
            self.screen_capture = 0

        if keys[K_g]:
            self.generate_data = True

        if keys[K_l]:
            self.loop_state = True
        if keys[K_l] and (pygame.key.get_mods() & pygame.KMOD_SHIFT):
            self.loop_state = False
        control.hand_brake = keys[K_SPACE]

        car.apply_control(control)
        return False

    @staticmethod
    def set_image(weak_self, img):
        """
        Sets image coming from camera sensor.
        The self.capture flag is a mean of synchronization - once the flag is
        set, next coming image will be stored.
        """
        self = weak_self()
        if self.capture:
            self.image = img
            self.capture = False

        if self.rgb_record:
            i = np.array(img.raw_data)
            i2 = i.reshape((VIEW_HEIGHT, VIEW_WIDTH, 4))
            i3 = i2[:, :, :3]
            cv2.imwrite('custom_data/image' + str(self.image_count) + '.png', i3)           
            print("RGB(custom)Image")

    @staticmethod
    def set_segmentation(weak_self, img):
        """
        Sets image coming from camera sensor.
        The self.capture flag is a mean of synchronization - once the flag is
        set, next coming image will be stored.
        """

        self = weak_self()
        if self.capture_segmentation:
            self.segmentation_image = img
            self.capture_segmentation = False


        if self.seg_record:
            img.convert(cc.CityScapesPalette)
            i = np.array(img.raw_data)
            i2 = i.reshape((VIEW_HEIGHT, VIEW_WIDTH, 4))
            i3 = i2[:, :, :3]
            cv2.imwrite('SegmentationImage/seg' + str(self.image_count) +'.png', i3)
            print("SegmentationImage")

    def process_image(self):
        """
        Processes image from camera sensor without displaying it.
        """
        if self.image is not None:
            array = np.frombuffer(self.image.raw_data, dtype=np.dtype("uint8"))
            array = np.reshape(array, (self.image.height, self.image.width, 4))
            array = array[:, :, :3]
            array = array[:, :, ::-1]
            return array
        return None

    def game_loop(self, args):
        """
        Main program loop.
        """

        try:
            pygame.init()

            self.client = carla.Client('127.0.0.1', 2000)
            self.client.set_timeout(2.0)
            self.world = self.client.get_world()

            self.setup_car()
            self.setup_camera()

            # No pygame display is created, we'll use the CARLA spectator instead
            pygame_clock = pygame.time.Clock()

            self.set_synchronous_mode(True)
            vehicles = self.world.get_actors().filter('vehicle.*')
            pedestrians = self.world.get_actors().filter('walker.pedestrian.*')

            self.image_count = 0
            self.time_interval = 0

            global vehicle_bbox_record
            global pedestrian_bbox_record
            global count

            print("Simulation started. Use the main CARLA window to view the vehicle.")
            print("Press ESC to quit, WASD to control the vehicle manually.")

            while True:
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

                # Process image data without displaying
                image_array = self.process_image()

                self.time_interval += 1
                if ((self.time_interval % args.CaptureLoop) == 0 and self.loop_state):
                    self.image_count = self.image_count + 1 
                    self.rgb_record = True
                    self.seg_record = True
                    vehicle_bbox_record = True
                    pedestrian_bbox_record = True
                    count = self.image_count
                    print("-------------------------------------------------")
                    print("ImageCount - %d" %self.image_count)

                if self.screen_capture == 1:
                    self.image_count = self.image_count + 1 
                    self.rgb_record = True
                    self.seg_record = True
                    vehicle_bbox_record = True
                    pedestrian_bbox_record = True
                    count = self.image_count
                    print("-------------------------------------------------")
                    print("Captured! ImageCount - %d" %self.image_count)

                if self.generate_data:
                    print("-------------------------------------------------")
                    print("Generating data...")
                    self.generate_data = False
                    generate_and_process_data()
                    print("Data generation complete!")
                    print("-------------------------------------------------")

                bounding_boxes = VehicleBoundingBoxes.get_bounding_boxes(vehicles, self.camera)
                pedestrian_bounding_boxes = PedestrianBoundingBoxes.get_bounding_boxes(pedestrians, self.camera)

                # Record bounding boxes data to files
                VehicleBoundingBoxes.draw_bounding_boxes(bounding_boxes)
                PedestrianBoundingBoxes.draw_bounding_boxes(pedestrian_bounding_boxes)

                # Process data after bounding boxes are calculated
                if self.rgb_record and self.seg_record:
                    time.sleep(0.1)  # Small delay to ensure data is saved
                    process_single_image(self.image_count)
                    print("Data processing complete!")

                time.sleep(0.03)
                self.rgb_record = False
                self.seg_record = False

                # Handle pygame events for keyboard input
                pygame.event.pump()
                if self.control(self.car):
                    return

        finally:
            self.set_synchronous_mode(False)
            self.camera.destroy()
            self.camera_segmentation.destroy()
            self.car.destroy()
            # No need to call pygame.quit() as we're not using a pygame window

# ==============================================================================
# -- Data Processing Functions -------------------------------------------------
# ==============================================================================

# Brings Images and Bounding Box Information
def reading_data(index):
    global rgb_info, seg_info
    v_data = []
    w_data = []
    k = 0
    w = 0

    rgb_img = cv2.imread('custom_data/image'+ str(index)+ '.png', cv2.IMREAD_COLOR)
    seg_img = cv2.imread('SegmentationImage/seg'+ str(index)+ '.png', cv2.IMREAD_COLOR)

    if str(rgb_img) != "None" and str(seg_img) != "None":
        # Vehicle
        with open('VehicleBBox/bbox'+ str(index), 'r') as fin:
            v_bounding_box_rawdata = fin.read()

        v_bounding_box_data = re.findall(r"-?\d+", v_bounding_box_rawdata)
        v_line_length = len(v_bounding_box_data) // 16 

        v_bbox_data = [[0 for col in range(8)] for row in range(v_line_length)] 

        for i in range(len(v_bounding_box_data)//2):
            j = i*2
            v_data.append(tuple((int(v_bounding_box_data[j]), int(v_bounding_box_data[j+1]))))

        for i in range(len(v_bounding_box_data)//16):
            for j in range(8):
                v_bbox_data[i][j] = v_data[k]
                k += 1

        # Walker (Pedestrian)
        with open('PedestrianBBox/bbox'+ str(index), 'r') as w_fin:
            w_bounding_box_rawdata = w_fin.read()

        w_bounding_box_data = re.findall(r"-?\d+", w_bounding_box_rawdata)
        w_line_length = len(w_bounding_box_data) // 16 

        w_bb_data = [[0 for col in range(8)] for row in range(w_line_length)] 

        for i in range(len(w_bounding_box_data)//2):
            j = i*2
            w_data.append(tuple((int(w_bounding_box_data[j]), int(w_bounding_box_data[j+1]))))

        for i in range(len(w_bounding_box_data)//16):
            for j in range(8):
                w_bb_data[i][j] = w_data[w]
                w += 1

        origin_rgb_info = rgb_img
        rgb_info = rgb_img
        seg_info = seg_img
        return v_bbox_data, v_line_length, w_bb_data, w_line_length 

    else:
        return False

# Converts 8 Vertices to 4 Vertices
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

# Gets Object's Bounding Box Area
def object_area(data):
    global area_info
    area_info = np.zeros(shape=[VIEW_HEIGHT, VIEW_WIDTH, 3], dtype=np.uint8)

    for vehicle_area in data:
        array_x = []
        array_y = []
        for i in range(4):
           array_x.append(vehicle_area[i][0])
        for j in range(4):
           array_y.append(vehicle_area[j][1])

        for i in range(4):
            if array_x[i] <= 0:
                array_x[i] = 1
            elif array_x[i] >= VIEW_WIDTH:
                array_x[i] = VIEW_WIDTH -1
        for j in range(4):
            if array_y[j] <= 0:
                array_y[j] = 1
            elif array_y[j] >= VIEW_HEIGHT:
                array_y[j] = VIEW_HEIGHT -1

        min_x = min(array_x) 
        max_x = max(array_x) 
        min_y = min(array_y) 
        max_y = max(array_y) 
        array = [min_x, max_x, min_y, max_y]
        if filtering(array, Vehicle_COLOR): 
            cv2.rectangle(area_info, (min_x, min_y), (max_x, max_y), Vehicle_COLOR, -1)

# Fits Bounding Box to the Object
def fitting_x(x1, x2, range_min, range_max, color):
    global seg_info
    state = False
    cali_point = 0
    if (x1 < x2):
        for search_point in range(x1, x2):
            for range_of_points in range(range_min, range_max):
                if seg_info[range_of_points, search_point][0] == color[0]:
                    cali_point = search_point
                    state = True
                    break
            if state == True:
                break

    else:
        for search_point in range(x1, x2, -1):
            for range_of_points in range(range_min, range_max):
                if seg_info[range_of_points, search_point][0] == color[0]:
                    cali_point = search_point
                    state = True
                    break
            if state == True:
                break

    return cali_point

def fitting_y(y1, y2, range_min, range_max, color):
    global seg_info
    state = False
    cali_point = 0
    if (y1 < y2):
        for search_point in range(y1, y2):
            for range_of_points in range(range_min, range_max):
                if seg_info[search_point, range_of_points][0] == color[0]:
                    cali_point = search_point
                    state = True
                    break
            if state == True:
                break

    else:
        for search_point in range(y1, y2, -1):
            for range_of_points in range(range_min, range_max):
                if seg_info[search_point, range_of_points][0] == color[0]:
                    cali_point = search_point
                    state = True
                    break
            if state == True:
                break

    return cali_point

# Removes small objects that obstruct to learning
def small_objects_excluded(array, bb_min):
    diff_x = array[1]- array[0]
    diff_y = array[3] - array[2]
    if (diff_x > bb_min and diff_y > bb_min):
        return True
    return False

# Filters occluded objects
def post_occluded_objects_excluded(array, color):
    global seg_info
    top_left = seg_info[array[2]+1, array[0]+1][0]
    top_right = seg_info[array[2]+1, array[1]-1][0] 
    bottom_left = seg_info[array[3]-1, array[0]+1][0] 
    bottom_right = seg_info[array[3]-1, array[1]-1][0]
    if top_left == color[0] and top_right == color[0] and bottom_left == color[0] and bottom_right == color[0]:
        return False

    return True

def pre_occluded_objects_excluded(array, area_image, color):
    top_left = area_image[array[2]-1, array[0]-1][0]
    top_right = area_image[array[2], array[1]+1][0] 
    bottom_left = area_image[array[3]+1, array[1]-1][0] 
    bottom_right = area_image[array[3]+1, array[0]+1][0]
    if top_left == color[0] and top_right == color[0] and bottom_left == color[0] and bottom_right == color[0]:
        return False

    return True

# Filters objects not in the scene
def filtering(array, color):
    global seg_info
    for y in range(array[2], array[3]):
        for x in range(array[0], array[1]):
            if seg_info[y, x][0] == color[0]:
                return True
    return False

# Processes Post-Processing
def processing(img, v_data, w_data, index):
    global seg_info, area_info
    vehicle_class = 0
    walker_class = 1

    object_area(v_data)
    f = open("custom_data/image"+str(index) + ".txt", 'w')

    # Vehicle
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

        if filtering(v_bb_array, Vehicle_COLOR) and pre_occluded_objects_excluded(v_bb_array, area_info, Vehicle_COLOR): 
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

                f.write(str(vehicle_class) + ' ' + str("%0.6f" % darknet_x) + ' ' + str("%0.6f" % darknet_y) + ' ' + 
                str("%0.6f" % darknet_width) + ' ' + str("%0.6f" % darknet_height) + "\n")

                cv2.line(img, (cali_min_x, cali_min_y), (cali_max_x, cali_min_y), VBB_COLOR, 2)
                cv2.line(img, (cali_max_x, cali_min_y), (cali_max_x, cali_max_y), VBB_COLOR, 2)
                cv2.line(img, (cali_max_x, cali_max_y), (cali_min_x, cali_max_y), VBB_COLOR, 2)
                cv2.line(img, (cali_min_x, cali_max_y), (cali_min_x, cali_min_y), VBB_COLOR, 2)

    # Walker (Pedestrian)
    object_area(w_data)

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
        if filtering(w_bb_array, Walker_COLOR) and pre_occluded_objects_excluded(w_bb_array, area_info, Walker_COLOR): 
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

                f.write(str(walker_class) + ' ' + str("%0.6f" % darknet_x) + ' ' + str("%0.6f" % darknet_y) + ' ' + 
                str("%0.6f" % darknet_width) + ' ' + str("%0.6f" % darknet_height) + "\n")

                cv2.line(img, (cali_min_x, cali_min_y), (cali_max_x, cali_min_y), WBB_COLOR, 2)
                cv2.line(img, (cali_max_x, cali_min_y), (cali_max_x, cali_max_y), WBB_COLOR, 2)
                cv2.line(img, (cali_max_x, cali_max_y), (cali_min_x, cali_max_y), WBB_COLOR, 2)
                cv2.line(img, (cali_min_x, cali_max_y), (cali_min_x, cali_min_y), WBB_COLOR, 2)

    f.close()
    cv2.imwrite('draw_bounding_box/image'+str(index)+'.png', img)

def process_single_image(image_index):
    """
    Process a single image with the given index
    """
    global rgb_info

    # Read data only once
    data = reading_data(image_index)
    if data != False:
        v_four_points = converting(data[0], data[1])
        w_four_points = converting(data[2], data[3])
        processing(rgb_info, v_four_points, w_four_points, image_index)

        # Update train.txt file to include this image
        with open("my_data/train.txt", 'a') as train:
            train.write(str('custom_data/image'+str(image_index) + '.png') + "\n")

        print(f"Processed image {image_index}")
        return True
    return False

def generate_and_process_data():
    """
    Process all available images in the dataset
    """
    global rgb_info
    global index_count

    # Get the number of files in VehicleBBox directory
    dataEA = len(next(os.walk('VehicleBBox/'))[2])

    # Clear the train.txt file
    train = open("my_data/train.txt", 'w')
    train.close()

    # Reset index count
    index_count = 0

    for i in range(dataEA + 1):
        if process_single_image(i):
            index_count = index_count + 1

    print(f"Processed {index_count} images")

# ==============================================================================
# -- main() --------------------------------------------------------------------
# ==============================================================================

def main():
    """
    Initializes the client-side bounding box demo with data processing capabilities.
    """
    argparser = argparse.ArgumentParser(
        description=__doc__)
    argparser.add_argument(
        '-l', '--CaptureLoop',
        metavar='N',
        default=10,
        type=int,
        help='set Capture Cycle settings, Recommand : above 100')
    argparser.add_argument(
        '-m', '--mode',
        metavar='MODE',
        default='extract',
        choices=['extract', 'generate', 'both'],
        help='set operation mode: extract, generate, or both')

    args = argparser.parse_args()

    print(__doc__)

    try:
        if args.mode == 'extract' or args.mode == 'both':
            client = BasicSynchronousClient()
            client.game_loop(args)

        if args.mode == 'generate':
            generate_and_process_data()

    finally:
        print('EXIT')


if __name__ == '__main__':
    main()
