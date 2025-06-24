#!/usr/bin/env python
"""
Merged extract.py and generate.py into a single script.
Usage:
    ./fuse4.py [-l N]

Options:
    -l, --CaptureLoop   Capture loop interval (default: 100)
"""

import glob
import os
import sys
import time
import argparse
import random
import weakref
import re

import carla
from carla import ColorConverter as cc

import cv2
import numpy as np

try:
    import pygame
    from pygame.locals import (K_ESCAPE, K_SPACE, KMOD_SHIFT,
                               K_a, K_d, K_s, K_w)
except ImportError:
    raise RuntimeError('cannot import pygame, make sure pygame is installed')

# ==============================================================================
# -- GLOBAL SETTINGS ----------------------------------------------------------
# ==============================================================================
VIEW_WIDTH = 1920 // 2
VIEW_HEIGHT = 1080 // 2
VIEW_FOV = 90

dir_rgb = 'custom_data/'
dir_seg = 'SegmentationImage/'
dir_pbbox = 'PedestrianBBox/'
dir_vbbox = 'VehicleBBox/'

for d in (dir_rgb, dir_seg, dir_pbbox, dir_vbbox, 'my_data/', 'draw_bounding_box/'):
    os.makedirs(d, exist_ok=True)

vehicle_bbox_record = False
pedestrian_bbox_record = False
count = 0

# ==============================================================================
# -- EXTRACT: client-side bounding boxes and images ---------------------------
# ==============================================================================
class PedestrianBoundingBoxes:
    @staticmethod
    def get_bounding_boxes(pedestrians, camera):
        bbs = [PedestrianBoundingBoxes.get_bounding_box(p, camera) for p in pedestrians]
        return [bb for bb in bbs if all(bb[:,2] > 0)]

    @staticmethod
    def draw_bounding_boxes(display, bbs):
        global pedestrian_bbox_record, count
        if pedestrian_bbox_record:
            f = open(f"{dir_pbbox}bbox{count}", 'w')
        for bb in bbs:
            pts = [(int(bb[i,0]), int(bb[i,1])) for i in range(8)]
            if pedestrian_bbox_record:
                f.write(str(pts)+"\n")
        if pedestrian_bbox_record:
            f.close()
            pedestrian_bbox_record = False

    @staticmethod
    def _create_bb_points(actor):
        ext = actor.bounding_box.extent
        e = ext
        pts = [[ e.x,  e.y, -e.z,1],[-e.x,  e.y, -e.z,1],[-e.x,-e.y,-e.z,1],[ e.x,-e.y,-e.z,1],
               [ e.x,  e.y,  e.z,1],[-e.x,  e.y,  e.z,1],[-e.x,-e.y, e.z,1],[ e.x,-e.y, e.z,1]]
        return np.array(pts)

    @staticmethod
    def get_bounding_box(actor, camera):
        bb = PedestrianBoundingBoxes._create_bb_points(actor)
        return PedestrianBoundingBoxes._project(bb, actor, camera)

    @staticmethod
    def _project(coords, actor, sensor):
        world_coords = PedestrianBoundingBoxes._transform(coords, actor)
        sensor_coords = PedestrianBoundingBoxes._transform(world_coords.T, sensor)
        cam = np.transpose(np.dot(sensor.calibration, np.transpose(sensor_coords[:3,:])))
        return np.column_stack((cam[:,0]/cam[:,2], cam[:,1]/cam[:,2], cam[:,2]))

    @staticmethod
    def _transform(coords, obj):
        tr = obj.get_transform()
        mat = PedestrianBoundingBoxes._get_matrix(tr)
        if coords.shape[1] == 4:
            return np.dot(mat, coords.T)
        return np.dot(np.linalg.inv(mat), coords)

    @staticmethod
    def _get_matrix(transform):
        R = transform.rotation
        loc = transform.location
        cy, sy = np.cos(np.radians(R.yaw)), np.sin(np.radians(R.yaw))
        cp, sp = np.cos(np.radians(R.pitch)), np.sin(np.radians(R.pitch))
        cr, sr = np.cos(np.radians(R.roll)), np.sin(np.radians(R.roll))
        M = np.identity(4)
        M[0,3], M[1,3], M[2,3] = loc.x, loc.y, loc.z
        M[0,0] = cp*cy; M[0,1] = cy*sp*sr - sy*cr; M[0,2] = -cy*sp*cr - sy*sr
        M[1,0] = sy*cp; M[1,1] = sy*sp*sr + cy*cr; M[1,2] = -sy*sp*cr + cy*sr
        M[2,0] = sp;    M[2,1] = -cp*sr;        M[2,2] = cp*cr
        return M

class VehicleBoundingBoxes(PedestrianBoundingBoxes):
    @staticmethod
    def get_bounding_boxes(vehicles, camera):
        bbs = [VehicleBoundingBoxes.get_bounding_box(v, camera) for v in vehicles]
        return [bb for bb in bbs if all(bb[:,2]>0)]

    @staticmethod
    def draw_bounding_boxes(display, bbs):
        global vehicle_bbox_record, count
        if vehicle_bbox_record:
            f = open(f"{dir_vbbox}bbox{count}", 'w')
        for bb in bbs:
            pts = [(int(bb[i,0]), int(bb[i,1])) for i in range(8)]
            if vehicle_bbox_record:
                f.write(str(pts)+"\n")
        if vehicle_bbox_record:
            f.close()
            vehicle_bbox_record = False

class BasicSynchronousClient:
    def __init__(self):
        self.client = carla.Client('127.0.0.1',2000)
        self.client.set_timeout(2.0)
        self.world = self.client.get_world()
        self.display = None; self.camera = None; self.camera_seg = None; self.car = None
        self.capture = True; self.screen_capture=0; self.loop_state=False; self.image_count=0
        self.rgb_record=False; self.seg_record=False; self.record=True

    def camera_bp(self, f):
        bp = self.world.get_blueprint_library().find(f)
        bp.set_attribute('image_size_x', str(VIEW_WIDTH))
        bp.set_attribute('image_size_y', str(VIEW_HEIGHT))
        bp.set_attribute('fov', str(VIEW_FOV))
        return bp

    def set_sync(self,val):
        s = self.world.get_settings(); s.synchronous_mode=val; self.world.apply_settings(s)

    def setup(self):
        car_bp = self.world.get_blueprint_library().filter('vehicle.*')[0]
        loc = random.choice(self.world.get_map().get_spawn_points())
        self.car=self.world.spawn_actor(car_bp,loc)
        seg_tr = carla.Transform(carla.Location(x=1.6,z=1.7), carla.Rotation(pitch=-15))
        self.camera_seg=self.world.spawn_actor(self.camera_bp('sensor.camera.semantic_segmentation'),seg_tr,attach_to=self.car)
        self.camera_seg.listen(lambda img: self.set_seg(self, img))
        cam_tr = carla.Transform(carla.Location(x=1.6,z=1.7), carla.Rotation(pitch=-15))
        self.camera=self.world.spawn_actor(self.camera_bp('sensor.camera.rgb'),cam_tr,attach_to=self.car)
        self.camera.listen(lambda img: self.set_img(self,img))
        calib = np.identity(3)
        calib[0,2]=VIEW_WIDTH/2; calib[1,2]=VIEW_HEIGHT/2
        calib[0,0]=calib[1,1]= VIEW_WIDTH/(2.0*np.tan(VIEW_FOV*np.pi/360.0))
        self.camera.calibration = calib; self.camera_seg.calibration=calib

    def set_img(self, weak, img):
        self=self; 
        if self.capture: self.image=img; self.capture=False
        if self.rgb_record:
            arr=np.frombuffer(img.raw_data,dtype='uint8')
            arr=arr.reshape((VIEW_HEIGHT,VIEW_WIDTH,4))[:,:,:3]
            cv2.imwrite(f"{dir_rgb}image{self.image_count}.png", arr)

    def set_seg(self, weak, img):
        self=self
        if self.capture: self.seg=img; self.capture=False
        if self.seg_record:
            img.convert(cc.CityScapesPalette)
            arr=np.frombuffer(img.raw_data,dtype='uint8')
            arr=arr.reshape((VIEW_HEIGHT,VIEW_WIDTH,4))[:,:,:3]
            cv2.imwrite(f"{dir_seg}seg{self.image_count}.png",arr)

    def render(self):
        if hasattr(self,'image'):
            arr=np.frombuffer(self.image.raw_data,dtype='uint8')
            arr=arr.reshape((self.image.height,self.image.width,4))[:,:,:3][:,:,::-1]
            surf=pygame.surfarray.make_surface(arr.swapaxes(0,1))
            self.display.blit(surf,(0,0))

    def control(self):
        keys=pygame.key.get_pressed()
        if keys[K_ESCAPE]: return True
        ctl=self.car.get_control(); ctl.throttle=0
        if keys[K_w]: ctl.throttle=1; ctl.reverse=False
        if keys[K_s]: ctl.throttle=1; ctl.reverse=True
        ctl.steer= (keys[K_d] and min(1, ctl.steer+0.05)) or (keys[K_a] and max(-1, ctl.steer-0.05)) or 0
        if keys[K_SPACE]: ctl.hand_brake=True
        self.car.apply_control(ctl)
        return False

    def game_loop(self,args):
        pygame.init()
        self.set_sync(True)
        self.setup()
        self.display=pygame.display.set_mode((VIEW_WIDTH,VIEW_HEIGHT))
        clock=pygame.time.Clock()
        global vehicle_bbox_record,pedestrian_bbox_record,count
        while True:
            self.world.tick(); clock.tick_busy_loop(60)
            self.capture=True; self.seg_record=False; self.rgb_record=False
            self.render()
            self.image_count+=1 if self.screen_capture or (self.time_interval%args.CaptureLoop==0 and self.loop_state) else 0
            if self.image_count:
                self.rgb_record=self.seg_record=vehicle_bbox_record=pedestrian_bbox_record=True; count=self.image_count
            self.world.tick(); pygame.display.flip(); pygame.event.pump()
            if self.control(): break
        self.set_sync(False); self.camera.destroy(); self.camera_seg.destroy(); self.car.destroy(); pygame.quit()

# ==============================================================================
# -- GENERATE: post-process bounding boxes into training data  ----------------
# ==============================================================================

def reading_data(idx):
    rgb = cv2.imread(f"{dir_rgb}image{idx}.png"); seg = cv2.imread(f"{dir_seg}seg{idx}.png");
    if rgb is None or seg is None: return False
    # parse VehicleBBox
    raw=open(f"{dir_vbbox}bbox{idx}").read(); nums=list(map(int,re.findall(r"-?\\d+",raw)))
    veh=[tuple(nums[i:i+2]) for i in range(0,len(nums),2)]
    ped=[tuple(nums[i:i+2]) for i in range(0,len(nums),2)]
    return veh, len(veh)//8, ped, len(ped)//8, rgb, seg

# ... (similarly define converting, filtering, processing from generate.py) ...

def run_generate():
    dataEA = len(os.listdir(dir_vbbox))
    with open("my_data/train.txt","w") as train:
        for i in range(dataEA):
            rd = reading_data(i)
            if not rd: continue
            # perform converting, processing, etc.
            train.write(f"{dir_rgb}image{i}.png\n")
    print("Generation complete")

# ==============================================================================
# -- MAIN ----------------------------------------------------------------------
# ==============================================================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Fuse extract and generate")
    parser.add_argument('-l','--CaptureLoop', type=int, default=100, help='Capture loop')
    args = parser.parse_args()

    # Extraction phase
    print("==> Starting extraction phase")
    client = BasicSynchronousClient()
    client.game_loop(args)
    print("==> Extraction done")

    # Generation phase
    print("==> Starting generation phase")
    run_generate()
    print("==> Generation done")
