#!/usr/bin/python

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from collections import defaultdict
import argparse
import cv2
import glob
import logging
import os
import time
from typing import Any, ClassVar, Dict, List
import torch
import numpy as np
import operator
import pdb, traceback, sys
import csv
import math
import skimage.io
import pickle
import open3d as o3d
import pyrealsense2 as rs
from scipy import ndimage
import json
import socket

from qsrnet.utils.construct_point_clouds import *
from qsrnet.utils.point_cloud_metrics import *
from qsrnet.utils.camera_util import *
from qsrnet.utils.etc import *
from qsrnet.engines.maskrcnn import *
from qsrnet.engines.densepose import *

# socket
IP = ''; PORT = 5050; SIZE = 4*1024; SERVER_ADDR = (IP, PORT)

#####################################################################
# initialize

with open('/home/appuser/qsrnet/configs/config.json') as json_file:
    configuration = json.load(json_file)

object_ids = get_object_ids(configuration['NAMES']['object_names'], configuration['NAMES']['class_names'])
pair_ids = get_pair_ids(configuration['NAMES']['pair_names'], configuration['NAMES']['class_names'])

#######################################

def compute_metrics_main():
    iteration = 0

    realsense = realsense_camera()
    maskrcnn_unit = maskrcnn_inference_engine(configuration)
    densepose_unit = densepose_inference_engine(configuration)

    try:
        previous_center_dict = {}; timestamp = time.time()
        while True:
            time_for_single_iteration = time.time()
            depth_image, color_image = realsense.get_realsense_images()
            # cv2.imwrite('/home/appuser/qsrnet/data_output/image/color/image' + str(iteration) + '.jpg', color_image)
            # cv2.imwrite('/home/appuser/qsrnet/data_output/image/depth/image' + str(iteration) + '.png', depth_image)

            # mask rcnn
            mask_results = maskrcnn_unit.compute_masks(color_image)
            # dense pose
            body_part_class_ids, body_part_masks, body_part_rois = densepose_unit.compute_masks(color_image, depth_image)
            mask_results['class_ids'] = np.append(mask_results['class_ids'], body_part_class_ids, axis = 0)
            mask_results['masks'] = np.append(mask_results['masks'], body_part_masks, axis = 0)
            mask_results['rois'] = np.append(mask_results['rois'], body_part_rois, axis = 0)
            mask_results = masks_for_objects_of_interest(mask_results, object_ids)
            # plot masks
            image_masks = image_with_masks(color_image, mask_results['masks'])
            cv2.imwrite('/home/appuser/qsrnet/data_output/image/mask/image' + str(iteration) + '.jpg', image_masks)

            # find point cloud
            pcd_dict = construct_point_cloud_dict(mask_results, color_image, depth_image, configuration)
            # save point clouds
            # point_cloud_masks = point_cloud_with_masks(pcd_dict, depth_image, configuration)
            # o3d.io.write_point_cloud('/home/appuser/qsrnet/data_output/point_cloud/mask/point_cloud' + str(iteration) + '.pcd', point_cloud_masks)
            # pcd = o3d.io.read_point_cloud('/home/sanguk/Desktop/qsrnet/data_output/point_cloud/mask/point_cloud0.pcd')
            # o3d.visualization.draw_geometries([pcd])

            # compute metrics
            metric_dict, previous_center_dict, timestamp = compute_metrics_from_point_clouds(pcd_dict, object_ids, pair_ids, timestamp, previous_center_dict)

            # socket
            client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            client_socket.connect(SERVER_ADDR)  # connect to server
            metric_msg = pickle.dumps(metric_dict)
            client_socket.send(metric_msg)  # send message to server
            client_socket.close()
            # msg = client_socket.recv(SIZE)  # receive message from server
            # print("resp from server : {}".format(msg))  # print message from server

            iteration += 1
            print(time.time() - time_for_single_iteration)

    finally:
        realsense.pipeline.stop()

#########################

if __name__ == '__main__':
    try:
        compute_metrics_main()
    except:
        extype, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)
