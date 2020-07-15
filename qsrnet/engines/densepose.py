#!/usr/bin/env python3

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
import multiprocessing as mp
from multiprocessing import Pool
import pyrealsense2 as rs
from scipy import ndimage
import json
import copy

from detectron2.config import get_cfg
from detectron2.engine.defaults import DefaultPredictor
from detectron2.structures.boxes import BoxMode

sys.path.append('/home/appuser/detectron2_repo/projects/DensePose/')
from densepose import add_densepose_config

from qsrnet.utils.filters import *

class densepose_inference_engine:
    def __init__(self, configuration):
        self.configuration = configuration
        self.cfg = self.setup_cfg()
        self.densepose_predictor = DefaultPredictor(self.cfg)

    def setup_cfg(self):
        confidence_threshold = 0.8
        cfg = get_cfg()
        add_densepose_config(cfg)
        cfg.merge_from_file(self.configuration['DIRECTORIES']['densepose_dir'] + '/configs/densepose_rcnn_R_50_FPN_s1x.yaml')
        cfg.merge_from_list(['MODEL.WEIGHTS', self.configuration['DIRECTORIES']['qsrnet_dir'] + '/weights/densepose_weight.pkl'])
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = confidence_threshold
        cfg.MODEL.DEVICE = 'cuda:0'
        cfg.freeze()
        return cfg

    def compute_masks(self, color_image, depth_image):
        args = self.configuration['ARGUMENTS']; depth_camera_info = self.configuration['CAMERA INTRINSIC']['DEPTH CAMERA']
        color_image_densepose = copy.copy(color_image)

        with torch.no_grad():
            predictions = self.densepose_predictor(color_image_densepose)
        boxes_XYWH = BoxMode.convert(predictions['instances'].get("pred_boxes").tensor.cpu(), BoxMode.XYXY_ABS, BoxMode.XYWH_ABS)
        result_for_all = predictions['instances'].get("pred_densepose").to_result(boxes_XYWH)
        num_instances = len(predictions['instances']); height = predictions['instances']._image_size[0]; width = predictions['instances']._image_size[1]
        iuv_image = np.zeros((num_instances, height, width))

        for instance in range(num_instances):
            iuv_arr = result_for_all.results[instance]
            box_XYWH = boxes_XYWH[instance, :]
            start_x = math.ceil(box_XYWH[0].item()); start_y = math.ceil(box_XYWH[1].item())
            end_x = start_x + math.floor(box_XYWH[2].item()); end_y = start_y + math.floor(box_XYWH[3].item())
            iuv_image[instance, start_y:end_y, start_x:end_x] = iuv_arr[0, :, :]

        # right hand index
        total_body_part_class_ids = []; total_body_part_masks = []; total_body_part_rois = []
        for body_part in self.configuration['NAMES']['body_part_names']:
            if num_instances>0:
                human_ind = 0; densepose_id = self.body_part_densepose_id(body_part); body_part_mask = iuv_image[human_ind, :, :]==densepose_id
            else:
                body_part_mask = np.full((height, width), False)
            body_part_mask = filter_mask_with_width(body_part_mask, color_image, depth_image, self.configuration); body_part_mask_list = body_part_mask.tolist()
            body_part_class_id = self.configuration['NAMES']['class_names'].index(body_part)
            if np.sum(body_part_mask) >= args['min_num_pixel_to_make_pcd']:
                body_part_roi = np.array([np.min(np.argwhere(np.sum(body_part_mask, axis = 1)>0)), np.min(np.argwhere(np.sum(body_part_mask, axis = 0)>0)), \
                                          np.max(np.argwhere(np.sum(body_part_mask, axis = 1)>0)) + 1, np.max(np.argwhere(np.sum(body_part_mask, axis = 0)>0)) + 1]).tolist()
            else:
                body_part_roi = np.array([0, 0, depth_camera_info['height'], depth_camera_info['width']]).tolist()
            total_body_part_class_ids.append(body_part_class_id); total_body_part_masks.append(body_part_mask_list); total_body_part_rois.append(body_part_roi)
        return np.array(total_body_part_class_ids), np.array(total_body_part_masks), np.array(total_body_part_rois)

    def body_part_densepose_id(self, body_part):
        if body_part == 'righthand': return 3
