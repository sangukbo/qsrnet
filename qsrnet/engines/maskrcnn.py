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
import multiprocessing as mp
from multiprocessing import Pool
import pyrealsense2 as rs
from scipy import ndimage
import json
import copy

from detectron2.config import get_cfg
from detectron2.engine.defaults import DefaultPredictor

class maskrcnn_inference_engine:
    def __init__(self, configuration):
        self.configuration = configuration
        self.cfg = self.setup_cfg()
        self.maskrcnn_predictor = DefaultPredictor(self.cfg)

    def setup_cfg(self):
        confidence_threshold = 0.5
        cfg = get_cfg()
        cfg.merge_from_file(self.configuration['DIRECTORIES']['detectron_dir'] + '/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml')
        cfg.merge_from_list(['MODEL.WEIGHTS', 'detectron2://COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl'])
        # cfg.merge_from_list(['MODEL.WEIGHTS', self.configuration['DIRECTORIES']['qsrnet_dir'] + '/weights/maskrcnn_weight.pkl'])
        cfg.MODEL.RETINANET.SCORE_THRESH_TEST = confidence_threshold
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = confidence_threshold
        cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = confidence_threshold
        cfg.MODEL.DEVICE = 'cuda:0'
        cfg.freeze()
        return cfg

    def compute_masks(self, color_image):
        results = {}
        predictions = self.maskrcnn_predictor(color_image)
        results['class_ids'] = predictions['instances'].pred_classes.cpu().numpy()
        results['masks'] = predictions['instances'].pred_masks.cpu().numpy()
        results['rois'] = self.rois_flip_height_and_width(predictions['instances'].pred_boxes.tensor.int().cpu().numpy())
        return results

    def rois_flip_height_and_width(self, rois_original):
        rois_flipped = copy.copy(rois_original); num_instances = rois_original.shape[0]
        for instance_ind in range(num_instances):
            rois_flipped[instance_ind, :] = np.array([rois_original[instance_ind, 1], rois_original[instance_ind, 0], rois_original[instance_ind, 3], rois_original[instance_ind, 2]])
        return rois_flipped
