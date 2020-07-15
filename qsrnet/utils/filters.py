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

def filter_mask_with_width(mask, color_image, depth_image, configuration):
    args = configuration['ARGUMENTS']
    mask_depth = mask * depth_image
    depth_median = np.median(np.extract(mask_depth > 0, mask_depth))
    if np.isnan(depth_median):
        return mask
    else:
        return (depth_median - args['object_width'] < mask_depth) & (mask_depth < depth_median + args['object_width']) & mask
