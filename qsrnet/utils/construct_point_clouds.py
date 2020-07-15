from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import time
import copy
import open3d as o3d
from scipy import ndimage

def construct_point_cloud_dict(mask_results, color_image, depth_image, configuration):
    args = configuration['ARGUMENTS']; depth_camera_info = configuration['CAMERA INTRINSIC']['DEPTH CAMERA']
    ids = mask_results['class_ids']; pcd_dict = {}
    masks_with_no_overlap = handle_mask_overlap(len(ids), mask_results['masks'], color_image, depth_image, configuration)
    fx = depth_camera_info['K'][0]; cx = depth_camera_info['K'][2]; fy = depth_camera_info['K'][4]; cy = depth_camera_info['K'][5]
    for j in range(len(ids)):
        obj_id = ids[j]
        roi = mask_results['rois'][j]
        reduced_mask = ~ ndimage.binary_dilation(~ masks_with_no_overlap[j, :, :], iterations = args['inner_point_cloud_iteration'])
        reduced_mask_int = reduced_mask.astype(np.uint16)
        if (configuration['NAMES']['class_names'][obj_id] not in configuration['NAMES']['body_part_names']) or (np.sum(reduced_mask_int) >= args['min_num_pixel_to_make_pcd']): # right hand index
            camera_intrinsic_for_mask = o3d.camera.PinholeCameraIntrinsic()
            camera_intrinsic_for_mask.set_intrinsics(roi[3]-roi[1], roi[2]-roi[0], fx, fy, cx-roi[1], cy-roi[0])
            cropped_depth_image = crop_image(reduced_mask_int, depth_image, roi)
            depth_image_for_mask = o3d.geometry.Image(copy.copy(cropped_depth_image))
            # depth_image_for_mask = o3d.geometry.Image(copy.copy(depth_image[0:100, 0:100]))
            pcd_for_mask = o3d.geometry.PointCloud.create_from_depth_image(depth_image_for_mask, camera_intrinsic_for_mask, \
                                                                           depth_scale = 1, depth_trunc = 10000, stride = args['point_cloud_stride'])
        else:
            pcd_for_mask = o3d.geometry.PointCloud()

        if args['point_cloud_filter_mode'] == 'no':
            cl = pcd_for_mask
        elif args['point_cloud_filter_mode'] == 'radius':
            cl, ind = pcd_for_mask.remove_radius_outlier(nb_points=args['nb_points'], radius=args['radius'])
        elif args['point_cloud_filter_mode'] == 'statistical':
            cl, ind = pcd_for_mask.remove_statistical_outlier(nb_neighbors=args['nb_neighbors'], std_ratio=args['std_ratio'])
        else:
            filtered_cl, ind = pcd_for_mask.remove_radius_outlier(nb_points=args['nb_points'], radius=args['radius'])
            cl, ind = filtered_cl.remove_statistical_outlier(nb_neighbors=args['nb_neighbors'], std_ratio=args['std_ratio'])
        pcd_dict[obj_id] = cl
    return pcd_dict

def crop_image(reduced_mask_int, depth_image, roi):
    original_mask_image = reduced_mask_int * depth_image
    cropped_depth_image = original_mask_image[roi[0]:roi[2], roi[1]:roi[3]]
    return cropped_depth_image

def handle_mask_overlap(num_objects, masks, color_image, depth_image, configuration):
    args = configuration['ARGUMENTS']
    skip_ind = args['mask_overlap_skip_ind']
    for i in range(num_objects):
        for j in range(i+1, num_objects):
            overlapped_pixels_no_skip = masks[i, :, :] & masks[j, :, :]
            overlapped_pixels = masks[i, ::skip_ind, ::skip_ind] & masks[j, ::skip_ind, ::skip_ind]
            overlapped_pixels_int = overlapped_pixels.astype(int)
            if np.sum(overlapped_pixels_int)>0:
                overlap_sum = np.sum(overlapped_pixels_int)
                overlapped_pixels_mean_rgb = np.array([np.sum(overlapped_pixels_int*color_image[::skip_ind, ::skip_ind, 0]) / overlap_sum, \
                                                       np.sum(overlapped_pixels_int*color_image[::skip_ind, ::skip_ind, 1]) / overlap_sum, \
                                                       np.sum(overlapped_pixels_int*color_image[::skip_ind, ::skip_ind, 2]) / overlap_sum])

                dilated_overlaps = ndimage.binary_dilation(overlapped_pixels, iterations = args['overlap_dilation_iteration'])
                mask_for_i = dilated_overlaps * masks[i, ::skip_ind, ::skip_ind]
                mask_for_j = dilated_overlaps * masks[j, ::skip_ind, ::skip_ind]
                # mask_for_i = dilated_overlaps * masks[i, ::skip_ind, ::skip_ind] * (~overlapped_pixels)
                # mask_for_j = dilated_overlaps * masks[j, ::skip_ind, ::skip_ind] * (~overlapped_pixels)
                mask_for_i_int = mask_for_i.astype(int); mask_for_j_int = mask_for_j.astype(int)
                mask_for_i_sum = np.sum(mask_for_i_int); mask_for_j_sum = np.sum(mask_for_j_int)
                mask_for_i_mean_rgb = np.array([np.sum(mask_for_i_int*color_image[::skip_ind, ::skip_ind, 0]) / mask_for_i_sum, \
                                                np.sum(mask_for_i_int*color_image[::skip_ind, ::skip_ind, 1]) / mask_for_i_sum, \
                                                np.sum(mask_for_i_int*color_image[::skip_ind, ::skip_ind, 2]) / mask_for_i_sum])
                mask_for_j_mean_rgb = np.array([np.sum(mask_for_j_int*color_image[::skip_ind, ::skip_ind, 0]) / mask_for_j_sum, \
                                                np.sum(mask_for_j_int*color_image[::skip_ind, ::skip_ind, 1]) / mask_for_j_sum, \
                                                np.sum(mask_for_j_int*color_image[::skip_ind, ::skip_ind, 2]) / mask_for_j_sum])

                if np.linalg.norm(overlapped_pixels_mean_rgb - mask_for_i_mean_rgb) <= np.linalg.norm(overlapped_pixels_mean_rgb - mask_for_j_mean_rgb):
                    masks[j, :, :] = masks[j, :, :] * (~overlapped_pixels_no_skip)
                else:
                    masks[i, :, :] = masks[i, :, :] * (~overlapped_pixels_no_skip)
    return masks
