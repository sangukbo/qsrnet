import numpy as np
import random
import open3d as o3d
import copy

def get_object_ids(object_names, class_names):
    object_ids = []
    for object_name in object_names:
        object_ids.append(class_names.index(object_name))
    return object_ids

def get_pair_ids(pair_names, class_names):
    pair_ids = []
    for pair_name in pair_names:
        obj1_name, obj2_name = pair_name[1:-1].split(',')
        obj1_id = class_names.index(obj1_name); obj2_id = class_names.index(obj2_name)
        pair_ids.append('(' + str(obj1_id) + ',' + str(obj2_id) + ')')
    return pair_ids

def pair_name_from_id(pair_id, class_names):
    obj1_id, obj2_id = pair_id[1:-1].split(',')
    obj1_name = class_names[int(obj1_id)]; obj2_name = class_names[int(obj2_id)]
    return '(' + obj1_name + ',' + obj2_name + ')'

def image_with_masks(color_image, masks):
    n_objects = masks.shape[0]
    for i in range(n_objects):
        r = random.randint(0, 255); g = random.randint(0, 255); b = random.randint(0, 255)
        color_image[masks[i]] = (b, g, r)
    return color_image

def point_cloud_with_masks(pcd_dict, depth_image, configuration):
    args = configuration['ARGUMENTS']; depth_camera_info = configuration['CAMERA INTRINSIC']['DEPTH CAMERA']
    width = depth_camera_info['width']; height = depth_camera_info['height']
    fx = depth_camera_info['K'][0]; cx = depth_camera_info['K'][2]; fy = depth_camera_info['K'][4]; cy = depth_camera_info['K'][5]
    camera_intrinsic = o3d.camera.PinholeCameraIntrinsic()
    camera_intrinsic.set_intrinsics(width, height, fx, fy, cx, cy)
    depth_image_point_cloud = o3d.geometry.Image(copy.copy(depth_image))
    pcd_with_mask = o3d.geometry.PointCloud.create_from_depth_image(depth_image_point_cloud, camera_intrinsic, \
                                                                    depth_scale = 1, depth_trunc = 10000, stride = args['point_cloud_stride']*2)
    r = random.randint(0, 255)/255; g = random.randint(0, 255)/255; b = random.randint(0, 255)/255
    pcd_with_mask.paint_uniform_color([r, g, b])
    for pcd_id in pcd_dict:
        r = random.randint(0, 255)/255; g = random.randint(0, 255)/255; b = random.randint(0, 255)/255
        pcd = copy.copy(pcd_dict[pcd_id])
        pcd.paint_uniform_color([r, g, b])
        pcd_with_mask = pcd_with_mask + pcd
    return pcd_with_mask

def masks_for_objects_of_interest(mask_results, object_ids):
    # compute new mask_results for objects in object_ids
    final_size = 0
    for i in range(len(mask_results['class_ids'])):
        if mask_results['class_ids'][i] in object_ids:
            final_size = final_size + 1
    new_class_ids = np.empty(final_size, dtype = int)
    new_masks = np.zeros((final_size, mask_results['masks'].shape[1], mask_results['masks'].shape[2]), dtype = bool)
    new_rois = np.zeros((final_size, 4), dtype = int)
    id_iteration = 0
    for i in range(len(mask_results['class_ids'])):
        if mask_results['class_ids'][i] in object_ids:
            new_class_ids[id_iteration] = mask_results['class_ids'][i]
            new_masks[id_iteration, :, :] = mask_results['masks'][i, :, :]
            new_rois[id_iteration, :] = mask_results['rois'][i, :]
            id_iteration = id_iteration + 1
    return {'class_ids': new_class_ids, 'masks': new_masks, 'rois': new_rois}
