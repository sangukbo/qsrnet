from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import time

def compute_metrics_from_point_clouds(pcd_dict, object_ids, pair_ids, timestamp, previous_center_dict):
    pcd_distance_dict = pcd_distances(pcd_dict, object_ids, pair_ids)
    center_dict = compute_centers(pcd_dict, object_ids)
    center_distance_dict = compute_center_distances(center_dict, object_ids, pair_ids)
    velocity1_dict, velocity2_dict, velocity3_dict, timestamp = compute_velocity(center_dict, previous_center_dict, object_ids, pair_ids, timestamp)
    metric_dict = {'pcd_distances': pcd_distance_dict, 'center_distances': center_distance_dict, \
                   'velocity1': velocity1_dict, 'velocity2': velocity2_dict, 'velocity3': velocity3_dict}
    return metric_dict, center_dict, timestamp

def pcd_distances(pcd_dict, object_ids, pair_ids):
    pcd_distance_dict = {}
    for m in range(len(object_ids)):
        for n in range(len(object_ids)):
            obj_id_m = object_ids[m]; obj_id_n = object_ids[n]
            pcd_distance_id = '(' + str(obj_id_m) + ',' + str(obj_id_n) + ')'
            if pcd_distance_id in pair_ids:
                if (not obj_id_m in pcd_dict) or (not obj_id_n in pcd_dict):
                    pcd_distance = 'empty'
                else:
                    if len(pcd_dict[obj_id_m].points) == 0 or len(pcd_dict[obj_id_n].points) == 0:
                        pcd_distance = 'empty' # pcd_distance = -1.0
                    else:
                        pcd_distance = np.asarray(pcd_dict[obj_id_m].compute_point_cloud_distance(pcd_dict[obj_id_n])).min()
                pcd_distance_dict[pcd_distance_id] = pcd_distance
    return pcd_distance_dict

def compute_centers(pcd_dict, object_ids):
    center_dict = {}
    for m in range(len(object_ids)):
        obj_id_m = object_ids[m]
        if (obj_id_m in pcd_dict):
            if len(pcd_dict[obj_id_m].points) > 0:
                mean, cov = pcd_dict[obj_id_m].compute_mean_and_covariance()
                center_dict[obj_id_m] = mean
    return center_dict

def compute_center_distances(center_dict, object_ids, pair_ids):
    center_distance_dict = {}
    for m in range(len(object_ids)):
        for n in range(len(object_ids)):
            obj_id_m = object_ids[m]; obj_id_n = object_ids[n]
            center_distance_id = '(' + str(obj_id_m) + ',' + str(obj_id_n) + ')'
            if center_distance_id in pair_ids:
                if (not obj_id_m in center_dict) or (not obj_id_n in center_dict):
                    center_distance = 'empty' # center_distance = -1.0
                else:
                    center_distance = np.linalg.norm(center_dict[obj_id_m] - center_dict[obj_id_n])
                center_distance_dict[center_distance_id] = center_distance
    return center_distance_dict

def compute_velocity(center_dict, previous_center_dict, object_ids, pair_ids, timestamp):
    if len(previous_center_dict)==0:
        velocity1_dict = {}; velocity2_dict = {}; velocity3_dict = {}; next_timestamp = time.time()
    else:
        velocity1_dict = {}; velocity2_dict = {}; velocity3_dict = {}
        time_step = time.time() - timestamp; next_timestamp = timestamp + time_step
        # print("function")
        # print(time_step)
        for m in range(len(object_ids)):
            for n in range(len(object_ids)):
                obj_id_m = object_ids[m]; obj_id_n = object_ids[n]
                velocity_id = '(' + str(obj_id_m) + ',' + str(obj_id_n) + ')'
                if velocity_id in pair_ids:

                    if (obj_id_m in previous_center_dict) and (obj_id_m in center_dict) and \
                       (obj_id_n in previous_center_dict) and (obj_id_n in center_dict):

                        # fix obj[n]
                        dist1_pre = np.linalg.norm(previous_center_dict[obj_id_m] - center_dict[obj_id_n])
                        dist1 = np.linalg.norm(center_dict[obj_id_m] - center_dict[obj_id_n])
                        velocity1_dict[velocity_id] = (dist1 - dist1_pre)/time_step

                        # fix obj[m]
                        dist2_pre = np.linalg.norm(center_dict[obj_id_m] - previous_center_dict[obj_id_n])
                        dist2 = np.linalg.norm(center_dict[obj_id_m] - center_dict[obj_id_n])
                        velocity2_dict[velocity_id] = (dist2 - dist2_pre)/time_step

                        dist3_pre = np.linalg.norm(previous_center_dict[obj_id_m] - previous_center_dict[obj_id_n])
                        dist3 = np.linalg.norm(center_dict[obj_id_m] - center_dict[obj_id_n])
                        velocity3_dict[velocity_id] = (dist3 - dist3_pre)/time_step

                    # obj[n] pre not detected
                    elif (obj_id_m in previous_center_dict) and (obj_id_m in center_dict) and (obj_id_n in center_dict):

                        # fix obj[n]
                        dist1_pre = np.linalg.norm(previous_center_dict[obj_id_m] - center_dict[obj_id_n])
                        dist1 = np.linalg.norm(center_dict[obj_id_m] - center_dict[obj_id_n])
                        velocity1_dict[velocity_id] = (dist1 - dist1_pre)/time_step

                        # fix obj[m]
                        velocity2_dict[velocity_id] = 'empty'

                        velocity3_dict[velocity_id] = 'empty'

                    # obj[m] pre not detected
                    elif (obj_id_m in center_dict) and (obj_id_n in previous_center_dict) and (obj_id_n in center_dict):

                        # fix obj[n]
                        velocity1_dict[velocity_id] = 'empty'

                        # fix obj[m]
                        dist2_pre = np.linalg.norm(center_dict[obj_id_m] - previous_center_dict[obj_id_n])
                        dist2 = np.linalg.norm(center_dict[obj_id_m] - center_dict[obj_id_n])
                        velocity2_dict[velocity_id] = (dist2 - dist2_pre)/time_step

                        velocity3_dict[velocity_id] = 'empty'

                    else:

                        # fix obj[n]
                        velocity1_dict[velocity_id] = 'empty'

                        # fix obj[m]
                        velocity2_dict[velocity_id] = 'empty'

                        velocity3_dict[velocity_id] = 'empty'

    return velocity1_dict, velocity2_dict, velocity3_dict, next_timestamp
