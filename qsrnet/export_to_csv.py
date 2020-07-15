import numpy as np
import pickle
import csv

full_ids = [40, 46, 61, 100]

open('/home/sanguk/Desktop/cv/data/data_output/' + 'timestamp.csv', 'w')

fig_num = 100

with open('/home/sanguk/Desktop/cv/data/data_output/' + 'timestamp.csv', 'a') as f:
        field_name = ['(40,46)_1', '(40,61)_1', '(40,100)_1', '(46,61)_1', '(46,100)_1', '(61,100)_1', \
                      '(40,46)_2', '(40,61)_2', '(40,100)_2', '(46,61)_2', '(46,100)_2', '(61,100)_2', \
                      '(40,46)_3', '(40,61)_3', '(40,100)_3', '(46,61)_3', '(46,100)_3', '(61,100)_3', \
                      '(40,46)_4', '(40,61)_4', '(40,100)_4', '(46,61)_4', '(46,100)_4', '(61,100)_4', \
                      '(40,46)_5', '(40,61)_5', '(40,100)_5', '(46,61)_5', '(46,100)_5', '(61,100)_5']
        writer = csv.DictWriter(f, fieldnames=field_name)
        writer.writerow({'(40,46)_1': 'bottle, bowl', '(40,61)_1': 'bottle table', '(40,100)_1': 'bottle hand', '(46,61)_1': 'bowl table', '(46,100)_1': 'bowl hand', '(61,100)_1': 'table hand', \
        	             '(40,46)_2': 'bottle, bowl', '(40,61)_2': 'bottle table', '(40,100)_2': 'bottle hand', '(46,61)_2': 'bowl table', '(46,100)_2': 'bowl hand', '(61,100)_2': 'table hand', \
        	             '(40,46)_3': 'bottle, bowl', '(40,61)_3': 'bottle table', '(40,100)_3': 'bottle hand', '(46,61)_3': 'bowl table', '(46,100)_3': 'bowl hand', '(61,100)_3': 'table hand', \
        	             '(40,46)_4': 'bottle, bowl', '(40,61)_4': 'bottle table', '(40,100)_4': 'bottle hand', '(46,61)_4': 'bowl table', '(46,100)_4': 'bowl hand', '(61,100)_4': 'table hand', \
        	             '(40,46)_5': 'bottle, bowl', '(40,61)_5': 'bottle table', '(40,100)_5': 'bottle hand', '(46,61)_5': 'bowl table', '(46,100)_5': 'bowl hand', '(61,100)_5': 'table hand'})

for fig_id in range(fig_num):
    file = open('/home/sanguk/Desktop/cv/data/data_output/metrics/' + 'metrics_' + str(fig_id) + '.pickle', 'rb')
    r = pickle.load(file, encoding="bytes")

    if '(40,46)' not in r['pcd_distances']: r['pcd_distances']['(40,46)'] = 'nil'
    if '(40,61)' not in r['pcd_distances']: r['pcd_distances']['(40,61)'] = 'nil'
    if '(40,100)' not in r['pcd_distances']: r['pcd_distances']['(40,100)'] = 'nil'
    if '(46,61)' not in r['pcd_distances']: r['pcd_distances']['(46,61)'] = 'nil'
    if '(46,100)' not in r['pcd_distances']: r['pcd_distances']['(46,100)'] = 'nil'
    if '(61,100)' not in r['pcd_distances']: r['pcd_distances']['(61,100)'] = 'nil'

    if '(40,46)' not in r['center_distances']: r['center_distances']['(40,46)'] = 'nil'
    if '(40,61)' not in r['center_distances']: r['center_distances']['(40,61)'] = 'nil'
    if '(40,100)' not in r['center_distances']: r['center_distances']['(40,100)'] = 'nil'
    if '(46,61)' not in r['center_distances']: r['center_distances']['(46,61)'] = 'nil'
    if '(46,100)' not in r['center_distances']: r['center_distances']['(46,100)'] = 'nil'
    if '(61,100)' not in r['center_distances']: r['center_distances']['(61,100)'] = 'nil'

    if '(40,46)' not in r['velocity1']: r['velocity1']['(40,46)'] = 'nil'
    if '(40,61)' not in r['velocity1']: r['velocity1']['(40,61)'] = 'nil'
    if '(40,100)' not in r['velocity1']: r['velocity1']['(40,100)'] = 'nil'
    if '(46,61)' not in r['velocity1']: r['velocity1']['(46,61)'] = 'nil'
    if '(46,100)' not in r['velocity1']: r['velocity1']['(46,100)'] = 'nil'
    if '(61,100)' not in r['velocity1']: r['velocity1']['(61,100)'] = 'nil'

    if '(40,46)' not in r['velocity2']: r['velocity2']['(40,46)'] = 'nil'
    if '(40,61)' not in r['velocity2']: r['velocity2']['(40,61)'] = 'nil'
    if '(40,100)' not in r['velocity2']: r['velocity2']['(40,100)'] = 'nil'
    if '(46,61)' not in r['velocity2']: r['velocity2']['(46,61)'] = 'nil'
    if '(46,100)' not in r['velocity2']: r['velocity2']['(46,100)'] = 'nil'
    if '(61,100)' not in r['velocity2']: r['velocity2']['(61,100)'] = 'nil'

    if '(40,46)' not in r['velocity3']: r['velocity3']['(40,46)'] = 'nil'
    if '(40,61)' not in r['velocity3']: r['velocity3']['(40,61)'] = 'nil'
    if '(40,100)' not in r['velocity3']: r['velocity3']['(40,100)'] = 'nil'
    if '(46,61)' not in r['velocity3']: r['velocity3']['(46,61)'] = 'nil'
    if '(46,100)' not in r['velocity3']: r['velocity3']['(46,100)'] = 'nil'
    if '(61,100)' not in r['velocity3']: r['velocity3']['(61,100)'] = 'nil'

    with open('/home/sanguk/Desktop/cv/data/data_output/' + 'timestamp.csv', 'a') as f:
        field_name = ['(40,46)_1', '(40,61)_1', '(40,100)_1', '(46,61)_1', '(46,100)_1', '(61,100)_1', \
                      '(40,46)_2', '(40,61)_2', '(40,100)_2', '(46,61)_2', '(46,100)_2', '(61,100)_2', \
                      '(40,46)_3', '(40,61)_3', '(40,100)_3', '(46,61)_3', '(46,100)_3', '(61,100)_3', \
                      '(40,46)_4', '(40,61)_4', '(40,100)_4', '(46,61)_4', '(46,100)_4', '(61,100)_4', \
                      '(40,46)_5', '(40,61)_5', '(40,100)_5', '(46,61)_5', '(46,100)_5', '(61,100)_5']
        writer = csv.DictWriter(f, fieldnames=field_name)
        writer.writerow({'(40,46)_1': str(r['pcd_distances']['(40,46)']), '(40,61)_1': str(r['pcd_distances']['(40,61)']), '(40,100)_1': str(r['pcd_distances']['(40,100)']), '(46,61)_1': str(r['pcd_distances']['(46,61)']), '(46,100)_1': str(r['pcd_distances']['(46,100)']), '(61,100)_1': str(r['pcd_distances']['(61,100)']), \
        	             '(40,46)_2': str(r['center_distances']['(40,46)']), '(40,61)_2': str(r['center_distances']['(40,61)']), '(40,100)_2': str(r['center_distances']['(40,100)']), '(46,61)_2': str(r['center_distances']['(46,61)']), '(46,100)_2': str(r['center_distances']['(46,100)']), '(61,100)_2': str(r['center_distances']['(61,100)']), \
        	             '(40,46)_3': str(r['velocity1']['(40,46)']), '(40,61)_3': str(r['velocity1']['(40,61)']), '(40,100)_3': str(r['velocity1']['(40,100)']), '(46,61)_3': str(r['velocity1']['(46,61)']), '(46,100)_3': str(r['velocity1']['(46,100)']), '(61,100)_3': str(r['velocity1']['(61,100)']), \
        	             '(40,46)_4': str(r['velocity2']['(40,46)']), '(40,61)_4': str(r['velocity2']['(40,61)']), '(40,100)_4': str(r['velocity2']['(40,100)']), '(46,61)_4': str(r['velocity2']['(46,61)']), '(46,100)_4': str(r['velocity2']['(46,100)']), '(61,100)_4': str(r['velocity2']['(61,100)']), \
        	             '(40,46)_5': str(r['velocity3']['(40,46)']), '(40,61)_5': str(r['velocity3']['(40,61)']), '(40,100)_5': str(r['velocity3']['(40,100)']), '(46,61)_5': str(r['velocity3']['(46,61)']), '(46,100)_5': str(r['velocity3']['(46,100)']), '(61,100)_5': str(r['velocity3']['(61,100)'])})

"""
main_dir = 'D:\\data complete\\data'

from PIL import Image

fig_num = 100

for fig_id in range(fig_num):
    color_image  = np.load(main_dir + '/' + 'rgb/' + 'rgb_im_' + str(fig_id) + '.npy')
    img = Image.fromarray(color_image, 'RGB')
    img.save('C:\\Users\\Sang Uk\\Desktop\\cv\\metrics\\png\\' + 'rgb_' + "{:06d}".format(fig_id) + '.png')
"""

"""
fig_id = 69
full_ids = [40, 46, 61, 100]
pcd_40 = o3d.io.read_point_cloud('C:\\Users\\Sang Uk\\Desktop\\cv\\pcd\\' + 'pcd_' + str(fig_id) + '_' + str(40) + '.pcd')
pcd_46 = o3d.io.read_point_cloud('C:\\Users\\Sang Uk\\Desktop\\cv\\pcd\\' + 'pcd_' + str(fig_id) + '_' + str(46) + '.pcd')
pcd_61 = o3d.io.read_point_cloud('C:\\Users\\Sang Uk\\Desktop\\cv\\pcd\\' + 'pcd_' + str(fig_id) + '_' + str(61) + '.pcd')
pcd_100 = o3d.io.read_point_cloud('C:\\Users\\Sang Uk\\Desktop\\cv\\pcd\\' + 'pcd_' + str(fig_id) + '_' + str(100) + '.pcd')
pcd_40.paint_uniform_color([1, 0, 0])
pcd_46.paint_uniform_color([0, 1, 0])
pcd_61.paint_uniform_color([0, 0, 1])
pcd_100.paint_uniform_color([1, 0, 1])
o3d.visualization.draw_geometries([pcd_40, pcd_46, pcd_61, pcd_100])

from matplotlib import pyplot as plt
from PIL import Image
fig_id = 69
main_dir = 'D:\\data complete\\data'
color = np.load(main_dir + "\\rgb\\rgb_im_" + str(fig_id) + ".npy")
depth = np.load(main_dir + "\\depth\\depth_im_" + str(fig_id) + ".npy")
img_color = Image.fromarray(color, 'RGB')
img_color.show()
img_depth = Image.fromarray(depth/3000*225)
img_depth.show()

camera_intrinsic_for_mask = o3d.camera.PinholeCameraIntrinsic()
camera_intrinsic_for_mask.set_intrinsics(depth_camera_info['width'], depth_camera_info['height'], fx, fy, cx, cy)

timer = time.time()

depth_image_for_mask = o3d.geometry.Image(depth_image)
for i in range(1):
    pcd = o3d.geometry.PointCloud.create_from_depth_image(depth_image_for_mask, camera_intrinsic, depth_scale = 1, depth_trunc = 10000)


print(time.time()-timer)

camera_intrinsic_for_mask = o3d.camera.PinholeCameraIntrinsic()
camera_intrinsic_for_mask.set_intrinsics(100, 100, fx, fy, cx, cy)

timer = time.time()

depth_image_for_mask = o3d.geometry.Image(copy.copy(depth_image[0:100, 0:100]))
for i in range(100):
    pcd2 = o3d.geometry.PointCloud.create_from_depth_image(depth_image_for_mask, camera_intrinsic, depth_scale = 1, depth_trunc = 10000)


print(time.time()-timer)

camera_intrinsic_for_mask = o3d.camera.PinholeCameraIntrinsic()
camera_intrinsic_for_mask.set_intrinsics(depth_camera_info['width'], depth_camera_info['height'], fx, fy, cx, cy)

timer = time.time()

a = np.zeros((720, 1280))
a[0:500, 0:500] = np.ones((500, 500))
a = a.astype(np.uint16)
depth_image_for_mask = o3d.geometry.Image(a * depth_image)
for i in range(100):
    pcd3 = o3d.geometry.PointCloud.create_from_depth_image(depth_image_for_mask, camera_intrinsic, depth_scale = 1, depth_trunc = 10000)


print(time.time()-timer)
timer = time.time()

pcd2.paint_uniform_color([0, 0, 1])
pcd3.paint_uniform_color([0, 1, 0])


timer = time.time()

for i in range(180):
    depth_image_multiple = np.sum(depth_image*depth_image)


print(time.time()-timer)
timer = time.time()


print(np.sum(overlapped_pixels_int))
print(np.sum(mask_for_j_int))
print(np.sum(dilated_overlaps * masks[::skip_ind, ::skip_ind, j]))
print((i, j))
print(overlapped_pixels_mean_rgb)
print(mask_for_i_mean_rgb)
print(mask_for_j_mean_rgb)
print(np.linalg.norm(overlapped_pixels_mean_rgb - mask_for_i_mean_rgb) <= np.linalg.norm(overlapped_pixels_mean_rgb - mask_for_j_mean_rgb))


import open3d as o3d
import numpy as np

from PIL import Image

import cv2

output_dir = 'D:\\data\\data 6'

color = np.load(output_dir + "\\rgb\\rgb_im_200.npy")
depth = np.load(output_dir + "\\depth\\depth_im_10.npy")

# cv2.imshow("WindowNameHere", depth)

# new_im = Image.fromarray(depth)
# new_im.save("depth_im.png")

from matplotlib import pyplot as plt
plt.imshow(depth, interpolation='nearest')
plt.savefig('depth_im.png')
plt.show()


(730-646)/641*750
(220-367)/641*750

i = 0

pcd1 = o3d.geometry.PointCloud()
file = open(output_dir_usb + '/' + 'pointcloud/positions/' + 'pointcloudbottle' + str(del_ind+i) + '.pickle', 'rb')
positions =  pickle.load(file)
file = open(output_dir_usb + '/' + 'pointcloud/colors/' + 'pointcloudbottle' + str(del_ind+i) + '.pickle', 'rb')
colors =  pickle.load(file)
colors = colors/256
pcd1.points = o3d.utility.Vector3dVector(positions)
pcd1.colors = o3d.utility.Vector3dVector(colors)

pcd2 = o3d.geometry.PointCloud()
file = open(output_dir_usb + '/' + 'pointcloud/positions/' + 'pointcloudhand' + str(del_ind+i) + '.pickle', 'rb')
positions =  pickle.load(file)
file = open(output_dir_usb + '/' + 'pointcloud/colors/' + 'pointcloudhand' + str(del_ind+i) + '.pickle', 'rb')
colors =  pickle.load(file)
colors = colors/256
pcd2.points = o3d.utility.Vector3dVector(positions)
pcd2.colors = o3d.utility.Vector3dVector(colors)

o3d.visualization.draw_geometries([pcd1, pcd2])

fig_id = 69
main_dir = 'D:\\data complete\\data'
file_mask_rcnn = open(main_dir + '/' + 'mask/' + 'maskrcnn' + str(fig_id) + '.pickle', 'rb')
r = pickle.load(file_mask_rcnn)
file_dense_pose = open(main_dir + '/' + 'maskhand/' + 'maskhand' + str(fig_id) + '.pickle', 'rb')
mask_img = pickle.load(file_dense_pose, encoding="bytes")
hand_mask = mask_img[:, :, 0]==3
r_mask = r['masks'][:, :, 3]
overlap_mask = r_mask & hand_mask
color_image = np.load(main_dir + "\\rgb\\rgb_im_" + str(fig_id) + ".npy")
depth_image = np.load(main_dir + "\\depth\\depth_im_" + str(fig_id) + ".npy")
masks_to_check = np.empty((720, 1280, 2), dtype = bool)
masks_to_check[:, :, 1] = r_mask
masks_to_check[:, :, 0] = hand_mask
masks_with_no_overlap = handle_mask_overlap(2, masks_to_check, color_image, depth_image, args)
overlap_mask_no_overlap = masks_with_no_overlap[:, :, 0] & masks_with_no_overlap[:, :, 1]

file_mask_rcnn = open(main_dir + '/' + 'mask/' + 'maskrcnn' + str(fig_id) + '.pickle', 'rb')
r = pickle.load(file_mask_rcnn)
overlap_mask = r['masks'][:, :, 1] & r['masks'][:, :, 3]
masks_to_check = np.empty((720, 1280, 2), dtype = bool)
masks_to_check[:, :, 0] = r['masks'][:, :, 1]
masks_to_check[:, :, 1] = r['masks'][:, :, 3]
masks_with_no_overlap = handle_mask_overlap(2, masks_to_check, color_image, depth_image, args)
"""
