from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import pyrealsense2 as rs
import cv2

###############################################################
# prepare realsense

class realsense_camera:
    def __init__(self):
        # Create a pipeline
        self.pipeline = rs.pipeline()

        # Create a config and configure the pipeline to stream
        # different resolutions of color and depth streams
        config = rs.config()
        config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)

        # Start streaming
        self.profile = self.pipeline.start(config)

        # Create an align object
        # rs.align allows us to perform alignment of depth frames to others frames
        # The "align_to" is the stream type to which we plan to align depth frames.
        align_to = rs.stream.color
        self.align = rs.align(align_to)

        """
        depth_profile = rs.video_stream_profile(self.profile.get_stream(rs.stream.depth))
        depth_intrinsics = depth_profile.get_intrinsics()
        print(depth_intrinsics)
        color_profile = rs.video_stream_profile(self.profile.get_stream(rs.stream.color))
        color_intrinsics = color_profile.get_intrinsics()
        print(color_intrinsics)
        """

        for x in range(100):
            self.pipeline.wait_for_frames()
        print("realsense ready")

    def get_realsense_images(self):
        while True:
            # Get frameset of color and depth
            # frames.get_depth_frame() is a 640x360 depth image
            frames = self.pipeline.wait_for_frames()
            # Align the depth frame to color frame
            aligned_frames = self.align.process(frames)
            # Get aligned frames
            aligned_depth_frame = aligned_frames.get_depth_frame() # aligned_depth_frame is a 640x480 depth image
            color_frame = aligned_frames.get_color_frame()
            # Validate that both frames are valid
            if not(not aligned_depth_frame or not color_frame):
                # Getting intrinsics
                # We need depth camera intrinsic, not color camera intrinsic
                # print(aligned_depth_frame.get_profile().as_video_stream_profile().get_intrinsics())
                depth_image = np.asanyarray(aligned_depth_frame.get_data())
                color_image = np.asanyarray(color_frame.get_data())
                color_image = cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR)
                return depth_image, color_image
