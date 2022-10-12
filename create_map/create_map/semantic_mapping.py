from email.mime import base
from functools import total_ordering
import rclpy
import numpy as np
import math
import message_filters
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import copy
import cv2

from cv_bridge import CvBridge
from pyquaternion import Quaternion
from sensor_msgs.msg import LaserScan
from sensor_msgs.msg import Image
from sensor_msgs.msg import CameraInfo
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import qos_profile_parameters
from build_map import OccupancyMapRenderer, SemanticMapRenderer
import pyrealsense2 as rs
import open3d as o3d
import config as cfg


class CreateMap(Node):
    def __init__(self, base_map_path):      
        super().__init__("semantic_map")
        self.bridge = CvBridge()
        self.occp_map = OccupancyMapRenderer(base_map_path)
        self.grid_label_count = np.zeros((*self.occp_map.map.shape, cfg.NUM_CLASS), dtype=int)
        self.vis = o3d.visualization.Visualizer()
        self.frame_index = 0
        self.intrinsic = o3d.camera.PinholeCameraIntrinsic(848, 480, 418.874, 418.874, 427.171, 239.457)
        self.cam_to_base = cfg.CAM_TO_BASE
        depth = message_filters.Subscriber(self, Image, "/camera/depth/image_rect_raw")
        odom = message_filters.Subscriber(self, Odometry, "/odom")
        filtered_msg = message_filters.ApproximateTimeSynchronizer([depth, odom], 1, 0.1)
        filtered_msg.registerCallback(self.callback)

    def callback(self, depth, odom):
        pose = (odom.pose.pose)
        self.vis.create_window()
        self.vis.add_geometry(self.occp_map.build[0])
        transform_matrix = self.pose_to_matrix(pose)
        pcd_in_cam = self.get_point_cloud(depth)
        pcd_in_glb = self.transform_to_global(pcd_in_cam, transform_matrix)
        self.grid_label_count += self.count_labels(self.grid_label_count, pcd_in_glb)
        class_map = self.convert_to_semantic_map(self.grid_label_count) 
        smnt_map = SemanticMapRenderer(class_map)
        self.visualize_map(smnt_map)
        self.grid_label_count = np.zeros((*self.occp_map.map.shape, cfg.NUM_CLASS), dtype=int)
        self.frame_index += 1
        print(self.frame_index)

    def pose_to_matrix(self, pose):
        matrix = np.identity(4)
        quaternion = np.array([pose.orientation.w, pose.orientation.x, pose.orientation.y, pose.orientation.z])
        matrix[:3, :3] = o3d.geometry.get_rotation_matrix_from_quaternion(quaternion)
        matrix[:3, 3] = np.array([pose.position.x, pose.position.y, pose.position.z])
        return matrix

    def get_point_cloud(self, depth):
        depth_image = self.bridge.imgmsg_to_cv2(depth, "16UC1") 
        depth_image = o3d.geometry.Image(depth_image)
        pcd = o3d.geometry.PointCloud.create_from_depth_image(depth_image, self.intrinsic)
        # pcd.scale(0.001, center=np.array([0,0,0]))
        pcd = pcd.select_by_index(np.where(np.asarray(pcd.points)[:,2] < cfg.MAX_DEPTH)[0])
        pcd = pcd.select_by_index(np.where(np.asarray(pcd.points)[:,1] < 0.1)[0])
        return pcd

    def transform_to_global(self, pcd_in_cam, transform_matrix):
        pcd_in_rbt = copy.deepcopy(pcd_in_cam)
        pcd_in_rbt.transform(self.cam_to_base)
        pcd_in_glb = copy.deepcopy(pcd_in_rbt)
        pcd_in_glb.transform(transform_matrix)
        return pcd_in_glb

    def count_labels(self, map, pcd):
        points = (np.asarray(pcd.points) - cfg.GRID_ORIGIN) / cfg.GRID_RESOLUTION
        points[:, 1] *= -1
        points = points.astype(int)
        # TODO
        # map[points[:,1], points[:,0]] += 1
        for point in points:
            map[point[1] - 3][point[0] - 3] += 1
        return map

    def transform(self, pcd, position, orientation):
        trans_pcd = copy.deepcopy(pcd).translate(position)
        trans_pcd.rotate(trans_pcd.get_rotation_matrix_from_quaternion((orientation)))
        return trans_pcd

    def convert_to_semantic_map(self, grid_count):
        class_map = np.array(grid_count>200).astype(int) * 5
        return class_map[:, :, 0]

    def visualize_map(self, semantic_map):
        frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0, 0, 0.1])
        # o3d.visualization.draw_geometries(self.occp_map.build + semantic_map.build + [frame])
        self.vis.add_geometry(semantic_map.build[0])
        self.vis.add_geometry(frame)
        self.vis.run()




def main(args=None):
    rclpy.init(args=args)
    base_map_path = "/home/ri/bagfiles/test_lab/third/converted3.txt"
    node = CreateMap(base_map_path)
    rclpy.spin(node)


if __name__ == "__main__":
    main()
