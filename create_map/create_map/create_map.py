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
from build_map import MapRenderer
import pyrealsense2 as rs
import open3d as o3d

def get_grid_lineset(h_min_val, h_max_val, w_min_val, w_max_val, ignore_axis, grid_length, color):
    
    num_h_grid = int(np.round((h_max_val - h_min_val) // grid_length, -1))
    num_w_grid = int(np.round((w_max_val - w_min_val) // grid_length, -1))
    
    num_h_grid_mid = num_h_grid // 2
    num_w_grid_mid = num_w_grid // 2
    
    grid_vertexes_order = np.zeros((num_h_grid, num_w_grid)).astype(np.int16)
    grid_vertexes = []
    vertex_order_index = 0
    
    for h in range(num_h_grid):
        for w in range(num_w_grid):
            grid_vertexes_order[h][w] = vertex_order_index
            if ignore_axis == 0:
                grid_vertexes.append([0, grid_length*w + w_min_val, grid_length*h + h_min_val])
            elif ignore_axis == 1:
                grid_vertexes.append([grid_length*h + h_min_val, 0, grid_length*w + w_min_val])
            elif ignore_axis == 2:
                grid_vertexes.append([grid_length*w + w_min_val, grid_length*h + h_min_val, 0])
            else:
                pass                
            vertex_order_index += 1       
            
    next_h = [-1, 0, 0, 1]
    next_w = [0, -1, 1, 0]
    grid_lines = []
    for h in range(num_h_grid):
        for w in range(num_w_grid):
            here_h = h
            here_w = w
            for i in range(4):
                there_h = h + next_h[i]
                there_w = w +  next_w[i]            
                if (0 <= there_h and there_h < num_h_grid) and (0 <= there_w and there_w < num_w_grid):
                    grid_lines.append([grid_vertexes_order[here_h][here_w], grid_vertexes_order[there_h][there_w]])
                    
                    
    colors = [color for i in range(len(grid_lines))]
    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(grid_vertexes),
        lines=o3d.utility.Vector2iVector(grid_lines),
    )
    line_set.colors = o3d.utility.Vector3dVector(colors)
    
    return line_set   


class CreateMap(Node):
    def __init__(self):      
        super().__init__("create_map")
        file = "/home/ri/bagfiles/test_lab/first_sample/converted.txt"
        self.br = CvBridge()
        self.map = MapRenderer(file)
        self.point = np.zeros((0, 3))
        self.pcd = o3d.geometry.PointCloud()
        self.base_map = self.draw_base_map(self.map.map.shape)
        frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1, origin=[0, 0, 0])
        self.count = 0
        self.cam = o3d.camera.PinholeCameraIntrinsic(848, 480, 418.874, 418.874, 427.171, 239.457)
        depth = message_filters.Subscriber(self, Image, "/camera/depth/image_rect_raw")
        odom = message_filters.Subscriber(self, Odometry, "/odom")

        filtered_msg = message_filters.ApproximateTimeSynchronizer([depth, odom], 1, 0.1)
        filtered_msg.registerCallback(self.callback)

        ######

        range_min_xyz = (-80, -80, 0)
        range_max_xyz = (00, 80, 40)

        x_min_val, y_min_val, z_min_val = range_min_xyz
        x_max_val, y_max_val, z_max_val = range_max_xyz

        grid_len = 1

        R, G, B = 1, 0, 0
        lineset_yz_5 = get_grid_lineset(z_min_val, z_max_val, y_min_val, y_max_val, 0, grid_len, [R, G, B])
        lineset_zx_5 = get_grid_lineset(x_min_val, x_max_val, z_min_val, z_max_val, 1, grid_len, [R, G, B])
        lineset_xy_5 = get_grid_lineset(y_min_val, y_max_val, x_min_val, x_max_val, 2, grid_len, [R, G, B]) 

        self.pcd.extend([lineset_xy_5, lineset_yz_5, lineset_zx_5])
        o3d.visualization.draw_geometries(self.pcd)

        ####

        # subscribe depth, slam pose, intrinsic
        # convert depth to open3d image
        # convert intrinsic to open3d intrinsic

        # open3d (depth, intrinsic) to point cloud
        # open3d.geometry.create_point_cloud_from_depth_image

        # transform point cloud
        # http://www.open3d.org/docs/0.7.0/python_api/open3d.geometry.Geometry3D.html?highlight=transform#open3d.geometry.Geometry3D.transform
        # show
    
    def callback(self, dep, odo):
        pose = (odo.pose.pose)
        position = pose.position
        orientation = pose.orientation
        current_position = self.get_position(position, self.base_map.shape)
        current_orientation = self.get_orientation(orientation)
        pcd = self.get_point_cloud(dep)
        points = np.asarray(pcd.points)
        # pcd = pcd.select_by_index(np.where(points[:,1] > -0.3 )[0])
        # pcd = pcd.select_by_index(np.where(points[:,1] < 0.05)[0])
        pcd = pcd.select_by_index(np.where(points[:,2] < 3.5)[0])
        cam_to_base = np.array([0.5, 0.5, -0.5, 0.5])
        pcd.rotate(pcd.get_rotation_matrix_from_quaternion((cam_to_base)))
        pcd.scale(50, center = pcd.get_center())
        pcd_trans = self.transform(pcd, current_position, current_orientation)
        # pcd_trans.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
        pcd_trans = pcd_trans.voxel_down_sample(voxel_size=0.5)
        trans_point = np.asarray(pcd_trans.points)
        self.base_map += self.add_point(self.base_map, trans_point)
        print(self.base_map)
        self.point = np.concatenate((self.point, trans_point))
        self.pcd.points = o3d.utility.Vector3dVector(self.point)
        self.count += 1
        print(self.count)
        if self.count == 100:
            frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=5, origin=[0, 0, 0])
            self.pcd = self.pcd.voxel_down_sample(voxel_size=0.5)
            self.visualize_base_map(self.base_map)
            self.map.build = self.map.build.extend([self.pcd])
            o3d.visualization.draw_geometries([self.pcd, frame])

    def draw_base_map(self, shape):
        base_map = np.zeros(shape)
        return base_map

    def get_point_cloud(self, depth):
        depth_image = self.br.imgmsg_to_cv2(depth, "16UC1") 
        depth_image = o3d.geometry.Image(depth_image)
        pcd = o3d.geometry.PointCloud.create_from_depth_image(depth_image, self.cam)
        return pcd

    def get_position(self, position, shape):
        x = position.x - 19 + int(shape[1]/2)
        y = position.y + int(shape[0]/2)
        z = 0
        position_matrix = np.array([x, y, z])
        return position_matrix

    def get_orientation(self, orientation):
        q = np.array([orientation.w, orientation.x, orientation.y, orientation.z])
        return q
    
    def transform(self, pcd, position, orientation):
        trans_pcd = copy.deepcopy(pcd).translate(position)
        trans_pcd.rotate(trans_pcd.get_rotation_matrix_from_quaternion((orientation)))
        return trans_pcd

    def add_point(self, base, points):
        new_map = np.asarray(base)
        for point in points:
            if point[0] > new_map.shape[1] or point[1] > new_map.shape[0] or point[0] < 0 or point[1] < 0 :
                continue
            new_map[int(point[1])][int(point[0])] += 1
        return new_map

    def visualize_base_map(self, base_map):
        pass


def main(args=None):
    # intrinsic = o3d.camera.PinholeCameraIntrinsic(
    #     o3d.camera.PinholeCameraIntrinsicParameters.)
    # print(intrinsic)
    rclpy.init(args=args)
    node = CreateMap()
    # try:
    rclpy.spin(node)
    # except:
    #     node.get_logger().info("Keyboard Interrupt (SIGINT)")
    # finally:
    #     node.destroy_node()
    #     rclpy.shutdown()


if __name__ == "__main__":
    main()
