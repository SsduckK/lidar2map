import rclpy
from rclpy.qos import QoSProfile
import numpy as np
import message_filters
import copy
import cv2

from cv_bridge import CvBridge
from std_msgs.msg import Int8MultiArray, Int16MultiArray
from sensor_msgs.msg import Image
from nav_msgs.msg import Odometry
from rclpy.node import Node
from build_map import OccupancyMapRenderer, SemanticMapRenderer
import open3d as o3d
import config as cfg

class Create2DMap(Node):
    def __init__(self):
        super().__init__("map_2d")
        self.map, self.map_shape = np.zeros((1, 1)), 0
        qos_profile = QoSProfile(depth=10)
        self.occp_map = 0
        self.vis = o3d.visualization.Visualizer()
        self.map_sub = self.create_subscription(Int16MultiArray, "grid_map", self.map_callback, qos_profile)
        self.map_sub
        # self.vis.create_window()
    
    def map_callback(self, data):
        map, shape = data.data[:-2], data.data[-2:]
        self.map_shape = [shape[0], shape[1]]
        self.map = np.asarray(map).reshape(shape[0], shape[1])
        self.occp_map = SemanticMapRenderer(self.map)
        print("Saving mesh to file: map3D.ply")
        o3d.io.write_triangle_mesh("map2d.ply", self.occp_map.build)
        # self.vis.add_geometry(self.occp_map.build)

def main(args=None):
    rclpy.init(args=args)
    node = Create2DMap()
    rclpy.spin(node)

if __name__ == "__main__":
    main()