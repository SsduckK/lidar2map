import rclpy
import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d

from rclpy.node import Node
from nav_msgs.msg import Odometry


pcd = o3d.io.read_triangle_mesh("/home/ri/colcon_ws/src/lidar2map/data/lidar/map_pcl.ply")
o3d.visualization.draw_geometries([pcd])