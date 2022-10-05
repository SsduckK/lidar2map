from email import message
from re import S
import rclpy
import numpy as np
import math
import message_filters
import matplotlib.pyplot as plt

from std_msgs.msg import Float32MultiArray
from sensor_msgs.msg import LaserScan
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseArray
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import qos_profile_parameters


# class CreateMap(Node):
#     def __init__(self):
#         super().__init__("create_map")
#         self.x_subscription = message_filters.Subscriber(self, Float32MultiArray, "/position_x")
#         self.y_subscription = message_filters.Subscriber(self, Float32MultiArray, "/position_y")
#         self.odometry = message_filters.Subscriber(self, Float32MultiArray, "/odom")
#         self.idx = 0
#         self.subscriptions
#         self.listener_callback()

#     def callback(self):
#         self.get_logger().info("get_msg")
#         x = self.x_subscription
#         y = self.y_subscription
#         odom = self.odometry
#         self.calculate_position(self.idx, x, y)
#         self.idx += 1

#     def listener_callback(self):
#         filtered_msg = message_filters.ApproximateTimeSynchronizer([self.x_subscription, self.y_subscription, self.odometry], 10, 0.001, allow_headerless=True)
#         filtered_msg.registerCallback(self.callback)

#     def calculate_position(self, idx, x, y):
#         point_position = [0 for i in range(0, 360)]
#         for angle in range(0, 360):
#             point_position[angle] = (x[angle], y[angle])
#         self.draw_point(self, idx, point_position)
    
#     def draw_point(self, idx, position_list):
#         x_point, y_point = [], []
#         for x, y in position_list:
#             if x!= 0 and y != 0:
#                 x_point.append(x)
#                 y_point.append(y)
#         plt_show = plt.figure(figsize=(10, 10))
#         plt.plot(x_point, y_point, 'rp')
#         plt.plot(0, 0, 'bp')
#         plt.grid(True)
#         plt.savefig(f"/home/ri/draw_map/sample1/{idx}_th.png")
#         plt.close(plt_show)
#         plt.clf()    


class CreateMap(Node):
    def __init__(self):      
        super().__init__("create_map")

        lidar = message_filters.Subscriber(self, LaserScan, "/scan_filtered")
        odom = message_filters.Subscriber(self, Odometry, "/odom")

        filtered_msg = message_filters.ApproximateTimeSynchronizer([lidar, odom], 1, 0.1)
        filtered_msg.registerCallback(self.callback)
    
    def callback(self, lid, odo):
        self.get_logger().info("lid : {}".format(lid.ranges))
    

def main(args=None):
    rclpy.init(args=args)
    node = CreateMap()
    try:
        rclpy.spin(node)
    except:
        node.get_logger().info("Keyboard Interrupt (SIGINT)")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
