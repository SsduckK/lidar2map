from multiprocessing import current_process
import rclpy
import sqlite3
import numpy as np
import math

from sensor_msgs.msg import LaserScan
from rclpy.node import Node
from rosidl_runtime_py.utilities import get_message
from rclpy.serialization import deserialize_message

class ReadLidar(Node):
    def __init__(self):
        super().__init__("read_lidar")
        self.recieve_lidar = self.create_subscription(LaserScan, "/scan_filtered", self.listen_callback, 10)
        self.subscriptions

    def listen_callback(self, frames):
        self.get_logger().info("Recieved")
        range_list = frames.ranges
        self.calculate_position(range_list)
    
    def calculate_position(self, ranges):
        point_position = [0 for i in range(0, 360)]         
        for angle in range(0, 360):
            if ranges[angle] > 4 or ranges[angle] == 0:
                point_position[angle] = (0, 0)
                continue
            x = ranges[angle] * math.cos(math.radians(angle))
            y = ranges[angle] * math.sin(math.radians(angle))
            point_position[angle] = (x, y)

def main(args=None):
    rclpy.init(args=args)
    node = ReadLidar()
    try:
        rclpy.spin(node)
    except:
        node.get_logger().info("Keyboard Interrupt (SIGINT)")
    finally:
        node.destroy_node
        rclpy.shutdown

if __name__ == "__main__":
    main()