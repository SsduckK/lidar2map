from array import array
import rclpy
import numpy as np
import math

from std_msgs.msg import Float32MultiArray
from sensor_msgs.msg import LaserScan
from rclpy.node import Node

class ReadLidar(Node):
    def __init__(self):
        super().__init__("read_lidar")
        self.recieve_lidar = self.create_subscription(LaserScan, "/scan_filtered", self.listen_callback, 10)
        self.publish_position_x = self.create_publisher(Float32MultiArray, "/position_x", 10)
        self.publish_position_y = self.create_publisher(Float32MultiArray, "/position_y", 10)
        self.subscriptions

    def listen_callback(self, frames):
        self.get_logger().info("Recieved")
        range_list = frames.ranges
        position_x, position_y = self.calculate_position(range_list)
        self.publish_position_msg(position_x, position_y)
    
    def calculate_position(self, ranges):
        point_position = [0 for i in range(0, 360)]
        x_list = []
        y_list = []
        for angle in range(0, 360):
            if ranges[angle] > 4 or ranges[angle] == 0:
                point_position[angle] = (0, 0)
                continue
            x = ranges[angle] * math.cos(math.radians(angle))
            y = ranges[angle] * math.sin(math.radians(angle))
            #point_position[angle] = [x, y]
            x_list.append(x)
            y_list.append(y)
        #return point_position
        return x_list, y_list

    def publish_position_msg(self, x, y):
        position_x, position_y = Float32MultiArray(), Float32MultiArray()
        position_x.data = x
        position_y.data = y
        self.publish_position_x.publish(position_x)
        self.publish_position_y.publish(position_y)

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