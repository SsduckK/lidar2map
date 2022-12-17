import os
import sys
import rclpy
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image as PImage
from rclpy.node import Node
from rclpy.qos import QoSProfile
from cv_bridge import CvBridge
from std_msgs.msg import String
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image
from std_msgs.msg import Header
from rclpy.time import Time

# add current dir to python path
cur_path = os.path.dirname(os.path.abspath(__file__))
if cur_path not in sys.path:
    sys.path.append(cur_path)

import inference


class SegMapPub(Node):
    def __init__(self):
        super().__init__('segmap_pub')
        self.image = Image()
        self.subscriptions_ = self.create_subscription(Image, "/frames", self.listener_callback, 10)
        self.publishers_ = self.create_publisher(Image, "/inference_segmap", 1)
        self.num_ctgr = 11       # TODO: ros parameter register
        # self.cap = cv2.VideoCapture(0)
        self.br = CvBridge()
        odom_msg = self.create_subscription(Odometry, "/odom", self.odom_callback, 10)
        self.odom_time = 0

    def listener_callback(self, data):
        current_frame = self.br.imgmsg_to_cv2(data)
        if self.odom_time != 0:
            self.publish_image_msg(current_frame)

    def odom_callback(self, odom):
        self.odom_time = odom.header.stamp

    def publish_image_msg(self, c_frame):
        frame = cv2.rotate(c_frame, cv2.ROTATE_90_CLOCKWISE)
        inf_frame = inference.show_segmap(frame)
        inf_frame = np.asarray(inf_frame)
        self.image.height = 640
        self.image.width = 480
        self.image.encoding = "bgr8"
        self.image.step = 1920
        self.image = self.br.cv2_to_imgmsg(inf_frame)
        self.image.header.stamp = self.odom_time
        self.publishers_.publish(self.image)
        
        self.get_logger().info('Publishing video frame')


def main(args=None):
    rclpy.init(args=args)
    node = SegMapPub()
    rclpy.spin(node)

if __name__ == "__main__":
    main()
