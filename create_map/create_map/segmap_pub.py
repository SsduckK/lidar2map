import rclpy
import cv2
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image as PImage
from rclpy.node import Node
from rclpy.qos import QoSProfile
from cv_bridge import CvBridge
from std_msgs.msg import String
from sensor_msgs.msg import Image
from std_msgs.msg import Header
from inference import inference
from rclpy.time import Time

class SegMapPub(Node):
    def __init__(self):
        super().__init__('segmap_pub')
        self.image = Image()
        self.subscriptions_ = self.create_subscription(Image, "/frames", self.listener_callback, 10)
        self.publishers_ = self.create_publisher(Image, "/inference_segmap", 1)
        self.num_ctgr = 7       # TODO: ros parameter register
        # self.cap = cv2.VideoCapture(0)
        self.br = CvBridge()

    def listener_callback(self, data):
        current_frame = self.br.imgmsg_to_cv2(data)
        time_stamp = Time.from_msg(data.header.stamp).to_msg()
        self.publish_image_msg(current_frame, time_stamp)

    def publish_image_msg(self, c_frame, time_stamp):
        now = self.get_clock().now().to_msg()
        frame = cv2.rotate(c_frame, cv2.ROTATE_90_CLOCKWISE)
        inf_frame = inference.show_segmap(frame)
        inf_frame = np.asarray(inf_frame)
        self.image.height = 640
        self.image.width = 480
        self.image.encoding = "bgr8"
        self.image.step = 1920
        self.image = self.br.cv2_to_imgmsg(inf_frame)
        self.image.header.stamp = now
        self.publishers_.publish(self.image)
        
        self.get_logger().info('Publishing video frame')


def main(args=None):
    rclpy.init(args=args)
    node = SegMapPub()
    rclpy.spin(node)

if __name__ == "__main__":
    main()