import rclpy
import numpy as np
import cv2
# import config as cfg
from rclpy.node import Node
from std_msgs.msg import Int16MultiArray
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

class MapPublisher(Node):
    def __init__(self, map_image):
        super().__init__('map_pub')
        self.map_image = map_image
        self.image = Image()
        self.map_publisher = self.create_publisher(Image, 'grid_map', 10)
        self.timer = self.create_timer(1, self.publish_map)
        self.br = CvBridge()

    def publish_map(self):
        input_image = cv2.imread(self.map_image)
        self.image.height, self.image.width, _ = input_image.shape
        self.image.encoding = "8UC1"
        self.image.step = 1920
        self.image = self.br.cv2_to_imgmsg(input_image)
        self.map_publisher.publish(self.image)
    
def main(args=None):
    rclpy.init(args=args)
    image = "/home/ri/bagfiles/221026/map/first_date-second.pgm"
    node = MapPublisher(image)
    rclpy.spin(node)

    
if __name__ == "__main__":
    main()