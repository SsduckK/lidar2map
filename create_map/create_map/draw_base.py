import rclpy
import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d
import message_filters
import tf2_ros

from rclpy.node import Node
from nav_msgs.msg import Odometry
from tf2_msgs.msg import TFMessage
from geometry_msgs.msg import TransformStamped

class OdomCorrector(Node):
    def __init__(self):
        super().__init__('odom_corrector')
        self.base_odom = Odometry()
        self.transform_pub = self.create_publisher(TransformStamped, "/base_footprint", 1)
        map_msg = self.create_subscription(TFMessage, "/tf", self.tf_callback, 10)


    def tf_callback(self, tf):
        transform = tf.transforms
        self.tf = transform[0]
        x, y = self.tf.transform.translation.x, self.tf.transform.translation.y
        self.draw_point(x, y)
        
    def draw_point(self, x_, y_):
        x, y = x_, y_
        plt.plot(x, y, 'go')
        plt.savefig("only_base.png")
    
def main(args=None):
    rclpy.init(args=args)
    node = OdomCorrector()
    rclpy.spin(node)

    
if __name__ == "__main__":
    main()

    ###
    # x, y, z = odom.pose.pose.position.x, odom.pose.pose.position.y, odom.pose.pose.position.z
    #     plt.plot(x, y, 'ro')
    #     plt.savefig("only_odom.png")
