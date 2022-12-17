import rclpy
import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d
import message_filters
import tf2_ros


from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener

from rclpy.time import Time
from rclpy.node import Node
from nav_msgs.msg import Odometry
from tf2_msgs.msg import TFMessage
from geometry_msgs.msg import TransformStamped


class OdomPublisher(Node):
    def __init__(self):
        super().__init__('odom_publisher')
        self.base_odom = Odometry()
        self.base_odom_publsiher = self.create_publisher(Odometry, "/new_odom", 1)

        # Declare and acquire `target_frame` parameter
        self.target_frame = self.declare_parameter(
          'target_frame', 'turtle1').get_parameter_value().string_value

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Call on_timer function every second
        self.timer = self.create_timer(0.2, self.on_timer)

    def on_timer(self):
        # Store frame names in variables that will be used to
        # compute transformations
        from_frame_rel = 'base_footprint'
        to_frame_rel = 'map'

        # Look up for the transformation between target_frame and turtle2 frames
        # and send velocity commands for turtle2 to reach target_frame
        try:
            t = self.tf_buffer.lookup_transform(
                to_frame_rel,
                from_frame_rel,
                rclpy.time.Time())
        except TransformException as ex:
            self.get_logger().info(
                f'Could not transform {to_frame_rel} to {from_frame_rel}: {ex}')
            return
        print("trans\n\n\n\n")
        print(t)
        self.base_odom.header.stamp = self.get_clock().now().to_msg()
        self.base_odom.pose.pose.position.x = t.transform.translation.x
        self.base_odom.pose.pose.position.y = t.transform.translation.y
        self.base_odom.pose.pose.position.z = t.transform.translation.z
        self.base_odom.pose.pose.orientation = t.transform.rotation
        self.base_odom_publsiher.publish(self.base_odom)

                
                
def main(args=None):
    rclpy.init(args=args)
    node = OdomPublisher()
    rclpy.spin(node)

    
if __name__ == "__main__":
    main()
