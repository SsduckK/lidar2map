import rclpy
from rclpy.node import Node
from tf2_msgs.msg import TFMessage


class TfFilter(Node):
    def __init__(self):
        super().__init__('tf_filter')
        base_msg = self.create_subscription(TFMessage, "/tf_old", self.tf_callback, 10)
        self.tf_pub = self.create_publisher(TFMessage, "/tf", 1)

    def tf_callback(self, tf):
        transform = tf.transforms
        for trans in transform:
            print(f"[tf] base: {trans.header.frame_id}, child: {trans.child_frame_id}")
            if trans.child_frame_id == "base_footprint" and trans.header.frame_id == "odom":
                self.tf_pub.publish(tf)
    
def main(args=None):
    rclpy.init(args=args)
    node = TfFilter()
    rclpy.spin(node)

    
if __name__ == "__main__":
    main()
