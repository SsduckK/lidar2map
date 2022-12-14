import rclpy
import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d
import message_filters
import tf2_ros

from rclpy.time import Time
from rclpy.node import Node
from nav_msgs.msg import Odometry
from tf2_msgs.msg import TFMessage
from geometry_msgs.msg import TransformStamped

class OdomCorrector(Node):
    def __init__(self):
        super().__init__('odom_corrector')
        self.base_odom = Odometry()
        self.transform_pub = self.create_publisher(TransformStamped, "/base_footprint", 1)
        base_msg = self.create_subscription(TFMessage, "/tf", self.tf_callback, 10)
        odom_msg = self.create_subscription(Odometry, "/odom", self.odom_callback, 10)
        self.base_odom_publsiher = self.create_publisher(Odometry, "/new_odom", 1)
        self.prv_pose = [0, 0, 0, 0]
        self.footprint = 1
        self.outlier_cnt = 0
        self.frame_cnt = 0

    def tf_callback(self, tf):
        transform = tf.transforms
        tf = transform[0]
        tf_rot, tf_trans = tf.transform.rotation, tf.transform.translation
        tf_quaternion = tf_rot.w, tf_rot.x, tf_rot.y, tf_rot.z
        if tf_rot.x != 0 or tf_rot.y != 0:
            pass
        assert np.isclose(np.linalg.norm(tf_quaternion), 1, 0.0001), f"[pose_to_matrix] quaterion norm={np.linalg.norm(tf_quaternion)}"
        tf_matrix = np.identity(4)
        tf_matrix[:3, 3] = np.array([tf_trans.x, tf_trans.y, tf_trans.z])
        tf_matrix[:3, :3] = o3d.geometry.get_rotation_matrix_from_quaternion(tf_quaternion)
        self.footprint = tf_matrix

    def odom_callback(self, odom):
        time = Time.from_msg(odom.header.stamp).to_msg()
        odom_pose_x, odom_pose_y, odom_pose_z = odom.pose.pose.position.x, odom.pose.pose.position.y, odom.pose.pose.position.z
        odom_ori_w, odom_ori_x, odom_ori_y, odom_ori_z = odom.pose.pose.orientation.w, odom.pose.pose.orientation.x, odom.pose.pose.orientation.y, odom.pose.pose.orientation.z
        odom_matrix = np.identity(4)
        odom_quaternion = odom_ori_w, odom_ori_x, odom_ori_y, odom_ori_z
        assert np.isclose(np.linalg.norm(odom_quaternion), 1, 0.0001), f"[pose_to_matrix] quaterion norm={np.linalg.norm(odom_quaternion)}"
        odom_matrix[:3, :3] = o3d.geometry.get_rotation_matrix_from_quaternion(odom_quaternion)
        odom_matrix[:3, 3] = np.array([odom_pose_x, odom_pose_y, odom_pose_z])
        new_matrix = self.footprint@odom_matrix
        self.pub_new_odom(time, new_matrix, odom_matrix)

    def pub_new_odom(self, time, matrix, odom_matrix):
        rotation = matrix[:3, :3]
        translation = matrix[:3, 3]
        quatertnion = self.rotationMatrixToQuaternion(rotation)
        self.base_odom.pose.pose.position.x = translation[0]
        self.base_odom.pose.pose.position.y = translation[1]
        self.base_odom.pose.pose.position.z = translation[2]
        self.base_odom.pose.pose.orientation.w = quatertnion[3]
        self.base_odom.pose.pose.orientation.x = quatertnion[0]
        self.base_odom.pose.pose.orientation.y = quatertnion[1]
        self.base_odom.pose.pose.orientation.z = quatertnion[2]
        self.base_odom.header.stamp = time
        dist = np.linalg.norm(np.array([translation[0] - self.prv_pose[0], translation[1] - self.prv_pose[1]]))
        quat_dist1 = np.linalg.norm(np.array([quatertnion[2] - self.prv_pose[2], quatertnion[3] - self.prv_pose[3]]))
        quat_dist2 = np.linalg.norm(np.array([quatertnion[2] + self.prv_pose[2], quatertnion[3] + self.prv_pose[3]]))
        quat_dist = min(quat_dist1, quat_dist2)
        # print(f"dist: {dist:1.4f}, {quat_dist:1.4f}, trans: {translation[0]:1.4f}, {translation[1]:1.4f}, quat: {quatertnion[3]:1.4f}, {quatertnion[2]:1.4f}" \
        #     f", odom: {odom_matrix[0,3]:1.4f}, {odom_matrix[1,3]:1.4f}"
        # )
        # print(f"diff : {quatertnion[2] - self.prv_pose[3]}")
        if (quat_dist > 0.2 or dist > 0.5) and self.outlier_cnt <= 2:
            print("outlier!")
            self.outlier_cnt += 1
        else:
            self.outlier_cnt = 0
            self.prv_pose = [translation[0], translation[1], quatertnion[2], quatertnion[3]]
            self.draw_point(matrix)
            self.base_odom_publsiher.publish(self.base_odom)

    def rotationMatrixToQuaternion(self, m):
        #q0 = qw
        t = np.matrix.trace(m)
        q = np.asarray([0.0, 0.0, 0.0, 0.0], dtype=np.float64)

        if(t > 0):
            t = np.sqrt(t + 1)
            q[3] = 0.5 * t
            t = 0.5/t
            q[0] = (m[2,1] - m[1,2]) * t
            q[1] = (m[0,2] - m[2,0]) * t
            q[2] = (m[1,0] - m[0,1]) * t

        else:
            i = 0
            if (m[1,1] > m[0,0]):
                i = 1
            if (m[2,2] > m[i,i]):
                i = 2
            j = (i+1)%3
            k = (j+1)%3

            t = np.sqrt(m[i,i] - m[j,j] - m[k,k] + 1)
            q[i] = 0.5 * t
            t = 0.5 / t
            q[3] = (m[k,j] - m[j,k]) * t
            q[j] = (m[j,i] + m[i,j]) * t
            q[k] = (m[k,i] + m[i,k]) * t

        return q

    def draw_point(self, matrix):
        x, y = matrix[:, 3][0], matrix[:, 3][1]
        plt.plot(x, y, 'bo')
        self.frame_cnt += 1
        if self.frame_cnt % 20 == 0:
            plt.savefig("new_odom_base.png")
    
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
