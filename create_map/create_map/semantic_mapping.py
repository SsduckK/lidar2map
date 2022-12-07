import rclpy
from rclpy.qos import QoSProfile
import numpy as np
import message_filters
import copy
import cv2

from cv_bridge import CvBridge
from std_msgs.msg import Int8MultiArray, Int16MultiArray
from sensor_msgs.msg import Image
from nav_msgs.msg import Odometry
from rclpy.node import Node
from build_map import OccupancyMapRenderer, SemanticMapRenderer
import open3d as o3d
import config as cfg


class Mao3DBuilder(Node):
    def __init__(self):      
        super().__init__("semantic_map")
        self.bridge = CvBridge()
        self.map, self.map_shape = np.zeros((1, 1)), 0
        qos_profile = QoSProfile(depth=10)
        self.map_sub = self.create_subscription(Int16MultiArray, "grid_map", self.map_callback, qos_profile)
        self.occp_map = 0
        self.grid_label_count = 0
        self.vis = o3d.visualization.Visualizer()
        self.frame_index = 0
        self.accumulated_map = []
        self.intrinsic = o3d.camera.PinholeCameraIntrinsic(848, 480, 418.874, 418.874, 427.171, 239.457)
        self.cam_to_base = cfg.CAM_TO_BASE

        depth = message_filters.Subscriber(self, Image, "/camera/depth/image_rect_raw")
        odom = message_filters.Subscriber(self, Odometry, "/odom")
        filtered_msg = message_filters.ApproximateTimeSynchronizer([depth, odom], 1, 0.3)
        filtered_msg.registerCallback(self.callback)

        # self.rgb_subscriber = self.create_subscription(Image, '/frames', self.show_rgb, qos_profile)

        self.vis.create_window()
        self.vis.get_view_control()
    
    def map_callback(self, data):
        map, shape = data.data[:-2], data.data[-2:]
        self.map_shape = [shape[0], shape[1]]
        self.map = np.asarray(map).reshape(shape[0], shape[1])
        self.occp_map = SemanticMapRenderer(self.map)
        print("Saving mesh to file: map3D.ply")
        o3d.io.write_triangle_mesh("map111.ply", self.occp_map.build)
        # self.vis.add_geometry(self.occp_map.build)


    def callback(self, depth, odom):
        pose = (odom.pose.pose)
        self.grid_label_count = np.zeros((*self.map.shape, cfg.NUM_CLASS), dtype=int)
        transform_matrix = self.pose_to_matrix(pose)
        pcd_in_cam = self.get_point_cloud(depth)
        pcd_in_glb = self.transform_to_global(pcd_in_cam, transform_matrix)
        self.grid_label_count += self.count_labels(self.grid_label_count, pcd_in_glb)
        class_map = self.convert_to_semantic_map(self.grid_label_count) 
        smnt_map = SemanticMapRenderer(class_map)
        self.accumulated_map = smnt_map.build
        self.visualize_map(smnt_map, pcd_in_glb)
        self.frame_index += 1
        print(self.frame_index) 

    def pose_to_matrix(self, pose):
        matrix = np.identity(4)
        quaternion = np.array([pose.orientation.w, pose.orientation.x, pose.orientation.y, pose.orientation.z])
        matrix[:3, :3] = o3d.geometry.get_rotation_matrix_from_quaternion(quaternion)
        matrix[:3, 3] = np.array([pose.position.x, pose.position.y, pose.position.z])
        return matrix

    def get_point_cloud(self, depth):
        depth_image = self.bridge.imgmsg_to_cv2(depth, "16UC1") 
        depth_image = o3d.geometry.Image(depth_image)
        pcd = o3d.geometry.PointCloud.create_from_depth_image(depth_image, self.intrinsic)
        pcd = pcd.select_by_index(np.where(np.asarray(pcd.points)[:,2] < 2)[0])
        pcd = pcd.select_by_index(np.where(np.asarray(pcd.points)[:,0] < 0.4)[0])
        pcd = pcd.select_by_index(np.where(np.asarray(pcd.points)[:,0] > -0.4)[0])
        pcd = pcd.select_by_index(np.where(np.asarray(pcd.points)[:,1] < 0.1)[0])
        pcd = pcd.select_by_index(np.where(np.asarray(pcd.points)[:,1] > -0.9)[0])
        return pcd

    def transform_to_global(self, pcd_in_cam, transform_matrix):
        pcd_in_rbt = copy.deepcopy(pcd_in_cam)
        pcd_in_rbt.transform(self.cam_to_base)
        pcd_in_glb = copy.deepcopy(pcd_in_rbt)
        pcd_in_glb.transform(transform_matrix)
        return pcd_in_glb

    def count_labels(self, map, pcd):
        points = (np.asarray(pcd.points) - cfg.GRID_ORIGIN) / cfg.GRID_RESOLUTION
        points[:, 1] *= 1
        points = points.astype(int)
        # TODO
        # map[points[:,1], points[:,0]] += 1
        for point in points:
            if point[1] < self.map_shape[1] and point[0] < self.map_shape[0]:
                map[point[1]][point[0]] += 1
        return map

    def transform(self, pcd, position, orientation):
        trans_pcd = copy.deepcopy(pcd).translate(position)
        trans_pcd.rotate(trans_pcd.get_rotation_matrix_from_quaternion((orientation)))
        return trans_pcd

    def convert_to_semantic_map(self, grid_count):
        class_map = np.array(grid_count>500).astype(int) * 4
        return class_map[:, :, 0]

    def visualize_map(self, semantic_map, pcd_in_glb):
        frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0, 0, 0.1])
        if self.frame_index == 1000:
            self.vis.close()
            final_mesh_file = "/home/ri/bagfiles/test_lab/third/mesh.tri"
            # o3d.io.write_triangle_mesh(final_mesh_file, semantic_map.build[0])
            o3d.visualization.draw_geometries(self.occp_map.build + self.accumulated_map + [frame])
        # o3d.visualization.draw_geometries(self.occp_map.build + semantic_map.build + [frame])
        # o3d.visualization.draw_geometries(semantic_map.build)
        # o3d.visualization.draw_geometries([pcd_in_glb])
        self.vis.add_geometry(frame)
        self.vis.add_geometry(pcd_in_glb)
        self.vis.add_geometry(semantic_map.build)
        # for i in semantic_map.build:
        #     self.vis.add_geometry(i)
        self.vis.run()
        # for i in semantic_map.build:
        #     self.vis.remove_geometry(i)
        self.vis.remove_geometry(pcd_in_glb)

    def show_rgb(self, rgb):
        rgb_image = self.bridge.imgmsg_to_cv2(rgb, "8UC3")
        print("rgb image", rgb_image.shape)
        cv2.imshow("rgb", rgb_image)
        cv2.waitKey(10)


def main(args=None):
    rclpy.init(args=args)
    node = Mao3DBuilder()
    rclpy.spin(node)


if __name__ == "__main__":
    main()
