import config as cfg
import rclpy
import numpy as np
import open3d as o3d
import tf2_ros
import cv2

from cv_bridge import CvBridge
from std_msgs.msg import Int16MultiArray
from rclpy.node import Node
from sensor_msgs.msg import Image
from nav_msgs.msg import Odometry
from rclpy.time import Time
from depth_to_map import DepthToMap
from geometry_msgs.msg import PoseWithCovarianceStamped
from sensor_msgs.msg import LaserScan
from build_map import SemanticMapRenderer
from build_map import GridMapRenderer

class MultiMsgSub(Node):
    def __init__(self):
        super().__init__('subsribe_multi_msg')
        self.map_, self.map_shape = np.zeros((1, 1)), 0
        self.odom_time = np.zeros(0)
        self.odom_list = []
        self.lidar_time = np.zeros(0)
        self.lidar_list = []
        self.depth_time = np.zeros(0)
        self.depth_list = []
        self.segmap_time = np.zeros(0)
        self.segmap = []
        # self.tolerance = 1e+8
        self.tolerance = 4907045650409216.0
        self.br = CvBridge()
        self.grid_map = 0
        self.num_ctgr = 12       # TODO: read ros parameter
        self.label_map = None
        self.callback_count = 0
        map2d = self.create_subscription(Image, "grid_map", self.map_callback, 10)
        # odom_msg = self.create_subscription(PoseWithCovarianceStamped, "/amcl_pose", self.odom_callback, 10)
        odom_msg = self.create_subscription(Odometry, "/new_odom", self.odom_callback, 10)
        depth_msg = self.create_subscription(Image, "/camera/depth/image_rect_raw", self.depth_callback, 10)
        segmap_msg = self.create_subscription(Image, "/inference_segmap", self.segmap_callback, 10)
        lidar_msg = self.create_subscription(LaserScan, "/scan_filtered", self.lidar_callback, 10)
        self.visualize = False
        self.vis = o3d.visualization.Visualizer()
        # self.vis = o3d.visualization.Visualizer()
        # self.vis.create_window()
        # self.vis.get_view_control()

    def map_callback(self, map):
        map = self.br.imgmsg_to_cv2(map)
        map = map[..., 0]
        self.grid_map = np.zeros_like(map, dtype=np.uint8)
        self.label_map = np.zeros((map.shape[0], map.shape[1], self.num_ctgr), dtype=int) if self.label_map is None else self.label_map
        ctgr = [205, 254, 0]    # [cfg.GRID_TYPE_NONE, cfg.GRID_TYPE_FLOOR, cfg.GRID_TYPE_WALL]
        for i, c in enumerate(ctgr):
            self.grid_map[map == c] = i
        
    def odom_callback(self, odom):
        nano_sec = Time.from_msg(odom.header.stamp).nanoseconds
        self.odom_time = np.append(self.odom_time, np.asarray(nano_sec))
        self.odom_list += [odom.pose.pose]
        if len(self.odom_time) > 100:
            self.odom_time = self.odom_time[-100:]
            self.odom_list = self.odom_list[-100:]

    def lidar_callback(self, lidar):
        nano_sec = Time.from_msg(lidar.header.stamp).nanoseconds
        self.lidar_time = np.append(self.lidar_time, np.asarray(nano_sec))
        self.lidar_list += [[lidar.ranges][0]]
        if len(self.lidar_time) > 100:
            self.lidar_time = self.lidar_time[-100:]
            self.lidar_list = self.lidar_list[-100:]

    def depth_callback(self, depth):
        nano_sec = Time.from_msg(depth.header.stamp).nanoseconds
        self.depth_time = np.append(self.depth_time, np.asarray(nano_sec))
        self.depth_list += [self.br.imgmsg_to_cv2(depth)]
        if len(self.depth_time) > 100:
            self.depth_time = self.depth_time[-100:]
            self.depth_list = self.depth_list[-100:]

    def segmap_callback(self, segmap):
        segmap_time = Time.from_msg(segmap.header.stamp).nanoseconds
        segmap = self.br.imgmsg_to_cv2(segmap)
        if len(self.odom_time) and len(self.lidar_time) and type(self.grid_map) != int:
            sync_odom, odom_diff = self.sync_data(segmap_time, self.odom_time, self.odom_list)
            sync_lidar, lidar_diff = self.sync_data(segmap_time, self.lidar_time, self.lidar_list)
            self.get_logger().info(f"sync_data: {odom_diff}, {lidar_diff}")
            if odom_diff < self.tolerance and lidar_diff < self.tolerance:
                self.update_map(self.grid_map, sync_odom, sync_lidar, segmap)

    def sync_data(self, segmap_time, other_time, other_list):
        other_index = np.argmin(abs(other_time - segmap_time))
        time_diff = np.min(abs(other_time - segmap_time))
        sync_other = other_list[other_index]
        return sync_other, time_diff

    def update_map(self, grid_map, odom, lidar, segmap):
        map = DepthToMap(grid_map, odom, lidar, segmap)
        frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1, origin=[0, 0, 0.1])
        self.label_map += map.grid_label_count
        class_map = self.convert_to_semantic_map(self.label_map)
        color_map = self.create_color_map(grid_map, class_map)
        self.callback_count += 1
        print(self.callback_count)
        if self.callback_count == 920:
            grid_map = GridMapRenderer(self.grid_map).build
            total_map = SemanticMapRenderer(class_map).build
            o3d.visualization.draw_geometries([frame, grid_map, total_map])
            for i in range(1, 12):
                image = np.array(self.label_map[:, :, i], dtype=np.uint8)
                cv2.imwrite(str(i)+".png", image)
                image[image != 0] = 255
                cv2.imwrite(str(i)+"_2.png", image)
            print("saved")
            o3d.io.write_triangle_mesh("map_pcl.ply", grid_map + total_map)

        if self.visualize:
            grid_map = GridMapRenderer(self.grid_map).build
            total_map = SemanticMapRenderer(class_map).build
            self.vis.create_window()
            self.vis.add_geometry(frame)
            self.vis.add_geometry(map.point_cloud)
            self.vis.add_geometry(grid_map)
            self.visualize_map(total_map)
            # self.vis.remove_geometry(grid_map)
    
    def create_color_map(self, grid_map, class_map):
        print("grid_map", grid_map.shape, grid_map.dtype)
        color_map = (255 - cv2.cvtColor(grid_map, cv2.COLOR_GRAY2RGB) * 100).astype(np.uint8)
        # class_view = (cv2.cvtColor(class_map.astype(np.uint8), cv2.COLOR_GRAY2BGR) * 50).astype(np.uint8)
        for i, color in enumerate(cfg.CTGR_COLOR):
            if i==0:
                continue
            # color_map[class_map==i] = [(np.array(color) * 255).astype(np.uint8)]
            color_map[class_map==i] = [color[2] * 255, color[1] * 255, color[0] * 255]
        cv2.imshow("map image", color_map)
        cv2.waitKey(10)
        # cv2.imshow("class view", class_view)
        # cv2.waitKey(10)
        return color_map

    def convert_to_semantic_map(self, grid_count):
        class_map = np.argmax(grid_count, axis=2)
        return class_map

    def visualize_map(self, semantic_map):
        # if self.frame_index == 1000:
        #     self.vis.close()
        #     final_mesh_file = "/home/ri/bagfiles/test_lab/third/mesh.tri"
        #     # o3d.io.write_triangle_mesh(final_mesh_file, semantic_map.build[0])
        #     o3d.visualization.draw_geometries(self.occp_map.build + self.accumulated_map + [frame])
        # o3d.visualization.draw_geometries(self.occp_map.build + semantic_map.build + [frame])
        # o3d.visualization.draw_geometries(semantic_map.build)
        # o3d.visualization.draw_geometries([pcd_in_glb])
        self.vis.add_geometry(semantic_map)
        # for i in semantic_map.build:
        #     self.vis.add_geometry(i)
        # for i in semantic_map.build:
        #     self.vis.remove_geometry(i)
        self.vis.run()
        self.vis.remove_geometry(semantic_map)      


def main(args=None):
    rclpy.init(args=args)
    node = MultiMsgSub()
    rclpy.spin(node)


if __name__ == "__main__":
    main()
