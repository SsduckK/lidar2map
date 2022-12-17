import rclpy
import os
import numpy as np
import open3d as o3d
import cv2
import matplotlib.pyplot as plt
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image
from nav_msgs.msg import Odometry
from rclpy.time import Time
from sensor_msgs.msg import LaserScan

import create_map.config as cfg
from create_map.build_map import SemanticMapRenderer
from create_map.build_map import GridMapRenderer
from create_map.depth_to_map import DepthToMap


class MultiMsgSub(Node):
    def __init__(self):
        super().__init__('subsribe_multi_msg')
        self.map_, self.map_shape = np.zeros((1, 1)), 0
        self.sub_data_heap = {key: {"time": np.zeros(0), "data": []} for key in ["odom", "lidar", "depth"]}
        # self.tolerance = 1e+8
        self.tolerance = 5251135165601792.0
        self.br = CvBridge()
        self.grid_map = None
        self.num_ctgr = 12       # TODO: read ros parameter
        self.label_map = None
        self.callback_count = 0
        self.use_depth = True
        self.use_lidar = False
        map2d = self.create_subscription(Image, "grid_map", self.map_callback, 10)
        odom_msg = self.create_subscription(Odometry, "/new_odom", self.odom_callback, 10)
        # odom_msg = self.create_subscription(Odometry, "/odom", self.odom_callback, 10)
        segmap_msg = self.create_subscription(Image, "/inference_segmap", self.segmap_callback, 10)
        if self.use_depth:
            depth_msg = self.create_subscription(Image, "/camera/depth/image_rect_raw", self.depth_callback, 10)
        if self.use_lidar:
            lidar_msg = self.create_subscription(LaserScan, "/scan", self.lidar_callback, 10)
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
    
    def update_data(self, name, nano_sec, data):
        self.sub_data_heap[name]["time"] = np.append(self.sub_data_heap[name]["time"], np.asarray(nano_sec))
        self.sub_data_heap[name]["data"].append(data)
        self.sub_data_heap[name]["time"] = self.sub_data_heap[name]["time"][-100:]
        self.sub_data_heap[name]["data"] = self.sub_data_heap[name]["data"][-100:]
        
    def odom_callback(self, odom):
        nano_sec = Time.from_msg(odom.header.stamp).nanoseconds
        self.update_data("odom", nano_sec, odom.pose.pose)

    def lidar_callback(self, lidar):
        nano_sec = Time.from_msg(lidar.header.stamp).nanoseconds
        self.update_data("lidar", nano_sec, lidar.ranges)
        
    def depth_callback(self, depth):
        nano_sec = Time.from_msg(depth.header.stamp).nanoseconds
        self.update_data("depth", nano_sec, self.br.imgmsg_to_cv2(depth))
        
    def segmap_callback(self, segmap):
        segmap_time = Time.from_msg(segmap.header.stamp).nanoseconds
        segmap = self.br.imgmsg_to_cv2(segmap)
        if self.grid_map is None:
            return
        sync_odom, odom_diff = self.sync_data(segmap_time, self.sub_data_heap["odom"]["time"], self.sub_data_heap["odom"]["data"])
        sync_depth, depth_diff = self.sync_data(segmap_time, self.sub_data_heap["depth"]["time"], self.sub_data_heap["depth"]["data"])
        sync_lidar, lidar_diff = self.sync_data(segmap_time, self.sub_data_heap["lidar"]["time"], self.sub_data_heap["lidar"]["data"])
        self.get_logger().info(f"sync_data: {odom_diff}, {depth_diff}, {lidar_diff}")
        if odom_diff < self.tolerance and lidar_diff < self.tolerance:
            self.update_map(self.grid_map, sync_odom, sync_depth, sync_lidar, segmap)

    def sync_data(self, segmap_time, other_time, other_list):
        if len(other_list) == 0:
            return None, 0
        other_index = np.argmin(abs(other_time - segmap_time))
        time_diff = np.min(abs(other_time - segmap_time))
        sync_other = other_list[other_index]
        return sync_other, time_diff

    def update_map(self, grid_map, odom, depth, lidar, segmap):
        if lidar is not None:
            lidar_pts = self.lidar_to_point_cloud(lidar)
            lidar_map = DepthToMap(grid_map, odom, lidar_pts, segmap)
            self.label_map += lidar_map.grid_label_count
        if depth is not None:
            depth_pts = self.depth_to_point_cloud(depth)
            if depth_pts is not None:
                depth_map = DepthToMap(grid_map, odom, depth_pts, segmap)
                self.label_map += depth_map.grid_label_count
            else:
                print("no points from depth map!!")
        
        class_map = self.convert_to_semantic_map(self.label_map)
        self.show_class_color_map(grid_map, class_map)
        self.callback_count += 1
        print("callback count:", self.callback_count)
        if self.callback_count == 1300:
            self.finalize(class_map)

    def draw_point(self, x, y):
        plt.plot(x, y, 'bo')
        plt.savefig("/home/ri/colcon_ws/src/lidar2map/data/run.png")

    def lidar_to_point_cloud(self, ranges):
        idx = np.arange(360)
        ranges = np.array(ranges)
        points = np.transpose(np.stack([-np.sin(np.deg2rad(idx)) * ranges, 
                                                  np.full((360), 0.1), 
                                                  np.cos(np.deg2rad(idx)) * ranges + 0.02, 
                                                  np.full((360), 1)]))
        points = points[np.all(np.stack([0.1 < ranges, ranges < 3.], axis=1), axis=1)]
        X, Z = points[:, 0], points[:, 2]
        points = points[np.all(np.stack([-1. < X, X < 1, Z > 0.], axis=1), axis=1), :]
        return points

    def depth_to_point_cloud(self, depth):
        depth_image = o3d.geometry.Image((depth).astype(np.uint16))
        print("depth shape", depth.shape)
        pcd = o3d.geometry.PointCloud.create_from_depth_image(depth_image, o3d.camera.PinholeCameraIntrinsic(*cfg.SCH_INSTRINSIC))
        pcd_cam = np.asarray(pcd.points)
        pcd = pcd.select_by_index(np.where(np.asarray(pcd.points)[:,2] < 2)[0])
        pcd = pcd.select_by_index(np.where(np.asarray(pcd.points)[:,0] < 0.4)[0])
        pcd = pcd.select_by_index(np.where(np.asarray(pcd.points)[:,0] > -0.4)[0])
        pcd = pcd.select_by_index(np.where(np.asarray(pcd.points)[:,1] < 0.1)[0])
        pcd = pcd.select_by_index(np.where(np.asarray(pcd.points)[:,1] > -0.9)[0])
        pcd_cam = np.asarray(pcd.points)
        row = pcd_cam.shape[0]
        if row == 0:
            return None
        pcd_cam = np.concatenate([pcd_cam, np.ones((row, 1))], axis=1)
        return pcd_cam
    
    def finalize(self, class_map):
        frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1, origin=[0, 0, 0.1])
        grid_map = GridMapRenderer(self.grid_map).build
        total_map = SemanticMapRenderer(class_map).build
        o3d.visualization.draw_geometries([frame, grid_map, total_map])
        for i in range(1, 12):
            image = np.array(self.label_map[:, :, i], dtype=np.uint8)
            cv2.imwrite(os.path.join(cfg.RESULT_PATH, f"label_count_{i}.png"), image)
            image[image != 0] = 255
            cv2.imwrite(os.path.join(cfg.RESULT_PATH, f"label_binary_{i}.png"), image)
        print("saved")
        o3d.io.write_triangle_mesh(os.path.join(cfg.RESULT_PATH, "map_pcl.ply"), grid_map + total_map)

    def show_class_color_map(self, grid_map, class_map):
        print("grid_map", grid_map.shape, grid_map.dtype)
        class_color_map = (255 - cv2.cvtColor(grid_map, cv2.COLOR_GRAY2RGB) * 100).astype(np.uint8)
        for i, color in enumerate(cfg.CTGR_COLOR):
            if i==0:
                continue
            class_color_map[class_map==i] = [color[2] * 255, color[1] * 255, color[0] * 255]
        cv2.imshow("class map", class_color_map)
        cv2.waitKey(10)
        return class_color_map

    def convert_to_semantic_map(self, grid_count):
        class_map = np.argmax(grid_count, axis=2)
        grid_map_mask = np.array([self.grid_map==2], dtype=int)
        class_map = class_map * grid_map_mask[0]
        return class_map


def main(args=None):
    rclpy.init(args=args)
    node = MultiMsgSub()
    rclpy.spin(node)


if __name__ == "__main__":
    main()
