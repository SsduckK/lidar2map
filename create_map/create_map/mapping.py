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


START_TIME_221216 = 1671186576547699200
END_TIME_221216 = 1671186696404034560


class GridMapClassifier(Node):
    def __init__(self):
        super().__init__('grid_map_classifier')
        self.map_, self.map_shape = np.zeros((1, 1)), 0
        self.sub_data_heap = {key: {"time": np.zeros(0), "data": []} for key in ["odom", "lidar", "depth"]}
        self.tolerance = 1e+8
        # self.tolerance = 5251135165601792.0
        self.br = CvBridge()
        self.grid_map = None
        self.num_ctgr = 12       # TODO: read ros parameter
        self.label_map = None
        self.callback_count = 0
        self.use_depth = True
        self.use_lidar = False
        self.mapping_on = False
        self.latest_time = 0
        self.count_thresh = 0
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
        if not os.path.isdir(cfg.RESULT_PATH):
            os.makedirs(cfg.RESULT_PATH, exist_ok=True)
        # self.vis = o3d.visualization.Visualizer()
        # self.vis.create_window()
        # self.vis.get_view_control()

    def map_callback(self, map):
        map = self.br.imgmsg_to_cv2(map)
        self.grid_map = map[..., 0]
        self.label_map = np.zeros((map.shape[0], map.shape[1], self.num_ctgr), dtype=int) if self.label_map is None else self.label_map
    
    def update_data(self, name, nano_sec, data):
        self.sub_data_heap[name]["time"] = np.append(self.sub_data_heap[name]["time"], np.asarray(nano_sec))
        self.sub_data_heap[name]["data"].append(data)
        self.sub_data_heap[name]["time"] = self.sub_data_heap[name]["time"][-100:]
        self.sub_data_heap[name]["data"] = self.sub_data_heap[name]["data"][-100:]
        
    def odom_callback(self, odom):
        nano_sec = Time.from_msg(odom.header.stamp).nanoseconds
        self.update_data("odom", nano_sec, odom.pose.pose)

    def lidar_callback(self, lidar):
        self.latest_time = Time.from_msg(lidar.header.stamp).nanoseconds
        if START_TIME_221216 < self.latest_time < END_TIME_221216:
            self.mapping_on = True
            self.update_data("lidar", self.latest_time, lidar.ranges)
        
    def depth_callback(self, depth):
        self.latest_time = Time.from_msg(depth.header.stamp).nanoseconds
        print("time(s):", (self.latest_time - START_TIME_221216) // 1e9)
        if START_TIME_221216 < self.latest_time < END_TIME_221216:
            self.mapping_on = True
            self.update_data("depth", self.latest_time, self.br.imgmsg_to_cv2(depth))
        
    def segmap_callback(self, segmap):
        if self.mapping_on is False or self.grid_map is None:
            return
        segmap_time = Time.from_msg(segmap.header.stamp).nanoseconds
        segmap = self.br.imgmsg_to_cv2(segmap)
        sync_odom, odom_diff = self.sync_data(segmap_time, self.sub_data_heap["odom"]["time"], self.sub_data_heap["odom"]["data"])
        sync_depth, depth_diff = self.sync_data(segmap_time, self.sub_data_heap["depth"]["time"], self.sub_data_heap["depth"]["data"])
        sync_lidar, lidar_diff = self.sync_data(segmap_time, self.sub_data_heap["lidar"]["time"], self.sub_data_heap["lidar"]["data"])
        self.get_logger().info(f"sync_data: {odom_diff}, {depth_diff}, {lidar_diff}")
        if odom_diff < self.tolerance and lidar_diff < self.tolerance:
            class_color_map = self.update_map(self.grid_map, sync_odom, sync_depth, sync_lidar, segmap)
            if self.latest_time > END_TIME_221216 - 1e9:
                self.finalize(class_color_map)


    def sync_data(self, segmap_time, other_time, other_list):
        if len(other_list) == 0:
            return None, 0
        other_index = np.argmin(abs(other_time - segmap_time))
        time_diff = np.min(abs(other_time - segmap_time))
        sync_other = other_list[other_index]
        return sync_other, time_diff

    def update_map(self, grid_map, odom, depth, lidar, segmap):
        cv2.imshow("seg_map", segmap)
        cv2.waitKey(10)
        if lidar is not None:
            lidar_pts = self.lidar_to_point_cloud(lidar)
            lidar_map = DepthToMap(grid_map, odom, lidar_pts, segmap)
            self.label_map += lidar_map.grid_label_count
        if depth is not None:
            depth_pts = self.depth_to_point_cloud(depth)
            if depth_pts is not None:
                depth_map = DepthToMap(grid_map, odom, depth_pts, segmap)
                self.label_map += depth_map.grid_label_count
                print("grid count frame", np.sum(depth_map.grid_label_count))
            else:
                print("no points from depth map!!")
        
        class_map = self.convert_to_semantic_map(self.label_map)
        class_color_map = self.show_class_color_map(grid_map, class_map)
        self.callback_count += 1
        print("callback count:", self.callback_count)
        return class_color_map

    def convert_to_semantic_map(self, grid_count):
        print("grid count accum", np.sum(grid_count))
        class_map = np.argmax(grid_count, axis=2)
        class_mask = np.max(grid_count, axis=2) > self.count_thresh
        # grid_map_mask = np.array([self.grid_map==cfg.GRID_MAP_VALUE["wall"]], dtype=int)
        class_map = class_map * class_mask
        return class_map

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

    def show_class_color_map(self, grid_map, class_map):
        class_color_map = cv2.cvtColor(grid_map, cv2.COLOR_GRAY2RGB)
        for i, color in enumerate(cfg.CTGR_COLOR):
            if i==0:
                continue
            class_color_map[class_map==i] = (np.array(color)*255).astype(np.uint8)[::-1]
        
        class_view = cv2.resize(class_color_map, (int(class_map.shape[1]*3), int(class_map.shape[0]*3)), cv2.INTER_NEAREST)
        cv2.imshow("class map", class_view)
        cv2.waitKey(10)
        return class_color_map

    def finalize(self, class_color_map):
        cv2.imwrite(os.path.join(cfg.RESULT_PATH, f"class_color_map.png"), class_color_map)
        np.save(os.path.join(cfg.RESULT_PATH, f"label_count_map.npy"), self.label_map)
        # frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1, origin=[0, 0, 0.1])
        # grid_map = GridMapRenderer(self.grid_map).build
        # total_map = SemanticMapRenderer(class_map).build
        # o3d.visualization.draw_geometries([frame, grid_map, total_map])
        # for i in range(1, 12):
        #     image = np.array(self.label_map[:, :, i], dtype=np.uint8)
        #     cv2.imwrite(os.path.join(cfg.RESULT_PATH, f"label_count_{i}.png"), image)
        #     image[image != 0] = 255
        #     cv2.imwrite(os.path.join(cfg.RESULT_PATH, f"label_binary_{i}.png"), image)
        # print("saved")
        # o3d.io.write_triangle_mesh(os.path.join(cfg.RESULT_PATH, "map_pcl.ply"), grid_map + total_map)


def main(args=None):
    rclpy.init(args=args)
    node = GridMapClassifier()
    rclpy.spin(node)


if __name__ == "__main__":
    main()
