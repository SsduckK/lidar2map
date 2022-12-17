import numpy as np
import open3d as o3d
import copy
import cv2

import create_map.config as cfg
from create_map.build_map import SemanticMapRenderer


class DepthToMap():
    def __init__(self, map, odom, points_opt, segmap):
        # TODO : intrinsic -> rgb camera
        self.rgb_intrinsic = o3d.camera.PinholeCameraIntrinsic(480, 640, 713.52, 715.21, 224.48, 340.72)  #sch_robot
        self.ir_to_rgb = cfg.IR2RGB
        self.grid_map = map
        self.grid_label_count = np.zeros((map.shape[0], map.shape[1], 12), dtype=int)
        self.accumulated_map = []
        self.grid_label_count, self.points_glb = self.mapping(odom, points_opt, segmap)

    def mapping(self, pose, points_opt, segmap):
        default_matrix = self.default_tf_matrix()
        base_to_glb = self.pose_to_matrix(pose)
        class_map = self.convert_classmap(segmap)
        label = self.align_segmap_to_pcd(class_map, points_opt)

        points_rbt = self.transform_point_cloud(points_opt, cfg.IR_TO_ROBOT)
        points_glb = self.transform_point_cloud(points_rbt, base_to_glb)
        #points_glb = self.transform_point_cloud(points_glb, default_matrix)
        self.grid_label_count += self.count_label(self.grid_label_count, points_glb, label)
        print("valid label:", np.sum(label >= 0))
        return self.grid_label_count, points_glb

    def convert_classmap(self, segmap):
        lb_map = {el['trainId']: [int(color*255) for color in el['color']] for el in cfg.SEGLABELCOLOR}
        class_map = np.zeros(segmap.shape[:2], dtype=int)
        for i, c in lb_map.items():
            class_map[(segmap==c).all(axis=2)] = i+1
        return class_map

    def default_tf_matrix(self):
        matrix = np.identity(4)
        angle = np.deg2rad(0)
        quaternion = np.array([np.cos(angle/2.), 0.0, 0.0, -np.sin(angle/2.)])
        assert np.isclose(np.linalg.norm(quaternion), 1, 0.001), f"[pose_to_matrix] quaterion norm={np.linalg.norm(quaternion)}"
        matrix[:3, :3] = o3d.geometry.get_rotation_matrix_from_quaternion(quaternion)
        # matrix[:3, 3] = np.array([0.14854032641747172, 0.019878675544998736, 0.0])        
        # matrix[:3, 3] = np.array([-2.9854032641747172, -2.7878675544998736, 0.0])
        # matrix[:3, 3] = cfg.GRID_ORIGIN
        matrix = np.linalg.inv(matrix)
        return matrix

    def transform_point_cloud(self, points, matrix):
        points_tfm = np.transpose(matrix@np.transpose(points))
        return points_tfm

    def pose_to_matrix(self, pose):
        matrix = np.identity(4)
        quaternion = np.array([pose.orientation.w, pose.orientation.x, pose.orientation.y, pose.orientation.z])
        assert np.isclose(np.linalg.norm(quaternion), 1, 0.0001), f"[pose_to_matrix] quaterion norm={np.linalg.norm(quaternion)}"
        matrix[:3, :3] = o3d.geometry.get_rotation_matrix_from_quaternion(quaternion)
        matrix[:3, 3] = np.array([pose.position.x, pose.position.y, pose.position.z])
        return matrix

    def align_segmap_to_pcd(self, class_map, points_ir):
        height, width = class_map.shape
        points_rgb = np.transpose(self.ir_to_rgb@np.transpose(points_ir))
        fx, fy = self.rgb_intrinsic.get_focal_length()
        cx, cy = self.rgb_intrinsic.get_principal_point()
        X, Y, Z = points_rgb[..., 0], points_rgb[..., 1], points_rgb[..., 2]
        pixel = np.stack([fy * Y / Z + cy, fx * X / Z + cx], axis=1).astype(int)
        outside_img = np.array([pixel[:, 0] < 0, pixel[:, 0] >= height, pixel[:, 1] < 0, pixel[:, 1] >= width]).any(axis=0)
        pixel = np.clip(pixel, [0, 0], [height - 1, width - 1])
        pcd_label = class_map[pixel[..., 0], pixel[..., 1]]
        pcd_label[outside_img] = -1
        return pcd_label

    def count_label(self, label_count_map, points, label):
        grid_yx = np.stack([-points[:, 1] + cfg.GRID_ORIGIN[1], points[:, 0] - cfg.GRID_ORIGIN[0]], axis=1) / cfg.GRID_RESOLUTION
        grid_yx = grid_yx.astype(int)
        valid_mask = np.array([grid_yx[:, 1] > 0,
                               grid_yx[:, 0] > 0,
                               grid_yx[:, 1] < label_count_map.shape[1], 
                               grid_yx[:, 0] < label_count_map.shape[0], 
                               label >= 0], dtype=int).all(axis=0)    # TODO self.grid_map - and
        grid_yx = grid_yx[valid_mask]
        label = label[valid_mask]
        label_count_map[grid_yx[:,0],grid_yx[:,1],label] += 1
        return label_count_map
