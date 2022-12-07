import numpy as np
import config as cfg
import open3d as o3d
import copy
import cv2

from build_map import SemanticMapRenderer

class DepthToMap():
    def __init__(self, map, odom, depth, segmap):
        # TODO : intrinsic -> cfg
        self.intrinsic = o3d.camera.PinholeCameraIntrinsic(848, 480, 418.874, 418.874, 427.171, 239.457)  #sch_robot
        # self.intrinsic = o3d.camera.PinholeCameraIntrinsic(424, 240, 213.0231, 213.0231, 213.6875, 116.998) #LGE_robot
        self.ir_to_rgb = cfg.IR2RGB
        self.cam_to_base = cfg.CAM_TO_BASE
        self.grid_map = map
        self.grid_label_count = np.zeros((map.shape[0], map.shape[1], 7), dtype=int)
        self.accumulated_map = []
        self.grid_per_classpoint = np.zeros(cfg.NUM_CLASS, dtype=int)
        self.mapping(odom, depth, segmap)

    def mapping(self, pose, depth, segmap):
        default_matrix = self.default_tf_matrix()
        base_to_glb = self.pose_to_matrix(pose)
        pcd_ir = self.get_point_cloud(depth)
        class_map = self.convert_classmap(segmap)
        pcd_label = self.align_segmap_to_pcd(class_map, pcd_ir)
        pcd_glb = self.transform_to_global(pcd_ir, base_to_glb)
        pcd_glb = self.default_transform(pcd_glb, default_matrix)
        self.grid_label_count += self.count_label(self.grid_label_count, pcd_glb, pcd_label)
        return self.grid_label_count
        
    def get_point_cloud(self, depth):
        depth_image = o3d.geometry.Image((depth).astype(np.uint16))
        pcd = o3d.geometry.PointCloud.create_from_depth_image(depth_image, self.intrinsic)
        pcd = pcd.select_by_index(np.where(np.asarray(pcd.points)[:,2] < 2)[0])
        pcd = pcd.select_by_index(np.where(np.asarray(pcd.points)[:,0] < 0.4)[0])
        pcd = pcd.select_by_index(np.where(np.asarray(pcd.points)[:,0] > -0.4)[0])
        # pcd = pcd.select_by_index(np.where(np.asarray(pcd.points)[:,1] < 0.1)[0])
        # pcd = pcd.select_by_index(np.where(np.asarray(pcd.points)[:,1] > -0.9)[0])
        return pcd

    def convert_classmap(self, segmap):
        lb_map = {el['trainId']: [int(color*255) for color in el['color']] for el in cfg.SEGLABELCOLOR}
        class_map = np.zeros(segmap.shape[:2], dtype=int)
        for i, c in lb_map.items():
            class_map[(segmap==c).all(axis=2)] = i
        return class_map

    def default_tf_matrix(self):
        matrix = np.identity(4)
        angle = np.deg2rad(62)
        quaternion = np.array([np.cos(angle/2.), 0.0, 0.0, -np.sin(angle/2.)])
        assert np.isclose(np.linalg.norm(quaternion), 1, 0.001), f"[pose_to_matrix] quaterion norm={np.linalg.norm(quaternion)}"
        matrix[:3, :3] = o3d.geometry.get_rotation_matrix_from_quaternion(quaternion)
        # matrix[:3, 3] = np.array([0.14854032641747172, 0.019878675544998736, 0.0])        
        matrix[:3, 3] = np.array([-2.9854032641747172, -2.7878675544998736, 0.0])
        # matrix[:3, 3] = cfg.GRID_ORIGIN
        matrix = np.linalg.inv(matrix)
        return matrix

    def default_transform(self, pcd, matrix):
        pcd_in_origin = copy.deepcopy(pcd)
        pcd_in_origin.transform(matrix)
        return pcd_in_origin

    def pose_to_matrix(self, pose):
        matrix = np.identity(4)
        quaternion = np.array([pose.orientation.w, pose.orientation.x, pose.orientation.y, pose.orientation.z])
        assert np.isclose(np.linalg.norm(quaternion), 1, 0.0001), f"[pose_to_matrix] quaterion norm={np.linalg.norm(quaternion)}"
        matrix[:3, :3] = o3d.geometry.get_rotation_matrix_from_quaternion(quaternion)
        matrix[:3, 3] = np.array([pose.position.x, pose.position.y, pose.position.z])
        return matrix

    def transform_to_global(self, pcd_in_cam, base_to_glb):
        pcd_in_rbt = copy.deepcopy(pcd_in_cam)
        pcd_in_rbt.transform(self.cam_to_base)
        pcd_in_glb = copy.deepcopy(pcd_in_rbt)
        pcd_in_glb.transform(base_to_glb)
        return pcd_in_glb

    def align_segmap_to_pcd(self, class_map, pcd_ir):
        height, width = class_map.shape
        pcd_rgb = copy.deepcopy(pcd_ir)
        pcd_rgb.transform(self.ir_to_rgb)
        points = np.asarray(pcd_rgb.points) 
        fx, fy = self.intrinsic.get_focal_length()
        cx, cy = self.intrinsic.get_principal_point()
        X, Y, Z = points[..., 0], points[..., 1], points[..., 2]
        pixel = np.stack([fy * Y / Z + cy, fx * X / Z + cx], axis=1).astype(int)
        outside_img = np.array([pixel[:, 0] < 0, pixel[:, 0] >= height, pixel[:, 1] < 0, pixel[:, 1] >= width]).any(axis=0)
        pixel = np.clip(pixel, [0, 0], [height - 1, width - 1])
        pcd_label = class_map[pixel[..., 0], pixel[..., 1]]
        pcd_label[outside_img] = -1
        return pcd_label

    def count_label(self, label_count_map, pcd, label):
        grid_yx = (np.asarray(pcd.points) - cfg.GRID_ORIGIN) / cfg.GRID_RESOLUTION
        grid_yx = grid_yx.astype(int)
        valid_mask = np.array([grid_yx[:, 1] < label_count_map.shape[1], grid_yx[:, 0] < label_count_map.shape[0]], dtype=int).all(axis=0)    # TODO self.grid_map - and
        grid_yx = grid_yx[valid_mask]
        label = label[valid_mask]
        label_count_map[grid_yx[:,0],grid_yx[:,1],label] += 1
        return label_count_map
