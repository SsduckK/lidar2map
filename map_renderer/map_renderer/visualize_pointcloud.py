import open3d as o3d

pcd = o3d.io.read_triangle_mesh("/home/ri/colcon_ws/src/lidar2map/data/1216_depth_v1/grid_pcl.ply")
o3d.visualization.draw_geometries([pcd])