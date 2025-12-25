import open3d as o3d

pcd = o3d.data.PLYPointCloud()
pointcloud = o3d.io.read_point_cloud(pcd.path)

o3d.visualization.draw_geometries([pointcloud])
