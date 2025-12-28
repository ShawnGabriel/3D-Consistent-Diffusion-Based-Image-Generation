import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
from save_depth import save_depth_for_controlnet

def load_pointcloud():
    pcd_data = o3d.data.PLYPointCloud()
    pcd = o3d.io.read_point_cloud(pcd_data.path)
    
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=0.1,
            max_nn=30
        )
    )
    return pcd

def get_camera_intrinsics(
    width=512,
    height=512,
    fov_deg=60.0
):
    fx = fy = 0.5 * width / np.tan(np.deg2rad(fov_deg / 2))
    cx = width / 2
    cy = height / 2
    
    intrinsic = o3d.camera.PinholeCameraIntrinsic()
    intrinsic.set_intrinsics(
        width, height, fx, fy, cx, cy
    )
    return intrinsic

def look_at(camera_pos, target=np.array([0, 0, 0]), up=np.array([0, 1, 0])):
    z = target - camera_pos
    z = z / np.linalg.norm(z)
    
    x = np.cross(up, z)
    x = x / np.linalg.norm(x)
    
    y = np.cross(z, x)
    
    pose = np.eye(4)
    pose[:3, :3] = np.stack([x, y, z], axis=1)
    pose[:3, 3] = camera_pos
    
    return np.linalg.inv(pose)

def project_points(points_cam, intrinsics):
    fx, fy, cx, cy = intrinsics
    x = points_cam[:, 0]
    y = points_cam[:, 1]
    z = points_cam[:, 2]

    u = (fx * x / z + cx).astype(np.int32)
    v = (fy * y / z + cy).astype(np.int32)

    return u, v, z


def render_pointcloud(
    pcd,
    camera_pose,
    width=512,
    height=512,
    fov_deg=60.0
):
    points = np.asarray(pcd.points)

    # Transform points to camera space
    homog = np.hstack([points, np.ones((points.shape[0], 1))])
    points_cam = (camera_pose @ homog.T).T[:, :3]

    # Keep points in front of camera
    mask = points_cam[:, 2] > 0
    points_cam = points_cam[mask]

    fx = fy = 0.5 * width / np.tan(np.deg2rad(fov_deg / 2))
    cx, cy = width / 2, height / 2
    intrinsics = (fx, fy, cx, cy)

    u, v, z = project_points(points_cam, intrinsics)

    depth = np.full((height, width), np.inf)

    valid = (
        (u >= 0) & (u < width) &
        (v >= 0) & (v < height)
    )

    u, v, z = u[valid], v[valid], z[valid]

    for ui, vi, zi in zip(u, v, z):
        if zi < depth[vi, ui]:
            depth[vi, ui] = zi

    return depth

def center_pointcloud(pcd):
    points = np.asarray(pcd.points)
    center = points.mean(axis=0)
    pcd.translate(-center)
    return pcd

def show_depth(depth):
    depth_vis = depth.copy()

    valid = np.isfinite(depth_vis)
    if not np.any(valid):
        print("No valid depth values to visualize.")
        return

    depth_vis[~valid] = 0
    max_val = depth_vis.max()

    if max_val > 0:
        depth_vis /= max_val

    plt.imshow(depth_vis, cmap="gray")
    plt.title("Depth Map")
    plt.axis("off")
    plt.show()
    
    
if __name__ == "__main__":
    pcd = load_pointcloud()
    pcd = center_pointcloud(pcd)
    
    camera_pose = look_at(
        camera_pos=np.array([0.0, 0, 2.0]),
    )
    
    points = np.asarray(pcd.points)
    
    depth = render_pointcloud(pcd, camera_pose)
    show_depth(depth)
    
    save_depth_for_controlnet(depth, "depth.png")
    print("Saved depth.png")