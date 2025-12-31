import sys
import os

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../..")
)
sys.path.append(PROJECT_ROOT)

import os
import numpy as np
import open3d as o3d

from render.render_depth import(
    render_pointcloud,
    look_at,
    center_pointcloud,
    align_pointcloud_up_axis
)

from render.save_depth import save_depth_for_controlnet


def get_camera_positions(radius=2.0):
    """
    Four cameras evenly spaced around the object (Y-up).
    """
    angles = [0, 90, 180, 270]
    positions = []
    
    for a in angles:
        theta = np.deg2rad(a)
        x = radius * np.cos(theta)
        z = radius * np.sin(theta)
        y = 0.0
        
        positions.append(np.array([x, y, z]))
        
    return positions


def render_object_views(pcd_path, object_name):
    print(f"\nRendering views for {object_name}")
    
    pcd = o3d.io.read_point_cloud(pcd_path)
    pcd = center_pointcloud(pcd)
    pcd = align_pointcloud_up_axis(pcd)
    
    out_dir = os.path.join(
        "data/objaverse/rendered",
        object_name
    )
    os.makedirs(out_dir, exist_ok=True)
    
    camera_positions = get_camera_positions(radius=2.0)
    
    for i, cam_pos in enumerate(camera_positions):
        pose = look_at(
            camera_pos=cam_pos,
            target=np.array([0.0, 0.0, 0.0])
        )
        
        depth = render_pointcloud(pcd, pose)
        
        depth_path = os.path.join(
            out_dir, f"view_{i}_depth.png"
        )
        pose_path = os.path.join(
            out_dir, f"view_{i}_pose.npy"
        )
        
        save_depth_for_controlnet(depth, depth_path)
        np.save(pose_path, pose)
        
        print(f"Saved view {i}")
        

def main():
    pcd_dir = "data/objaverse/pointclouds"
    
    for fname in os.listdir(pcd_dir):
        if not fname.endswith(".ply"):
            continue
        
        object_name = os.path.splitext(fname)[0]
        pcd_path = os.path.join(pcd_dir, fname)
        
        render_object_views(pcd_path, object_name)
        

if __name__ == "__main__":
    main()