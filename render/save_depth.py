import numpy as np
import imageio.v2 as imageio

def save_depth_for_controlnet(depth, out_path):
    # Work on a copy so callers keep the original depth untouched
    depth = depth.copy()
    
    # Remove invalid values
    valid = np.isfinite(depth)
    if not np.any(valid):
        raise ValueError("No valid depth values to save.")
    
    min_d = depth[valid].min()
    max_d = depth[valid].max()
    
    depth_norm = np.zeros_like(depth)
    depth_norm[valid] = (depth[valid] - min_d) / (max_d - min_d + 1e-8)
    
    # Invert: ControlNet expects white = near, black = far
    depth_norm = 1.0 - depth_norm
    
    depth_img = (depth_norm * 255).astype(np.uint8)
    imageio.imwrite(out_path, depth_img)
