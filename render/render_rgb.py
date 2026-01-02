import numpy as np

def render_rgb_from_depth(depth, valid_mask):
    """
    Create a simple RGB image from depth.
    Depth is assumed to be positive for visible pixels.
    """
    rgb = np.ones((*depth.shape, 3), dtype=np.float32)
    
    if not np.any(valid_mask):
        return rgb
    
    depth_norm = depth.copy()
    depth_norm[~valid_mask] = 0.0
    
    max_d = depth_norm.max()
    if max_d > 0:
        depth_norm = depth_norm / max_d
        
    gray = 1.0 - depth_norm
    gray[~valid_mask] = 1.0
    
    rgb[..., 0] = gray
    rgb[..., 1] = gray
    rgb[..., 2] = gray
    
    return rgb