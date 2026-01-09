import torch
import numpy as np
from PIL import Image
import os

# Load MiDaS
midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
midas.eval()

transform = torch.hub.load("intel-isl/MiDaS", "transforms").small_transform

def load_depth_gt(path):
    img = Image.open(path).convert("L").resize((512, 512))
    return np.array(img).astype(np.float32) / 255.0

def predict_depth(img_path):
    img = Image.open(img_path).convert("RGB")
    inp = transform(img).unsqueeze(0)
    
    with torch.no_grad():
        pred = midas(inp)
        pred = torch.nn.functional.interpolate(
            pred.unsqueeze(1),
            size=(512, 512),
            mode="bilinear",
            align_corners=False
        ).squeeze().numpy()
        
    pred = (pred - pred.min()) / (pred.max() - pred.min() + 1e-8)
    return pred

def eval_depth(image_dir, depth_dir, label):
    errors = []
    images = sorted([f for f in os.listdir(image_dir) if f.endswith(".png")])
    
    for f in images:
        pred = predict_depth(os.path.join(image_dir, f))
        gt = load_depth_gt(os.path.join(depth_dir, f.replace(".png", "_depth.png")))
        errors.append(np.mean(np.abs(pred - gt)))
        
    print(f"{label} | Mean Depth L1 Error: {np.mean(errors):.4f}")
    
if __name__ == "__main__":
    depth_gt_dir = "data/objaverse/rendered/chair"
    
    eval_depth("experiments/inference/baseline", depth_gt_dir, "Baseline")
    eval_depth("experiments/inference/finetuned", depth_gt_dir, "LoRA")
    eval_depth("experiments/inference/finetuned", depth_gt_dir, "Tiled")