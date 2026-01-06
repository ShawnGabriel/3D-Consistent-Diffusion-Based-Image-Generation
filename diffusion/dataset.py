import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class DepthRGBDataset(Dataset):
    def __init__(self, root_dir):
        """
        root_dir = data/objaverse/rendered
        """
        self.samples = []
        
        for obj in os.listdir(root_dir):
            obj_dir = os.path.join(root_dir, obj)
            if not os.path.isdir(obj_dir):
                continue
            
            for fname in os.listdir(obj_dir):
                if fname.endswith("_rgb.png"):
                    idx = fname.replace("_rgb.png", "")
                    self.samples.append({
                        "rgb": os.path.join(obj_dir, f"{idx}_rgb.png"),
                        "depth": os.path.join(obj_dir, f"{idx}_depth.png"),
                    })
                    
        self.rgb_transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        
        self.depth_transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor()
        ])
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        rgb = Image.open(sample["rgb"]).convert("RGB")
        depth = Image.open(sample["depth"]).convert("L")
        
        rgb = self.rgb_transform(rgb)
        depth = self.depth_transform(depth)
        
        return {
            "pixel_values": rgb,
            "conditioning_pixel_values": depth,
            "prompt": "a photo of an object"
        }
