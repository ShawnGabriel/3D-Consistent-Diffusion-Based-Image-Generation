import os
import torch
import numpy as np
from PIL import Image
from diffusers import (
    StableDiffusionControlNetPipeline,
    ControlNetModel
)

# Config
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

USE_LORA = True              # toggle this
LORA_PATH = "experiments/lora_depth"

DEPTH_DIR = "data/objaverse/rendered/chair"
OUT_DIR = "experiments/inference/finetuned" if USE_LORA else "experiments/inference/baseline"
PROMPT = "a photo of an object"

os.makedirs(OUT_DIR, exist_ok=True)

# Load pipeline
controlnet = ControlNetModel.from_pretrained(
    "lllyasviel/sd-controlnet-depth",
    torch_dtype=DTYPE
)

pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    controlnet=controlnet,
    torch_dtype=DTYPE
).to(DEVICE)

pipe.enable_attention_slicing()

# Load LoRA if needed
if USE_LORA:
    pipe.unet.load_attn_procs(LORA_PATH)
    print("Loaded LoRA weights")
    
pipe.eval()

# Helper function
def load_depth(path):
    depth = Image.open(path).convert("L").resize((512, 512))
    depth = torch.from_numpy(np.array(depth)).float() / 255.0
    depth = depth.unsqueeze(0).unsqueeze(0) # [1,1,H,W]
    depth = depth.repeat(1, 3, 1, 1) # [1,3,H,W]
    return depth.to(DEVICE, dtype=DTYPE)

# Inference loop
for fname in sorted(os.listdir(DEPTH_DIR)):
    if not fname.endswith("_depth.png"):
        continue
    
    depth_path = os.path.join(DEPTH_DIR, fname)
    depth = load_depth(depth_path)
    
    with torch.no_grad():
        image = pipe(
            prompt=PROMPT,
            image=depth,
            num_inference_steps=30,
            guidance_scale=7.5
        ).images[0]
        
    out_path = os.path.join(
        OUT_DIR,
        fname.replace("_depth.png", ".png")
    )
    image.save(out_path)
    print(f"Saved {out_path}")