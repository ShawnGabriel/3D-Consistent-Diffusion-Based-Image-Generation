import os
import re
import glob
import torch
import numpy as np
from PIL import Image
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel

# Config
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

USE_LORA = True
LORA_PATH = "experiments/lora_depth"  # must contain pytorch_lora_weights.bin

OBJECT_DIR = "data/objaverse/rendered/chair"  # change per object
OUT_DIR = "experiments/tile/finetuned" if USE_LORA else "experiments/tile/baseline"

PROMPT = "a photo of an object"
NEG_PROMPT = ""  # keep empty for now

NUM_STEPS = 30
GUIDANCE = 7.5

# Make this deterministic for fair comparisons
SEED = 1234

# Utilities
os.makedirs(OUT_DIR, exist_ok=True)

def load_depth_rgb3(path, size=(512, 512)):
    """
    Returns a torch tensor [1,3,H,W] in [0,1].
    """
    img = Image.open(path).convert("L").resize(size)
    arr = np.array(img).astype(np.float32) / 255.0  # [H,W]
    ten = torch.from_numpy(arr)[None, None, ...]    # [1,1,H,W]
    ten = ten.repeat(1, 3, 1, 1)                    # [1,3,H,W]
    return ten.to(device=DEVICE, dtype=DTYPE)

def tile_depths_horiz(depth_a, depth_b):
    """
    depth_a, depth_b: [1,3,H,W]
    returns: [1,3,H,2W]
    """
    return torch.cat([depth_a, depth_b], dim=3)

def split_wide_image(img_pil):
    """
    Splits a PIL image into left and right halves.
    """
    w, h = img_pil.size
    assert w % 2 == 0, f"Expected even width, got {w}"
    half = w // 2
    left = img_pil.crop((0, 0, half, h))
    right = img_pil.crop((half, 0, w, h))
    return left, right

def natural_key(s):
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", s)]

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

if USE_LORA:
    # Load LoRA adapters saved from training
    pipe.unet.load_attn_procs(LORA_PATH)
    print(f"Loaded LoRA from: {LORA_PATH}")

# Explicit module eval is optional but harmless
pipe.unet.eval()
pipe.controlnet.eval()
pipe.vae.eval()

# Generator for deterministic runs
gen = torch.Generator(device=DEVICE)
gen.manual_seed(SEED)

# Collect depth files
depth_files = sorted(glob.glob(os.path.join(OBJECT_DIR, "*_depth.png")), key=natural_key)
if len(depth_files) < 2:
    raise RuntimeError(f"Need at least 2 depth maps in {OBJECT_DIR}, found {len(depth_files)}")

print(f"Found {len(depth_files)} depth maps")

# We generate pairs: (0,1), (1,2), (2,3), ...
pairs = list(zip(depth_files[:-1], depth_files[1:]))

for i, (a_path, b_path) in enumerate(pairs):
    a_name = os.path.basename(a_path).replace("_depth.png", "")
    b_name = os.path.basename(b_path).replace("_depth.png", "")

    depth_a = load_depth_rgb3(a_path)  # [1,3,512,512]
    depth_b = load_depth_rgb3(b_path)  # [1,3,512,512]
    depth_tiled = tile_depths_horiz(depth_a, depth_b)  # [1,3,512,1024]

    # Important: SD1.5 expects latent width divisible by 8, 1024 is fine
    with torch.no_grad():
        out = pipe(
            prompt=PROMPT,
            negative_prompt=NEG_PROMPT,
            image=depth_tiled,
            num_inference_steps=NUM_STEPS,
            guidance_scale=GUIDANCE,
            generator=gen
        )

    wide = out.images[0]  # PIL (1024x512)
    left, right = split_wide_image(wide)

    # Save outputs
    wide_out = os.path.join(OUT_DIR, f"pair_{i:03d}_{a_name}__{b_name}_WIDE.png")
    left_out = os.path.join(OUT_DIR, f"pair_{i:03d}_{a_name}_LEFT.png")
    right_out = os.path.join(OUT_DIR, f"pair_{i:03d}_{b_name}_RIGHT.png")

    wide.save(wide_out)
    left.save(left_out)
    right.save(right_out)

    print(f"[{i+1}/{len(pairs)}] Saved: {os.path.basename(wide_out)}")

print("Done.")