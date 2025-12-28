import torch
from diffusers import StableDiffusionPipeline
from PIL import Image
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"

pipe = StableDiffusionPipeline.from_pretrained(
    "sudo-ai/zero123plus-v1.2",
    torch_dtype=torch.float32
).to(device)

# Input image (single view)
image = Image.open("./render/depth.png").convert("RGB").resize((256, 256))

# Example: relative camera pose encoded as text or conditioning
prompt = "a photo of the same object from a different viewpoint"

result = pipe(
    prompt=prompt,
    image=image,
    num_inference_steps=30,
    guidance_scale=9
).images[0]

result.save("zero123_result.png")
print("Saved zero123_result.png")