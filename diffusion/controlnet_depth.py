import torch
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"

controlnet = ControlNetModel.from_pretrained(
    "lllyasviel/sd-controlnet-depth",
    torch_dtype=torch.float32
)

pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    controlnet=controlnet,
    torch_dtype=torch.float32,
    safety_checker=None
)

pipe = pipe.to(device)

depth_image = Image.open("./render/depth.png").convert("L")

prompt = "a photo of a chair, studio lighting, high quality"

image = pipe(
    prompt,
    image=depth_image,
    num_inference_steps=30,
    guidance_scale=9,
    controlnet_conditioning_scale=1.0
).images[0]

image.save("controlnet_result1.png")
print("Saved controlnet_result1.png")