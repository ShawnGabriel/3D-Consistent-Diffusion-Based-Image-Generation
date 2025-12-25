import torch
from diffusers import StableDiffusionPipeline

device = "cuda" if torch.cuda.is_available() else "cpu"

pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5"
)

pipe = pipe.to(device)

image = pipe(
    "a photo of a chair in a studio, high quality",
    num_inference_steps=20
).images[0]

image.save("sd_test.png")
print("Saved sd_test.png on", device)