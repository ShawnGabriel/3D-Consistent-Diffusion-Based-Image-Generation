import torch
from torch.utils.data import DataLoader
from diffusers import (
    StableDiffusionControlNetPipeline,
    ControlNetModel
)
from peft import LoraConfig, get_peft_model
from dataset import DepthRGBDataset


device = "cuda" if torch.cuda.is_available() else "cpu"

controlnet = ControlNetModel.from_pretrained(
    "lllyasviel/sd-controlnet-depth",
    torch_dtype=torch.float16
)

pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    controlnet=controlnet,
    torch_dtype=torch.float16
).to(device)

pipe.vae.requires_grad_(False)
pipe.text_encoder.requires_grad_(False)
pipe.controlnet.requires_grad_(False)
pipe.unet.requires_grad_(False)


lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["to_q", "to_k", "to_v"],
    lora_dropout=0.05,
    bias="none",
)

pipe.unet = get_peft_model(pipe.unet, lora_config)
pipe.unet.train()
pipe.unet.print_trainable_parameters()

dataset = DepthRGBDataset("data/objaverse/rendered")

dataloader = DataLoader(
    dataset,
    batch_size=1,
    shuffle=True
)

optimizer = torch.optim.AdamW(
    pipe.unet.parameters(),
    lr=1e-4
)

num_steps = 500
pipe.scheduler.set_timesteps(1000)

for step, batch in enumerate(dataloader):
    if step >= num_steps:
        break

    rgb = batch["pixel_values"].to(device)
    depth = batch["conditioning_pixel_values"].to(device)
    prompt = batch["prompt"]

    # Encode RGB → latent
    latents = pipe.vae.encode(rgb).latent_dist.sample()
    latents = latents * pipe.vae.config.scaling_factor

    # Noise
    noise = torch.randn_like(latents)
    timesteps = torch.randint(
        0,
        pipe.scheduler.config.num_train_timesteps,
        (latents.shape[0],),
        device=device
    ).long()

    noisy_latents = pipe.scheduler.add_noise(
        latents, noise, timesteps
    )

    # Text embedding
    text_embeds = pipe._encode_prompt(
        prompt,
        device,
        num_images_per_prompt=1,
        do_classifier_free_guidance=False
    )

    # UNet forward
    noise_pred = pipe.unet(
        noisy_latents,
        timesteps,
        encoder_hidden_states=text_embeds,
        controlnet_cond=depth,
        return_dict=False
    )[0]

    loss = torch.nn.functional.mse_loss(noise_pred, noise)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step % 50 == 0:
        print(f"Step {step} | Loss {loss.item():.4f}")

pipe.unet.save_pretrained("experiments/lora_depth")
