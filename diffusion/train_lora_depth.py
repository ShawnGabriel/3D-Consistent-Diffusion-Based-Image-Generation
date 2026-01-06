import torch
from torch.utils.data import DataLoader
from diffusers import (
    StableDiffusionControlNetPipeline,
    ControlNetModel
)
from peft import LoraConfig, get_peft_model
from dataset import DepthRGBDataset
import itertools

# Used for testing by setting DRY_RUN = True
DRY_RUN = False

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if device == "cuda" else torch.float32

controlnet = ControlNetModel.from_pretrained(
    "lllyasviel/sd-controlnet-depth",
    torch_dtype=dtype
)

pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    controlnet=controlnet,
    torch_dtype=dtype
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

num_steps = 5 if DRY_RUN else 200
pipe.scheduler.set_timesteps(1000)

for step, batch in enumerate(itertools.cycle(dataloader)):
    if step >= num_steps:
        break

    rgb = batch["pixel_values"].to(device).to(dtype)
    depth = batch["conditioning_pixel_values"].to(device).to(dtype)
    
    if depth.shape[1] == 1:
        depth = depth.repeat(1, 3, 1, 1)
        
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

    prompt_embeds, _ = pipe.encode_prompt(
        prompt,
        device=device,
        num_images_per_prompt=1,
        do_classifier_free_guidance=False
    ) 

    # UNet forward
    down_block_res_samples, mid_block_res_sample = pipe.controlnet(
        noisy_latents,
        timesteps,
        encoder_hidden_states=prompt_embeds,
        controlnet_cond=depth,
        return_dict=False,
    )

    noise_pred = pipe.unet(
        noisy_latents,
        timesteps,
        encoder_hidden_states=prompt_embeds,
        down_block_additional_residuals=down_block_res_samples,
        mid_block_additional_residual=mid_block_res_sample,
        return_dict=False,
    )[0]


    loss = torch.nn.functional.mse_loss(noise_pred, noise)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step % 50 == 0:
        print(f"Step {step} | Loss {loss.item():.4f}")

pipe.unet.save_pretrained("experiments/lora_depth")
