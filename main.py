import torch
from diffusers import StableDiffusionPipeline

device = "cuda"
model_id = "stabilityai/stable-diffusion-2-1-base"
pipe = StableDiffusionPipeline.from_pretrained(
    model_id, torch_dtype=torch.float16, revision="fp16"
)
vae = pipe.components["vae"].to(device)
tolenizer = pipe.components["tokenizer"]
text_encoder = pipe.components["text_encoder"].to(device)
unet = pipe.components["unet"].to(device.to(device.to(device.to(device))))
scheduler = pipe.components["scheduler"]
