import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler

model_id = "XpucT/Deliberate"

# Use the DPMSolverMultistepScheduler (DPM-Solver++) scheduler here instead
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(model_id)
pipe = pipe.to("cuda")
pipe.enable_xformers_memory_efficient_attention()
# pipe.enable_vae_tiling()

prompt = "A digital painting of [blake lively:Ana de Armas:0.8 ] in street cityscape background, happy, full-body, contemporary top, dress, (stocking:1.2), by Artgerm, Guangjian, artstation, soft eyes, extremely detailed face, stunningly beautiful, highly detailed, sharp focus, radiant light rays, cinematic lighting, colorful, volumetric light"
negative_prompt = "ugly, disfigured, deformed, cropped"
image = pipe(prompt, negative_prompt=negative_prompt).images[0]   
image.save("/home/ubuntu/tangdu/images/girl.png")