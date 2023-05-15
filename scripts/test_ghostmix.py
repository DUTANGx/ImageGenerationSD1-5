import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler

model_id = "/home/ubuntu/tangdu/GhostMix"

# Use the DPMSolverMultistepScheduler (DPM-Solver++) scheduler here instead
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16, use_safetensors=False)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe = pipe.to("cuda")
pipe.enable_xformers_memory_efficient_attention()
pipe.enable_vae_tiling()

prompt = "(masterpiece, top quality, best quality, official art, beautiful and aesthetic:1.2), (1girl:1.3), (fractal art:1.3), highres, ultra detailed, a small house under sea"
negative_prompt = "EasyNegative, (worst quality, low quality:1.4), (blush:1.3), nude, text, username, watermark,nsfw, aidv1-neg, animestylenegativeembedding_dreamshaper, bad-artist, bad-artist-anime, sketch by bad-artist, painting by bad-artist, photograph by bad-artist, bad-picture-chill-75v, badhandv4, badhandv5, badv3, badv4, badv5, bad_prompt, bad_prompt_version2, EasyNegative, ng_deepnegative_v1_75t, verybadimagenegative_v1.2-6400, verybadimagenegative_v1.3, Unspeakable-Horrors-Composition-4v,(worst quality, low quality:1.4), (raw photo)"
image = pipe(prompt, negative_prompt=negative_prompt, width=512, height=768, guidance_scale=6.5).images[0]   
image.save("/home/ubuntu/tangdu/images/ghost_mix.png")