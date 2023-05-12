import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler

model_id = "~/.cache/huggingface/hub/models--sunnyweir--ghostmix_v12/snapshots/09dae8ad9f24615580286f54c198bcceec78752e/"

# Use the DPMSolverMultistepScheduler (DPM-Solver++) scheduler here instead
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(model_id)
pipe = pipe.to("cuda")
pipe.enable_xformers_memory_efficient_attention()
# pipe.enable_vae_tiling()

prompt = "8k, sfw, real picture, intricate details, ultra-detailed,(photorealistic, hyper-realistic:1.2),absurdres(best quality:1.3),(masterpiece:1.1),(illustration:1.2),(ultra-detailed:1.2), (Extremely detailed background:1.1), navel:1.5, red hair, jizi, <lora:jizi-000010:0.6>, makeup, yellow eyes, big eyes, beautiful Detailed Eyes, <lora:beautifulDetailedEyes_v10:1>"
negative_prompt = "text, username, watermark,nsfw, aidv1-neg, animestylenegativeembedding_dreamshaper, bad-artist, bad-artist-anime, sketch by bad-artist, painting by bad-artist, photograph by bad-artist, bad-picture-chill-75v, badhandv4, badhandv5, badv3, badv4, badv5, bad_prompt, bad_prompt_version2, EasyNegative, ng_deepnegative_v1_75t, verybadimagenegative_v1.2-6400, verybadimagenegative_v1.3, Unspeakable-Horrors-Composition-4v,(worst quality, low quality:1.4), (raw photo)"
image = pipe(prompt, negative_prompt=negative_prompt).images[0]   
image.save("/home/ubuntu/tangdu/images/ghost_mix.png")