import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler

model_id = "SG161222/Realistic_Vision_V2.0"

# Use the DPMSolverMultistepScheduler (DPM-Solver++) scheduler here instead
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16, resume_download=True, use_safetensors=False)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe = pipe.to("cuda")
pipe.enable_xformers_memory_efficient_attention()
# pipe.enable_vae_tiling()

prompt = "RAW photo, a close up portrait photo of 26 y.o woman in wastelander clothes, long haircut, pale skin, slim body, background is city ruins, (high detailed skin:1.2), 8k uhd, dslr, soft lighting, high quality, film grain, Fujifilm XT3"
negative_prompt = "nude, NSFW, (deformed iris, deformed pupils, semi-realistic, cgi, 3d, render, sketch, cartoon, drawing, anime:1.4), text, close up, cropped, out of frame, worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers, long neck"
image = pipe(prompt, negative_prompt=negative_prompt).images[0]   
image.save("/home/ubuntu/tangdu/images/girl.png")