# 导入Flask库
from flask import Flask, jsonify, request
import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
import pickle
import urllib
import json
from translate import translate


# 初始化应用程序
app = Flask(__name__)
file_name = ""

# 为API定义路由
@app.route('/api/image_genarate', methods=['post'])
def get_users():
    data = request.json
    prompt = data["prompt"]
    image = generate_image(prompt)
    file_name = "/home/ubuntu/stable_diffusion/images/image.png" 
    image.save(file_name)

    output = {"image_count": len(image), "file_path": file_name}

    return file_name
    # return data


def generate_image(text):
    # model_id = "prompthero/openjourney"
    model_id = "SG161222/Realistic_Vision_V2.0"

    height = 512                        # default height of Stable Diffusion
    width = 512                         # default width of Stable Diffusion
    num_inference_steps = 50            # Number of denoising steps
    guidance_scale = 7.5                # Scale for classifier-free guidance
    num_samples = 1
    num_rows = 1

    # Use the DPMSolverMultistepScheduler (DPM-Solver++) scheduler here instead
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16, resume_download=True, use_safetensors=False)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to("cuda")
    pipe.enable_xformers_memory_efficient_attention()

    generator = torch.Generator("cuda").manual_seed(2022)

    negative_prompt = "nude, NSFW, (deformed iris, deformed pupils, semi-realistic, cgi, 3d, render, sketch, cartoon, drawing, anime:1.4), text, close up, cropped, out of frame, worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers, long neck"
    text = translate(text)
    image = pipe(prompt=text,
                 negative_prompt=negative_prompt,
                 num_images_per_prompt=num_samples,
                 height=height,
                 width=width,
                 num_inference_steps=num_inference_steps,
                 guidance_scale=guidance_scale,
                 generator=generator,
                 ).images[0]
    
    return image


# 启动应用程序
if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=8000)

