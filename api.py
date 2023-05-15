# 导入Flask库
from flask import Flask, jsonify, request
import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
import pickle
import urllib
import json
from translate import translate
from datetime import datetime

realistic_pipe = StableDiffusionPipeline.from_pretrained("SG161222/Realistic_Vision_V2.0", torch_dtype=torch.float16, resume_download=True, use_safetensors=False)
ghost_mix_pipe = StableDiffusionPipeline.from_pretrained("/home/ubuntu/tangdu/GhostMix", torch_dtype=torch.float16, resume_download=True, use_safetensors=False)


# 初始化应用程序
app = Flask(__name__)
file_name = ""

# 为API定义路由
@app.route('/api/image_genarate', methods=['post'])
def get_users():
    data = request.json
    prompt = data["prompt"]
    model = data["model"]
    if model == "" or model == "Realistic":
        model_id = "Realistic"
    elif model == "GhostMix":
        model_id = "GhostMix"
    image = generate_image(prompt, model_id)
    timestamp = int(datetime.now().timestamp())
    file_name = "/home/ubuntu/stable_diffusion/images/{}.png".format(timestamp)
    image.save(file_name)
    file_link = "http://124.222.40.123:8001/image?{}.png".format(timestamp)

    output = {"file_link": file_link}

    return output
    # return data


def generate_image(text, model="Realistic"):
    # model_id = "prompthero/openjourney"
    # model_id = "SG161222/Realistic_Vision_V2.0"
    if model == "Realistic":
        pipe = realistic_pipe
        height = 768                        # default height of Stable Diffusion
        width = 768                         # default width of Stable Diffusion
        
    elif model == "GhostMix":
        pipe = ghost_mix_pipe
        height = 768                        # default height of Stable Diffusion
        width = 512                         # default width of Stable Diffusion

    image = pipeline(text, pipe, height, width)
    return image


def pipeline(text, pipe, height, width):
    num_inference_steps = 50            # Number of denoising steps
    guidance_scale = 7.5                # Scale for classifier-free guidance
    num_samples = 1
    num_rows = 1

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

