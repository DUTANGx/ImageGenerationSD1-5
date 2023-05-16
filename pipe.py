import torch
from datetime import datetime
from model import Model
file_name = ""


def pipeline(text, negative_prompt, model, height, width):
    num_inference_steps = 50            # Number of denoising steps
    guidance_scale = 6.5                # Scale for classifier-free guidance
    num_samples = 1
    num_rows = 1

    if model == "GhostMix":
        pipe = Model().ghost_mix_pipe
    else:
        pipe = Model().realistic_pipe

    image = pipe(prompt=text,
                 negative_prompt=negative_prompt,
                 num_images_per_prompt=num_samples,
                 height=height,
                 width=width,
                 num_inference_steps=num_inference_steps,
                 guidance_scale=guidance_scale,
                 ).images[0]

    timestamp = int(datetime.now().timestamp())
    file_name = "/home/ubuntu/stable_diffusion/images/{}.png".format(timestamp)
    image.save(file_name)
    file_link = "http://124.222.40.123:8001/image?{}.png".format(timestamp)
    return file_link
