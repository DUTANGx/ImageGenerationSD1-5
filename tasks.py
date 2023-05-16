from celery import Celery
import time
import config
from translate import translate
from pipe import pipeline

celery_app = Celery(config.celery_name, broker=config.BROKER_URL, backend=config.BACKEND_URL)


@celery_app.task
def generate_image(text, model="Realistic"):
    # model_id = "prompthero/openjourney"
    # model_id = "SG161222/Realistic_Vision_V2.0"
    text = translate(text)
    if text == None:
        return "ERR1"
    if model == "Realistic":
        height = 768  # default height of Stable Diffusion
        width = 768  # default width of Stable Diffusion
        prompt = "RAW photo, {}, 8k uhd, dslr, soft lighting, high quality, film grain, Fujifilm XT3".format(
            text)
        negative_prompt = "nude, NSFW, (deformed iris, deformed pupils, semi-realistic, cgi, 3d, render, sketch, cartoon, drawing, anime:1.4), text, close up, cropped, out of frame, worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers, long neck"

    elif model == "GhostMix":
        height = 768  # default height of Stable Diffusion
        width = 512  # default width of Stable Diffusion
        prompt = "(masterpiece, top quality, best quality, official art, beautiful and aesthetic:1.2), (fractal art:1.3), highres, ultra detailed, {}".format(
            text)
        negative_prompt = "EasyNegative, (worst quality, low quality:1.4), (blush:1.3), nude, text, username, watermark,nsfw, aidv1-neg, animestylenegativeembedding_dreamshaper, bad-artist, bad-artist-anime, sketch by bad-artist, painting by bad-artist, photograph by bad-artist, bad-picture-chill-75v, badhandv4, badhandv5, badv3, badv4, badv5, bad_prompt, bad_prompt_version2, EasyNegative, ng_deepnegative_v1_75t, verybadimagenegative_v1.2-6400, verybadimagenegative_v1.3, Unspeakable-Horrors-Composition-4v,(worst quality, low quality:1.4), (raw photo)"

    else:
        return "ERR2"

    image = pipeline(prompt, negative_prompt, model, height, width)
    return image
