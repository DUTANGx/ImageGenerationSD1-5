
import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler


def singleton(cls):
    _instance = {}

    def inner():
        if cls not in _instance:
            _instance[cls] = cls()
        return _instance[cls]
    return inner
    
@singleton
class Model(object):
    def __init__(cls):
        # initialize T2I model
        realistic_pipe = StableDiffusionPipeline.from_pretrained("SG161222/Realistic_Vision_V2.0", torch_dtype=torch.float16, resume_download=True, use_safetensors=False)
        realistic_pipe.scheduler = DPMSolverMultistepScheduler.from_config(realistic_pipe.scheduler.config)
        realistic_pipe = realistic_pipe.to("cuda")
        realistic_pipe.enable_xformers_memory_efficient_attention()
        ghost_mix_pipe = StableDiffusionPipeline.from_pretrained("/home/ubuntu/tangdu/GhostMix", torch_dtype=torch.float16, resume_download=True, use_safetensors=False)
        ghost_mix_pipe.scheduler = DPMSolverMultistepScheduler.from_config(ghost_mix_pipe.scheduler.config)
        ghost_mix_pipe = ghost_mix_pipe.to("cuda")
        ghost_mix_pipe.enable_xformers_memory_efficient_attention()
        cls.realistic_pipe = realistic_pipe
        cls.ghost_mix_pipe = ghost_mix_pipe
        