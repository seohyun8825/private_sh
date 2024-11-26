
from RegionalDiffusion_xl import RegionalDiffusionXLPipeline
from diffusers.schedulers import KarrasDiffusionSchedulers,DPMSolverMultistepScheduler
from mllm import local_llm,GPT4,Hard_Code
import torch
# If you want to use load ckpt, initialize with ".from_single_file". 
#pipe = RegionalDiffusionXLPipeline.from_single_file("path to your ckpt", torch_dtype=torch.float16, use_safetensors=True, variant="fp16")
# If you want to use diffusers, initialize with ".from_pretrained".
# pipe = RegionalDiffusionXLPipeline.from_pretrained("path to your diffusers",torch_dtype=torch.float16, use_safetensors=True, variant="fp16")
pipe = RegionalDiffusionXLPipeline.from_pretrained("comin/IterComp",torch_dtype=torch.float16, use_safetensors=True)
pipe.to("cuda")
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config,use_karras_sigmas=True)
pipe.enable_xformers_memory_efficient_attention()
## User input
prompt= 'Four cut picture of cute dog'
para_dict = Hard_Code(prompt,key='')

split_ratio = para_dict['Final split ratio']
regional_prompt = para_dict['Regional Prompt']
negative_prompt = ""

image_path= [
    "/home/kubig/RPG-DiffusionMaster/test_1.jpg",
    "/home/kubig/RPG-DiffusionMaster/test2.jpg",
    "/home/kubig/RPG-DiffusionMaster/test3.jpg",
    "/home/kubig/RPG-DiffusionMaster/test4.jpg",
]
print("split",split_ratio)
images = pipe(
    prompt = regional_prompt,
    split_ratio = split_ratio, # The ratio of the regional prompt, the number of prompts is the same as the number of regions, and the number of prompts is the same as the number of regions
    batch_size = 1, #batch size
    base_ratio = 0.5, # The ratio of the base prompt    
    base_prompt= prompt,       
    num_inference_steps=20, # sampling step
    height = 1024, 
    negative_prompt=negative_prompt, # negative prompt
    width = 1024, 
    seed = 2468,# random seed
    guidance_scale = 7.0,
    image_path = image_path,
).images[0]
images.save("test.png")