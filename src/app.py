from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
import asyncio
from cuda import cudart
from fastapi.middleware.cors import CORSMiddleware


# Import the necessary modules from the provided code
from src.txt2img_xl import StableDiffusionXLPipeline

app = FastAPI()

origins = [
    "http://localhost:5500",  # Adjust this to the address of your frontend
    # "http://127.0.0.1:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Initialize your model (adapt based on the provided code)
init_pipeline = {'version': 'xl-1.0', 'max_batch_size': 4, 'denoising_steps': 30, 
                 'scheduler': None, 'guidance_scale': 5.0, 'output_dir': 'output', 
                 'hf_token': None, 'verbose': False, 'nvtx_profile': False, 
                 'use_cuda_graph': True, 'lora_scale': None, 'lora_path': None, 
                 'framework_model_dir': 'pytorch_model', 'torch_inference': ''}

enable_refiner_user = False
seed_value = None
image_strength=0.3

onnx_dir = '/workspace/sdxl-1.0-refiner'
engine_dir = '/workspace/engine_xl_refiner'

model = StableDiffusionXLPipeline(vae_scaling_factor=0.13025, enable_refiner=enable_refiner_user, **init_pipeline)

@app.on_event("startup")
async def startup_event():
    # Load TensorRT engines and pytorch modules
    kwargs_load_refiner = {'onnx_refiner_dir': onnx_dir, 'engine_refiner_dir': engine_dir} if enable_refiner_user else {}
    load_engine = {'onnx_opset': 18, 'opt_batch_size': 1, 'opt_image_height': 1024, 'opt_image_width': 1024, 'static_batch': True, 
                   'static_shape': True, 'enable_all_tactics': False, 'enable_refit': False, 'timing_cache': None}
    model.loadEngines(
        'pytorch_model',
        '/workspace/sdxl-1.0-base',
        'engine',
        **kwargs_load_refiner,
        **load_engine)
    
    # Load resources
    _, shared_device_memory = cudart.cudaMalloc(model.get_max_device_memory())
    model.activateEngines(shared_device_memory)
    model.loadResources(load_engine['opt_image_height'], load_engine['opt_image_width'], load_engine['opt_batch_size'], seed_value)

class ImageRequest(BaseModel):
    prompt: str

@app.post("/generate/")
async def generate_image(request: ImageRequest):
    # Async handling of the model inference
    loop = asyncio.get_event_loop()
    # Run inference
    kwargs_infer_refiner = {'image_strength': image_strength} if enable_refiner_user else {}
    args_run_demo = ([request.prompt], [''], 1024, 1024, 1, 1, 1, True)
    # model.run(*args_run_demo, **kwargs_infer_refiner)
    
    result, base64_images = await loop.run_in_executor(None, model.run, *args_run_demo,  **kwargs_infer_refiner)
    
    # return: prompt, image path,
    # not sure what to do about user, uuid, or timestamp
    
    # there will always be one image so use first item and list and return the prompt as well
    return {"image": base64_images[0], "prompt": request.prompt}  # Adapt based on how the image is returned

