import torch
import argparse
from diffusers import StableDiffusionXLImg2ImgPipeline
from diffusers.utils import load_image

# Set up argument parser
parser = argparse.ArgumentParser(description='Generate chess move images using Stable Diffusion XL')
parser.add_argument('--input_img', type=str, required=True, help='Path to the input chess board image')
parser.add_argument('--save_img', type=str, required=True, help='Path where the generated image will be saved')
args = parser.parse_args()

pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-refiner-1.0", torch_dtype=torch.float16, variant="fp16", use_safetensors=True
)
pipe = pipe.to("cuda")

init_image = load_image(args.input_img).convert("RGB")
prompt = "This is a chess board position. You need to generate an image which is how the board position will look like AFTER the best move in this position is played by black. Don't output th same board, but board AFTER THE BEST MOVE IS PLAYED" 
image = pipe(prompt, image=init_image).images

#Save img
image[0].save(args.save_img)

