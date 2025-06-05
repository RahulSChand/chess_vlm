import torch
from diffusers import StableDiffusionXLImg2ImgPipeline
from diffusers.utils import load_image

pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-refiner-1.0", torch_dtype=torch.float16, variant="fp16", use_safetensors=True
)
pipe = pipe.to("cuda")
# url = "https://huggingface.co/datasets/patrickvonplaten/images/resolve/main/aa_xl/000000009.png"
path = "best_moves_128_with_matrix_train_1024_filter/screenshot_10.png"

init_image = load_image(path).convert("RGB")
prompt = "This is a chess board position. You need to generate an image which is how the board position will look like AFTER the best move in this position is played by black. Don't output th same board, but board AFTER THE BEST MOVE IS PLAYED" 
image = pipe(prompt, image=init_image).images

#Save img
image[0].save("diff_test_2.png")

