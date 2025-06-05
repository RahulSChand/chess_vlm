from openai import OpenAI
import base64
from tqdm import tqdm
import time

client = OpenAI(
  base_url="https://openrouter.ai/api/v1",
)

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")



text_msg = """
You are a Vision Language Model specialized in interpreting data from chess board images. Your task is to accurately describe the chess board position in the image using a 8x8 grid output. Each type of chess piece, both black and white, is represented by a unique number:\n- Empty squares: 0.'
- White pieces: Pawn=1, Knight=2, Bishop=3, Rook=4, Queen=5, King=6
- Black pieces: Pawn=-1, Knight=-2, Bishop=-3, Rook=-4, Queen=-5,King=-6

From the provided chessboard image, convert the visible board into this 8x8 matrix format. For example, the initial chess position would be represented as:

Game State: [[-4, -2, -3, -5, -6, -3, -2, -4],
[-1, -1, -1, -1, -1, -1, -1, -1],
[0, 0, 0, 0, 0, 0, 0, 0],
[0, 0, 0, 0, 0, 0, 0, 0],
[0, 0, 0, 0, 0, 0, 0, 0],
[0, 0, 0, 0, 0, 0, 0, 0],
[1, 1, 1, 1, 1, 1, 1, 1],
[4, 2, 3, 5, 6, 3, 2, 4]]

Ensure that your output strictly follows this format. Just return the matrix and no other text.

Solution:
"""



for i in range(128):
    image_path = f"dataset_random_384/screenshot_{i}.png"
    base64_image = encode_image(image_path)
    
    try:
        completion = client.chat.completions.create(
            extra_body={},
            model="mistralai/pixtral-12b",
            # model="meta-llama/llama-3.2-11b-vision-instruct",
            messages=[
                {
                "role": "user", 
                "content": [
                    {
                    "type": "text",
                    "text": text_msg
                    },
                    {
                    "type": "image_url",
                    "image_url": f"data:image/jpeg;base64,{base64_image}",
                    }
                ]
                }
            ]
        )
        print(completion.choices[0].message.content, flush=True)
        print("---------------", flush=True)
        
    except Exception as e:
        print(f"Error on image {i}: {e}")
        # Add a small delay to avoid hitting rate limits
        time.sleep(1)
        continue