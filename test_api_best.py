from openai import OpenAI
import base64
from tqdm import tqdm
import time
import pickle

client = OpenAI(
  base_url="https://openrouter.ai/api/v1",
)

text_msg_black = """
You are a model specialized in interpreting data from chess board images. Your task is think carefuly & predict the best move for the current position. You will play as black in this position.

Return the move in SAN notation. Return the move in the following format \\boxed{move}. For example, if the best move is e4, return \\boxed{e4}.

Answer:
"""

text_msg_white = """
You are a model specialized in interpreting data from chess board images. Your task is think carefuly & predict the best move for the current position. You will play as white in this position.

Return the move in SAN notation. Return the move in the following format \\boxed{move}. For example, if the best move is e4, return \\boxed{e4}.

Answer: 
"""

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def add_player_color(color):
    if color == "Black":
        return text_msg_black
    if color == "White":
        return text_msg_white
    # return "Descrive the contents of the img"
    
    assert False, "Invalid color"

folder = "best_moves_128_with_matrix"



with open(f'{folder}/color.pkl', 'rb') as f:
    color_data = pickle.load(f)

# folder = "test_random_images"

for i in range(128):
    image_path = f"{folder}/screenshot_{i}.png"
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
                    "text": add_player_color(color_data[i])
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
        

    time.sleep(3)