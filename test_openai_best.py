import base64
from openai import OpenAI
import pickle

client = OpenAI()

# Function to encode the image
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


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

# Path to your image
def add_player_color(color):
    if color == "Black":
        return text_msg_black
    if color == "White":
        return text_msg_white
    
    assert False, "Invalid color"

folder = "best_moves_128_with_matrix"



with open(f'{folder}/color.pkl', 'rb') as f:
    color_data = pickle.load(f)

for i in range(128):
    image_path = f"{folder}/screenshot_{i}.png"
    # Getting the Base64 string
    base64_image = encode_image(image_path)

    response = client.responses.create(
        model="gpt-4.1-mini",
        input=[
            {
                "role": "user",
                "content": [
                    { "type": "input_text", "text": add_player_color(color_data[i])},
                    {
                        "type": "input_image",
                        "image_url": f"data:image/jpeg;base64,{base64_image}",
                    },
                ],
            }
        ],
    )

    print(response.output_text, flush=True)
    print("---------------", flush=True)
    