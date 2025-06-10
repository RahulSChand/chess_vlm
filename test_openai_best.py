import base64
from openai import OpenAI
import pickle
import argparse
import os

# Set up argument parser
parser = argparse.ArgumentParser(description='Process chess board images and predict best moves using OpenAI API')
parser.add_argument('--folder', type=str, default='best_moves_128_with_matrix',
                    help='Path to the folder containing screenshots and color data')
args = parser.parse_args()

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

# Load color data
with open(os.path.join(args.folder, 'color.pkl'), 'rb') as f:
    color_data = pickle.load(f)

# Get list of screenshot files in the folder
screenshot_files = [f for f in os.listdir(args.folder) if f.startswith('screenshot_') and f.endswith('.png')]
screenshot_files.sort()  # Ensure files are processed in order

for screenshot_file in screenshot_files:
    image_path = os.path.join(args.folder, screenshot_file)
    # Extract index from filename for color data lookup
    i = int(screenshot_file.split('_')[1].split('.')[0])
    
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
    