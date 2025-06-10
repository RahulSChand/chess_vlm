import base64
from openai import OpenAI
import argparse
import os

# Set up argument parser
parser = argparse.ArgumentParser(description='Process chess board images using OpenAI API')
parser.add_argument('--dataset', type=str, default='dataset_random_384',
                    help='Path to the dataset directory containing screenshots')
args = parser.parse_args()

client = OpenAI()

# Function to encode the image
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


# Path to your image

text_msg = """
You are a model specialized in interpreting data from chess board images. Your task is to accurately describe the chess board position in the image using a 8x8 grid output. Each type of chess piece, both black and white, is represented by a unique number:

- Empty squares: 0
- White pieces: Pawn=1, Knight=2, Bishop=3, Rook=4, Queen=5, King=6
- Black pieces: Pawn=-1, Knight=-2, Bishop=-3, Rook=-4, Queen=-5, King=-6

From the provided image, convert the board into this 8x8 matrix format. For example, the initial chess position would be represented as:

[[-4, -2, -3, -5, -6, -3, -2, -4],
[-1, -1, -1, -1, -1, -1, -1, -1],
[0, 0, 0, 0, 0, 0, 0, 0],
[0, 0, 0, 0, 0, 0, 0, 0],
[0, 0, 0, 0, 0, 0, 0, 0],
[0, 0, 0, 0, 0, 0, 0, 0],
[1, 1, 1, 1, 1, 1, 1, 1],
[4, 2, 3, 5, 6, 3, 2, 4]]

Ensure that your output strictly follows this format. Just return the matrix, no other text.

Solution:
"""

# Get list of screenshot files in the dataset directory
screenshot_files = [f for f in os.listdir(args.dataset) if f.startswith('screenshot_') and f.endswith('.png')]
screenshot_files.sort()  # Ensure files are processed in order

for screenshot_file in screenshot_files:
    image_path = os.path.join(args.dataset, screenshot_file)
    # Getting the Base64 string
    base64_image = encode_image(image_path)

    response = client.responses.create(
        model="gpt-4.1-mini",
        input=[
            {
                "role": "user",
                "content": [
                    { "type": "input_text", "text": text_msg},
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