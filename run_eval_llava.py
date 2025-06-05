from transformers import AutoProcessor, AutoModelForVision2Seq, LlavaForConditionalGeneration
import torch
import numpy as np
from tqdm import tqdm
import ast
from PIL import Image
import pickle

import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluation script')
    parser.add_argument('--model', default="llava-hf/llava-1.5-7b-hf", type=str, required=True,
                        help='Path to the model')
    parser.add_argument('--dataset_path', default = "dataset/",type=str, required=True,
                        help='Path to the dataset')
    parser.add_argument('--limit', default=50,type=int, required=True,
                        help='Limit for evaluation')
    return parser.parse_args()

def return_dataset_images_test(dataset_path,i,create_msg,main_prompt):

    img_path = f"{dataset_path}/screenshot_{i}.png"
    board_pos_path = "best_moves_128_with_matrix/board_pos_0.npy"

    msg, board_pos, image = create_msg(img_path, board_pos_path, main_prompt)

    return msg, board_pos, image

def return_dataset_images(dataset_path,i,create_msg,main_prompt):

    img_path = f"{dataset_path}/screenshot_{i}.png"
    board_pos_path = f"{dataset_path}/board_pos_{i}.npy"

    msg, board_pos, image = create_msg(img_path, board_pos_path, main_prompt)

    return msg, board_pos, image

def return_dataset_images_move(dataset_path,i,create_msg,main_prompt):

    img_path = f"{dataset_path}/screenshot_{i}.png"
    board_pos_path = f"{dataset_path}/board_pos_{i}.npy"
    
    with open(f'{dataset_path}/color.pkl', 'rb') as f:
        color_data = pickle.load(f)

    color = color_data[i]

    msg, board_pos, image = create_msg(img_path, board_pos_path, main_prompt[color])

    return msg, board_pos, image

def create_msg(img_path, board_pos, prompt):
    # Load image
    image = Image.open(img_path).convert('RGB')
    
    msg = [
    {
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": prompt},
        ]
    },
    ]

    return msg, np.load(board_pos), image

def get_main_prompt():

    main_prompt = """You are a Vision Language Model specialized in interpreting data from chess board images. Your task is to accurately describe the chess board position in the image using a 8x8 grid output. Each type of chess piece, both black and white, is represented by a unique number:\n- Empty squares: 0.'
    - White pieces: Pawn=1, Knight=2, Bishop=3, Rook=4, Queen=5, King=6
    - Black pieces: Pawn=-1, Knight=-2, Bishop=-3, Rook=-4, Queen=-5,King=-6

    From the provided chessboard image, convert the visible board into this 8x8 matrix format. For example, the initial chess position would be represented as:
    Game State: 
    [[-4, -2, -3, -5, -6, -3, -2, -4],
    [-1, -1, -1, -1, -1, -1, -1, -1],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [1, 1, 1, 1, 1, 1, 1, 1],
    [4, 2, 3, 5, 6, 3, 2, 4]]
    Ensure that your output strictly follows this format

    Solution:
    """

    return main_prompt


def get_main_prompt_move():

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

    return {"Black": text_msg_black, "White": text_msg_white}


def get_test_prompt():

    prompt = """
    Describe the contents of the image.
    """

    return prompt


def call_model(batched_msgs, image, model):
    # Get the texts and images, and apply the chat template
    text = processor.apply_chat_template(batched_msgs, tokenize=False)
    
    # For LLaVA1.5, we handle single image
    if isinstance(model, LlavaForConditionalGeneration):
        images = image
    else:
        images = [image]

    # Tokenize the texts and process the images
    inputs = processor(text=text, images=images, return_tensors="pt", padding=True).to(model.device, dtype=torch.bfloat16)
    
    # print(f"\n--- Iteration ---")
    # if 'pixel_values' in inputs:
    #     print(f"Pixel values shape: {inputs['pixel_values'].shape}")
    #     print(f"Pixel values sum: {inputs['pixel_values'].sum().item()}")
    # else:
    #     print("Warning: 'pixel_values' not found in inputs. The processor might be using a different key or image handling.")
    # print(f"Input IDs sum (text prompt): {inputs['input_ids'].sum().item()}")
    
    prompt_length = inputs['input_ids'].shape[1]

    generated_ids = model.generate(**inputs, do_sample=False, max_new_tokens=256)

    generated_ids = generated_ids[:, prompt_length:]

    generated_texts = processor.batch_decode(
        generated_ids,
        skip_special_tokens=True,
    )

    return generated_texts

def eval_strict(out, board_pos):

    #! Only reward if non zero entries match. First find the total non zero entries in board_pos (which is ground truth).
    # Create a boolean mask for non-zero elements in board_pos
    non_zero_mask = board_pos != 0

    # If there are no non-zero elements in board_pos, return 1.0 (or handle as appropriate)
    if not np.any(non_zero_mask):
        return 1.0  # Or 0.0, or raise an error, depending on desired behavior

    # Compare elements of out and board_pos where board_pos is non-zero
    correct_matches = out[non_zero_mask] == board_pos[non_zero_mask]

    # Calculate accuracy
    accuracy = np.mean(correct_matches)
    
    return accuracy

def return_starting_board_pos():

    return np.array([[-4, -2, -3, -5, -6, -3, -2, -4],
[-1, -1, -1, -1, -1, -1, -1, -1],
[0, 0, 0, 0, 0, 0, 0, 0],
[0, 0, 0, 0, 0, 0, 0, 0],
[0, 0, 0, 0, 0, 0, 0, 0],
[0, 0, 0, 0, 0, 0, 0, 0],
[1, 1, 1, 1, 1, 1, 1, 1],
[4, 2, 3, 5, 6, 3, 2, 4]]
    )

def eval(out, board_pos):

    print(out==board_pos)

    mean = np.mean(out==board_pos)
    return mean

if __name__ == "__main__":
    args = parse_args()

    processor = AutoProcessor.from_pretrained(args.model)
    model = AutoModelForVision2Seq.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    ).to("cuda")

    main_prompt = get_main_prompt()
    main_prompt = get_main_prompt_move()
    # main_prompt = get_test_prompt()


    mean = 0.0
    running_mean = 0.0
    for i in tqdm(range(args.limit)):

        msg, board_pos, image = return_dataset_images_move(args.dataset_path,i,create_msg,main_prompt)

        out = call_model(msg, image, model)
        arr = out[0]
        # np_array = np.array(ast.literal_eval(arr))
        np_array = arr

        print(np_array, flush=True)
        # print(board_pos)
        print("--------", flush=True)

        # mean+= eval_strict(np_array, board_pos)

        # running_mean = mean/(i+1)
        # print(f"Running mean: {running_mean}")

    # print(mean/args.limit) 