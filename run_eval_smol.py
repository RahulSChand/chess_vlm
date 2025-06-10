from transformers import AutoProcessor, Idefics3ForConditionalGeneration
import torch
import numpy as np
from tqdm import tqdm
import ast
import pickle
from PIL import Image

# Add the system message constant from sft_train.py
MAIN_SYSTEM_MESSAGE = """You are a model specialized in reading chess board images and returning a 8x8 matrix of the board position"""

import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluation script')
    parser.add_argument('--model', default="HuggingFaceTB/SmolVLM2-2.2B-Instruct", type=str, required=True,
                        help='Path to the model')
    parser.add_argument('--dataset_path', default = "dataset/",type=str, required=True,
                        help='Path to the dataset')
    parser.add_argument('--limit', default=50,type=int, required=True,
                        help='Limit for evaluation')
    
    parser.add_argument('--task', type=str, default="describe_board", help='Task to evaluate on', choices=["describe_board", "best_move"])
    return parser.parse_args()

def return_dataset_images(dataset_path,i,create_msg,main_prompt):

    img_path = f"{dataset_path}/screenshot_{i}.png"
    board_pos_path = f"{dataset_path}/board_pos_{i}.npy"

    msg, board_pos = create_msg(img_path, board_pos_path, main_prompt)

    return msg, board_pos


def return_dataset_images_move(dataset_path,i,create_msg,main_prompt):

    img_path = f"{dataset_path}/screenshot_{i}.png"
    
    with open(f'{dataset_path}/best_moves.pkl', 'rb') as f:
        best_moves = pickle.load(f)
    with open(f'{dataset_path}/color.pkl', 'rb') as f:
        color_data = pickle.load(f)

    color = color_data[i]

    msg = create_msg(img_path, best_moves[i], main_prompt[color])

    return msg, None  # No board_pos for move prediction



def create_msg(img_path, board_pos, prompt):
    # Load image here and convert to PIL Image like in sft_train.py
    image = Image.open(img_path)
    if image.mode != "RGB":
        image = image.convert("RGB")
    
    # Return complete conversation format like sft_train.py
    msg = [
        {
            "role": "system",
            "content": [{"type": "text", "text": MAIN_SYSTEM_MESSAGE}],
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},  # Use PIL Image object
                {"type": "text", "text": prompt},
            ]
        }
    ]

    return msg, np.load(board_pos)

def get_main_prompt():
    # Use the exact prompt from sft_train.py
    main_prompt = """Your task is to accurately describe the chess board position in the image using a 8x8 grid output. Each type of chess piece, both black and white, is represented by a unique number:

- Empty squares: 0.
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

'''
For example, if the image has pawns only on the first two rows then the first two rows should have 1 and rest should have 0. The matrix should be like a compatible numpy array.
'''

def get_main_prompt_pawns():

    system_message_pawns = """You are a model specialized in interpreting data from chess board images. The image is a 5x5 board with 10 pawns on it.

    Return the (x,y) board positions where the pawns are present.
    """
    return system_message_pawns

def get_main_prompt_move():
    text_msg_black = """You are a model specialized in interpreting data from chess board images. Your task is think carefuly & predict the best move for the current position. You will play as black in this position.

Return the move in SAN notation. Return the move in the following format \\boxed{move}. For example, if the best move is e4, return \\boxed{e4}.

Answer:
"""

    text_msg_white = """You are a model specialized in interpreting data from chess board images. Your task is think carefuly & predict the best move for the current position. You will play as white in this position.

Return the move in SAN notation. Return the move in the following format \\boxed{move}. For example, if the best move is e4, return \\boxed{e4}.

Answer: 
"""

    return {"Black": text_msg_black, "White": text_msg_white}

def generate_text_from_sample(model, processor, sample, max_new_tokens=1024, device="cuda"):
    # Prepare the text input by applying the chat template (similar to smolvlm_example.py)
    text_input = processor.apply_chat_template(
        sample, add_generation_prompt=True
    )

    image_inputs = []
    # Extract image from the user message
    user_message = sample[1]  # The user message (after system message)
    for content_item in user_message["content"]:
        if content_item["type"] == "image":
            image = content_item["image"]  # PIL Image object
            if image.mode != "RGB":
                image = image.convert("RGB")
            image_inputs.append([image])
            break

    # Prepare the inputs for the model
    model_inputs = processor(
        text=text_input,
        images=image_inputs,
        return_tensors="pt",
    ).to(device)

    # Generate text with the model
    generated_ids = model.generate(**model_inputs, max_new_tokens=max_new_tokens)

    # Trim the generated ids to remove the input ids
    trimmed_generated_ids = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(model_inputs.input_ids, generated_ids)]

    # Decode the output text
    output_text = processor.batch_decode(
        trimmed_generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )

    return output_text[0]  # Return the first decoded output text

def call_model(batched_msgs, model):
    return generate_text_from_sample(model, processor, batched_msgs)

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

    # total_non_zero = np.count_nonzero(board_pos)

    #! Now match the non zero entries in out and board pos. 

    

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

def create_msg_move(img_path, pkl, prompt):
    # Load image here and convert to PIL Image like in sft_train.py
    image = Image.open(img_path)
    if image.mode != "RGB":
        image = image.convert("RGB")
    
    # Extract move from pkl (which is best_moves[i])
    move = pkl[0] if isinstance(pkl, (list, tuple)) else pkl
    
    # Return complete conversation format like sft_train.py
    return [
        {
            "role": "system",
            "content": [{"type": "text", "text": MAIN_SYSTEM_MESSAGE}],
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},  # Use PIL Image object
                {"type": "text", "text": prompt},
            ]
        }
    ]

if __name__ == "__main__":
    args = parse_args()

    processor = AutoProcessor.from_pretrained(args.model)
    model = Idefics3ForConditionalGeneration.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        _attn_implementation="flash_attention_2",
        device_map="auto"
    )

    if args.task == "describe_board":
        main_prompt = get_main_prompt()
    elif args.task == "best_move":
        main_prompt = get_main_prompt_move()
    else:
        assert False, "Invalid task"


    mean = 0.0
    running_mean = 0.0
    for i in tqdm(range(args.limit)):

        if args.task == "describe_board":
            msg, board_pos = return_dataset_images(args.dataset_path, i, create_msg, main_prompt)
        elif args.task == "best_move":
            msg, board_pos = return_dataset_images_move(args.dataset_path, i, create_msg, main_prompt)
        else:
            assert False, "Invalid task"

        out = call_model(msg, model)
        
        print(out, flush=True)
        print("--------")

        
        

    





    


