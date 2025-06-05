from transformers import AutoProcessor, AutoModelForImageTextToText
import torch
import numpy as np
from tqdm import tqdm
import ast
import pickle



import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluation script')
    parser.add_argument('--model', default="HuggingFaceTB/SmolVLM2-2.2B-Instruct", type=str, required=True,
                        help='Path to the model')
    parser.add_argument('--dataset_path', default = "dataset/",type=str, required=True,
                        help='Path to the dataset')
    parser.add_argument('--limit', default=50,type=int, required=True,
                        help='Limit for evaluation')
    return parser.parse_args()

def return_dataset_images(dataset_path,i,create_msg,main_prompt):

    img_path = f"{dataset_path}/screenshot_{i}.png"
    board_pos_path = f"{dataset_path}/board_pos_{i}.npy"

    msg, board_pos = create_msg(img_path, board_pos_path, main_prompt)

    return msg, board_pos


def return_dataset_images_move(dataset_path,i,create_msg,main_prompt):

    img_path = f"{dataset_path}/screenshot_{i}.png"
    board_pos_path = f"{dataset_path}/board_pos_{i}.npy"
    
    with open(f'{dataset_path}/color.pkl', 'rb') as f:
        color_data = pickle.load(f)

    color = color_data[i]

    msg, board_pos = create_msg(img_path, board_pos_path, main_prompt[color])

    return msg, board_pos



def create_msg(img_path, board_pos, prompt):
    msg = [
    {
        "role": "user",
        "content": [
            {"type": "image", "url": img_path},
            {"type": "text", "text": prompt},
        ]
    },
    ]

    return msg, np.load(board_pos)

def get_main_prompt():

    # main_prompt = """
    # You are a model specialized in interpreting chess boards. Your task is to accurately describe the chess board position in the image using a 8x8 grid output. Each type of chess piece, both black and white, is represented by a unique number:
    # - Empty squares: 0
    # - White pieces: Pawn=1, Knight=2, Bishop=3, Rook=4, Queen=5, King=6
    # - Black pieces: Pawn=-1, Knight=-2, Bishop=-3, Rook=-4, Queen=-5, King=-6

    # From the provided chessboard image, convert the visible board into this 8x8 matrix format. For example, the initial chess position would be represented as:

    # [[-4, -2, -3, -5, -6, -3, -2, -4],
    # [-1, -1, -1, -1, -1, -1, -1, -1],
    # [0, 0, 0, 0, 0, 0, 0, 0],
    # [0, 0, 0, 0, 0, 0, 0, 0],
    # [0, 0, 0, 0, 0, 0, 0, 0],
    # [0, 0, 0, 0, 0, 0, 0, 0],
    # [1, 1, 1, 1, 1, 1, 1, 1],
    # [4, 2, 3, 5, 6, 3, 2, 4]]

    # Ensure that your output follows this matrix format based on the pieces shown in the image. Don't output the same matrix as the prompt, look at the image and output the corresponding board position.
    # """

    #Taken from sft
    # main_prompt = """You are a Vision Language Model specialized in interpreting data from chess board images. Your task is to accurately describe the chess board position in the image using a 8x8 grid output. Each type of chess piece, both black and white, is represented by a unique number:\n- Empty squares: 0.'
    # - White pieces: Pawn=1, Knight=2, Bishop=3, Rook=4, Queen=5, King=6
    # - Black pieces: Pawn=-1, Knight=-2, Bishop=-3, Rook=-4, Queen=-5,King=-6

    # From the provided chessboard image, convert the visible board into a 8x8 matrix format. For example, the initial chess position would be represented as:

    # [[-4, -2, -3, -5, -6, -3, -2, -4],
    # [-1, -1, -1, -1, -1, -1, -1, -1],
    # [0, 0, 0, 0, 0, 0, 0, 0],
    # [0, 0, 0, 0, 0, 0, 0, 0],
    # [0, 0, 0, 0, 0, 0, 0, 0],
    # [0, 0, 0, 0, 0, 0, 0, 0],
    # [1, 1, 1, 1, 1, 1, 1, 1],
    # [4, 2, 3, 5, 6, 3, 2, 4]]

    # Ensure that your output strictly follows this matrix format with no deviations, based on the pieces shown in the image.
    # """


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

'''
For example, if the image has pawns only on the first two rows then the first two rows should have 1 and rest should have 0. The matrix should be like a compatible numpy array.
'''

def get_main_prompt_pawns():

    system_message_pawns = """You are a model specialized in interpreting data from chess board images. The image is a 5x5 board with 10 pawns on it.

    Return the (x,y) board positions where the pawns are present.
    """
    return system_message_pawns

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


def call_model(batched_msgs, model):

    inputs = processor.apply_chat_template(
        batched_msgs,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device, dtype=torch.bfloat16) # Ensure inputs are on the same device as the model
    # print(f"\n--- Iteration ---")
    # if 'pixel_values' in inputs:
    #     print(f"Pixel values shape: {inputs['pixel_values'].shape}")
    #     print(f"Pixel values sum: {inputs['pixel_values'].sum().item()}") # Using .item() to get a Python number
    #     # As a quick check, a changing sum usually indicates changing image data.
    # else:
    #     print("Warning: 'pixel_values' not found in inputs. The processor might be using a different key or image handling.")
    # print(f"Input IDs sum (text prompt): {inputs['input_ids'].sum().item()}")
    # ---- End Debugging ----
    prompt_length = inputs['input_ids'].shape[1]

    generated_ids = model.generate(**inputs, do_sample=False, max_new_tokens=256)

    generated_ids = generated_ids[:, prompt_length:]

    # print(generated_ids[-1])
    # assert False
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

if __name__ == "__main__":
    args = parse_args()

    processor = AutoProcessor.from_pretrained(args.model)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        _attn_implementation="flash_attention_2"
    ).to("cuda")

    main_prompt = get_main_prompt_pawns()
    main_prompt = get_main_prompt()
    main_prompt = get_main_prompt_move()

    mean = 0.0
    running_mean = 0.0
    for i in tqdm(range(args.limit)):

        msg, board_pos = return_dataset_images_move(args.dataset_path,i,create_msg,main_prompt)

        out = call_model(msg, model)
        arr = out[0]

        # np_array = np.array(ast.literal_eval(arr))
        # np_array = return_starting_board_pos()
        np_array = arr
        print(np_array, flush=True)
        # print(board_pos, flush=True)
        print("--------")

        # mean+= eval_strict(np_array, board_pos)
        # running_mean = mean/(i+1)
        # print(f"Running mean: {running_mean}")

    
    # print(mean/args.limit)





    


