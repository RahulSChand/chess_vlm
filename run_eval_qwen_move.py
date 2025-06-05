from transformers import AutoProcessor, AutoModelForImageTextToText
import torch
import numpy as np
from tqdm import tqdm
import ast
from qwen_vl_utils import process_vision_info
import pickle
from transformers import Qwen2VLForConditionalGeneration, Qwen2VLProcessor


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

SYSTEM_MESSAGE = """You are a model specialized in interpreting chess board images. Your task is think carefuly & predict the best move for the current position. """

def format_data(img_path, prompt):
    return [
        {
            "role": "system",
            "content": [{"type": "text", "text": SYSTEM_MESSAGE}],  # SYSTEM MESSAGE!
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": img_path,
                },
                {
                    "type": "text",
                    "text": prompt,
                },
            ],
        },
    ]




def get_main_prompt():

    text_msg_black = """
You will play as black in this position.

Return the move in SAN notation. Return the move in the following format \\boxed{move}. For example, if the best move is e4, return \\boxed{e4}.

Answer:
"""

    text_msg_white = """
You will play as white in this position.

Return the move in SAN notation. Return the move in the following format \\boxed{move}. For example, if the best move is e4, return \\boxed{e4}.

# Answer:
# """

    return {"White": text_msg_white, "Black": text_msg_black}
    

def get_main_prompt_3():

    text_msg_black = """
You will play as black in this position.

Return the top 3 moves in SAN notation. Return the moves in the following format \\boxed{move}. For example, if the best moves are e4, fxg5, g4, return \\boxed{e4} \\boxed{fxg5} \\boxed{g4}.

Answer:
"""

    text_msg_white = """
You will play as white in this position.

Return the top 3 moves in SAN notation. Return the moves in the following format \\boxed{move}. For example, if the best moves are e4, fxg5, g4, return \\boxed{e4} \\boxed{fxg5} \\boxed{g4}.

# Answer:
# """

    return {"White": text_msg_white, "Black": text_msg_black}




# def get_main_prompt():

#     main_prompt = """You are a Vision Language Model specialized in interpreting data from chess board images. Your task is to accurately describe the chess board position in the image using a 8x8 grid output. Each type of chess piece, both black and white, is represented by a unique number:
# - Empty squares: 0.
# - White pieces: Pawn=1, Knight=2, Bishop=3, Rook=4, Queen=5, King=6
# - Black pieces: Pawn=-1, Knight=-2, Bishop=-3, Rook=-4, Queen=-5,King=-6

# From the provided chessboard image, convert the visible board into this 8x8 matrix format. For example, the initial chess position would be represented as:
# Game State: 
# [[-4, -2, -3, -5, -6, -3, -2, -4],
# [-1, -1, -1, -1, -1, -1, -1, -1],
# [0, 0, 0, 0, 0, 0, 0, 0],
# [0, 0, 0, 0, 0, 0, 0, 0],
# [0, 0, 0, 0, 0, 0, 0, 0],
# [0, 0, 0, 0, 0, 0, 0, 0],
# [1, 1, 1, 1, 1, 1, 1, 1],
# [4, 2, 3, 5, 6, 3, 2, 4]]
# Ensure that your output strictly follows this format

# Solution:
# """

#     # main_prompt = """
#     # Describe the image in 2 sentences.
#     # """

#     return main_prompt

def call_model(batched_msgs, model):

    inputs = processor.apply_chat_template(
        batched_msgs,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device, dtype=torch.bfloat16) # Ensure inputs are on the same device as the model
    
    print(f"\n--- Iteration ---")
    if 'pixel_values' in inputs:
        print(f"Pixel values shape: {inputs['pixel_values'].shape}")
        print(f"Pixel values sum: {inputs['pixel_values'].sum().item()}") # Using .item() to get a Python number
        # As a quick check, a changing sum usually indicates changing image data.
    else:
        print("Warning: 'pixel_values' not found in inputs. The processor might be using a different key or image handling.")
    print(f"Input IDs sum (text prompt): {inputs['input_ids'].sum().item()}")
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
    


def get_the_boxed_move(move):
    return f"\\boxed{{{move}}}"

def get_3(arr):
    arr_3 = arr[:3]
    arr_3 = [get_the_boxed_move(move) for move in arr_3]
    return ' '.join(arr_3)


def eval(out, board_pos):

    print(out==board_pos)


    mean = np.mean(out==board_pos)
    return mean



def generate_text_from_sample(model, processor, sample, max_new_tokens=1024, device="cuda"):
    # Prepare the text input by applying the chat template
    text_input = processor.apply_chat_template(
        sample, tokenize=False, add_generation_prompt=True  # Use the sample without the system message
    )

    # Process the visual input from the sample
    image_inputs, _ = process_vision_info(sample)

    # Prepare the inputs for the model
    model_inputs = processor(
        text=[text_input],
        images=image_inputs,
        return_tensors="pt",
    ).to(
        device
    )  # Move inputs to the specified device

    # Generate text with the model
    generated_ids = model.generate(**model_inputs, max_new_tokens=max_new_tokens)

    # Trim the generated ids to remove the input ids
    trimmed_generated_ids = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(model_inputs.input_ids, generated_ids)]

    # Decode the output text
    output_text = processor.batch_decode(
        trimmed_generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )

    return output_text[0]  # Return the first decoded output text


if __name__ == "__main__":
    args = parse_args()

    processor = Qwen2VLProcessor.from_pretrained(args.model)
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )

    # main_prompt = get_main_prompt_3()
    main_prompt = get_main_prompt()

    import pickle
    with open(f"{args.dataset_path}/color.pkl", "rb") as f:
        color_data = pickle.load(f)

    with open(f"{args.dataset_path}/best_moves.pkl", "rb") as f:
        best_moves = pickle.load(f)
    # main_prompt = get_main_prompt_move()

    train_dataset = [format_data(f"{args.dataset_path}/screenshot_{i}.png", main_prompt[color_data[i]]) for i in range(args.limit)]


    #! For best move comment out if not
    # with open(f'{args.dataset_path}/color.pkl', 'rb') as f:
    #     color_data = pickle.load(f)
    
    # train_dataset = []
    # for i in range(args.limit):
    #     train_dataset.append(format_data(f"{args.dataset_path}/screenshot_{i}.png", main_prompt[color_data[i]]))

    
    for i in range(args.limit):
        output = generate_text_from_sample(model, processor, train_dataset[i])

        print(output)
        print("---------------")

    # mean = 0.0
    # running_mean = 0.0
    # for i in tqdm(range(args.limit)):
    #     msg, board_pos = return_dataset_images(args.dataset_path,i,create_msg,main_prompt)

    #     out = call_model(msg, model)
    #     arr = out[0]
    #     np_array = np.array(ast.literal_eval(arr))


    #     # np_array = return_starting_board_pos()
    #     print(np_array)
    #     print(board_pos)
       
    #     print("--------")

    #     mean+= eval_strict(np_array, board_pos)

    #     running_mean = mean/(i+1)
    #     print(f"Running mean: {running_mean}")

    
    # print(mean/args.limit)





    


