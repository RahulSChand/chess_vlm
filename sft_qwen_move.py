from transformers import AutoProcessor, AutoModelForImageTextToText
import torch
import numpy as np
from tqdm import tqdm
import ast
from qwen_vl_utils import process_vision_info
from trl import SFTConfig
import wandb
from transformers import Qwen2VLForConditionalGeneration, Qwen2VLProcessor
from trl import SFTTrainer
from transformers import TrainerCallback
from accelerate import Accelerator
import argparse
from transformers import AutoTokenizer


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluation script')
    parser.add_argument('--model', default="HuggingFaceTB/SmolVLM2-2.2B-Instruct", type=str, required=True,
                        help='Path to the model')
    parser.add_argument('--dataset_path', default = "dataset/",type=str, required=True,
                        help='Path to the dataset')
    parser.add_argument('--limit', default=50,type=int, required=True,
                        help='Limit for evaluation')
    
    parser.add_argument('--name', default="qwen2-7b",type=str, required=True,help='Name for wandb')

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


def format_data(img_path, prompt, npy_arr):
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
        {
            "role": "assistant",
            "content": [{"type": "text", "text": npy_arr}],
        }
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

def convert_npy_to_str(npy_path):
    
    arr = np.load(npy_path)
    python_list_of_lists = arr.tolist()
    string_rows = [str(row) for row in python_list_of_lists]
    readable_string_representation = "[" + ",\n".join(string_rows) + "]"
    return readable_string_representation

def get_the_boxed_move(move):
    return f"\\boxed{{{move}}}"

def get_3(arr):
    arr_3 = arr[:3]
    arr_3 = [get_the_boxed_move(move) for move in arr_3]
    return ' '.join(arr_3)

if __name__ == "__main__":
    args = parse_args()

    # accelerator = Accelerator()

    processor = Qwen2VLProcessor.from_pretrained(args.model)
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )

    main_prompt = get_main_prompt()
    # main_prompt = get_main_prompt_3()
    import pickle
    with open(f"{args.dataset_path}/color.pkl", "rb") as f:
        color_data = pickle.load(f)

    with open(f"{args.dataset_path}/best_moves.pkl", "rb") as f:
        best_moves = pickle.load(f)

    # print(len(best_moves))
    # print(best_moves[0])
    # assert False

    train_dataset = [format_data(f"{args.dataset_path}/screenshot_{i}.png", main_prompt[color_data[i]], get_the_boxed_move(best_moves[i][0])) for i in range(args.limit)]

    

    # train_dataset = [format_data(f"{args.dataset_path}/screenshot_{i}.png", main_prompt[color_data[i]], get_3(best_moves[i])) for i in range(args.limit)]

    eval_dataset = [train_dataset[0]]

    # Configure training arguments
    training_args = SFTConfig(
        output_dir=f"{args.name}",  # Directory to save the model
        num_train_epochs=500,  # Number of training epochs
        per_device_train_batch_size=2,  # Batch size for training
        per_device_eval_batch_size=1,  # Batch size for evaluation
        gradient_accumulation_steps=4,  # Steps to accumulate gradients
        
        # gradient_checkpointing=False, 
        # Optimizer and scheduler settings
        optim="adamw_torch_fused",  # Optimizer type
        learning_rate=5e-5,  # Learning rate for training
        lr_scheduler_type="constant",
        logging_steps=2,  # Steps interval for logging
        
        # eval_steps=4,  # Steps interval for evaluation
        # eval_strategy="steps",  # Strategy for evaluation
        
        save_strategy="steps",  # Strategy for saving the model
        save_steps=1000000,
        bf16=True,  # Use bfloat16 precision
        warmup_ratio=0.03,  # Ratio of total steps for warmup
        report_to="wandb",  # Reporting tool for tracking metrics
        dataset_text_field="",  # Text field in dataset
        dataset_kwargs={"skip_prepare_dataset": True},  # Additional dataset options   
    )

    training_args.remove_unused_columns = False  # Keep unused columns in dataset

    wandb.init(
        project="chess_vllm",  # change this
        name=args.name,  # change this
        config=training_args,
    )

    # Create a data collator to encode text and image pairs
    def collate_fn(examples):
        # Get the texts and images, and apply the chat template
        texts = [
            processor.apply_chat_template(example, tokenize=False, add_generation_prompt=True) for example in examples
        ]  # Prepare texts for processing

        image_inputs = [process_vision_info(example)[0] for example in examples]  # Process the images to extract inputs

        # Tokenize the texts and process the images
        batch = processor(
            text=texts, images=image_inputs, return_tensors="pt", padding=True
        )  # Encode texts and images into tensors

        # The labels are the input_ids, and we mask the padding tokens in the loss computation
        labels = batch["input_ids"].clone()  # Clone input IDs for labels
        
        for i, example in enumerate(examples):
        # Get system + user part (everything except assistant response)
            sysuser_conv = example[:-1]
            sysuser_text = processor.apply_chat_template(sysuser_conv, tokenize=False)
            sysuser_img, _ = process_vision_info(sysuser_conv)
            
            sysuser_inputs = processor(
                text=[sysuser_text],
                images=[sysuser_img], 
                return_tensors="pt",
                padding=True,
            )
            
            sysuser_len = sysuser_inputs["input_ids"].shape[1]
            labels[i, :sysuser_len] = -100  # Mask everything before assistant
        
        batch["labels"] = labels
        return batch  # Return the prepared batch




    #! This is for debugging the collate_fn function
    # num_samples_to_inspect = 4
    # sample_examples_for_collate = [train_dataset[i] for i in range(num_samples_to_inspect)]

    # batch_output = collate_fn(sample_examples_for_collate)

    # print(batch_output['input_ids'].shape)
    # token_ids = batch_output['input_ids'][0]

    # token_ids_labels = batch_output['labels'][0]

    # model_path = "Qwen/Qwen2-VL-7B-Instruct"

    # tokenizer = AutoTokenizer.from_pretrained(model_path)

    # print(tokenizer.decode(token_ids))

    # for i, token_id in enumerate(token_ids):
        
    #     print(i, token_id, tokenizer.decode(token_id))
    #     if token_ids_labels[i] != -100:
    #         print(i, token_ids_labels[i], tokenizer.decode(token_ids_labels[i]))
    #     else:
    #         print(i, token_ids_labels[i])
    #     print("----")


    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        # eval_dataset=eval_dataset,
        data_collator=collate_fn,
    )

    
    trainer.train()

    processor.save_pretrained(training_args.output_dir)



        


