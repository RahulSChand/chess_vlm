import wandb
import trl
import datasets
from trl import SFTConfig
import torch
import numpy as np
from trl import SFTTrainer
from transformers import AutoProcessor, AutoModelForImageTextToText, Idefics3ForConditionalGeneration
from transformers import AutoTokenizer
import wandb
from accelerate import Accelerator
from transformers import TrainerCallback
import argparse
import pickle
from PIL import Image

# Parse command line arguments
parser = argparse.ArgumentParser(description='SFT Training')
parser.add_argument('--per_device_batch_size', type=int, default=4, help='Batch size per device')
parser.add_argument('--grad_accum_steps', type=int, default=2, help='Gradient accumulation steps')
parser.add_argument('--model_name', type=str, default="llava-hf/llava-1.5-7b-hf", help='Model name/path')
parser.add_argument('--save_name', type=str, default="llava_finetuned_384_only_img_masked_2", help='Save name for output dir and wandb')
parser.add_argument('--dataset_name', type=str, default="dataset_random_384/", help='Dataset path')
args = parser.parse_args()

accelerator = Accelerator()

# MAIN_SYSTEM_MESSAGE = """You are a model specialized in reading chess board images and returning a 8x8 matrix of the board position"""

MAIN_SYSTEM_MESSAGE = """You are a model specialized in interpreting chess board images. Your task is think carefuly & predict the best move for the current position. """


system_message = """Your task is to accurately describe the chess board position in the image using a 8x8 grid output. Each type of chess piece, both black and white, is represented by a unique number:

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

system_message_pawns = """You are a model specialized in interpreting data from chess board images. The image is of a 5x5 board with 10 pawns on it. You need to output a 5x5 matrix where entry = 1 if the square is occupied by a pawn and 0 otherwise.

For example, if the image has pawns only on the first two rows the output is: 
[[1, 1, 1, 1, 1],
[1, 1, 1, 1, 1],
[0, 0, 0, 0, 0],
[0, 0, 0, 0, 0],
[0, 0, 0, 0, 0]]

Now answer for the attached image.Ensure that your output strictly follows this format:
"""

text_msg_black = """You are a model specialized in interpreting data from chess board images. Your task is think carefuly & predict the best move for the current position. You will play as black in this position.

Return the move in SAN notation. Return the move in the following format \\boxed{move}. For example, if the best move is e4, return \\boxed{e4}.

Answer:
"""

text_msg_white = """You are a model specialized in interpreting data from chess board images. Your task is think carefuly & predict the best move for the current position. You will play as white in this position.

Return the move in SAN notation. Return the move in the following format \\boxed{move}. For example, if the best move is e4, return \\boxed{e4}.

Answer: 
"""

system_message_move = {"Black": text_msg_black, "White": text_msg_white}


def return_dataset_images(dataset_path,create_msg,main_prompt, length = 50):

    all_msgs = []
    for i in range(length):

        img_path = f"{dataset_path}/screenshot_{i}.png"
        board_pos_path = f"{dataset_path}/board_pos_{i}.npy"

        msg = create_msg(img_path, board_pos_path, main_prompt)
        all_msgs.append(msg)

    return all_msgs

def return_dataset_images_move(dataset_path, create_msg, main_prompt, length=50):
    all_conversations = []
    
    with open(f'{dataset_path}/best_moves.pkl', 'rb') as f:
        best_moves = pickle.load(f)
    with open(f'{dataset_path}/color.pkl', 'rb') as f:
        color_data = pickle.load(f)

    for i in range(length):
        img_path = f"{dataset_path}/screenshot_{i}.png"
        conversation = create_msg(img_path, best_moves[i], main_prompt[color_data[i]])
        all_conversations.append(conversation)

    return all_conversations

def convert_npy_to_str(npy_path):
    
    arr = np.load(npy_path)
    python_list_of_lists = arr.tolist()
    string_rows = [str(row) for row in python_list_of_lists]
    readable_string_representation = "[" + ",\n".join(string_rows) + "]"
    return readable_string_representation

    



def create_msg(img_path, board_pos, prompt):
    # Load image here when creating dataset, not in collate function
    image = Image.open(img_path)
    if image.mode != "RGB":
        image = image.convert("RGB")
    
    # Return complete conversation format like smolvlm_example.py and sft_qwen.py
    return [
        {
            "role": "system",
            "content": [{"type": "text", "text": MAIN_SYSTEM_MESSAGE}],
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},  # Use "image" key with PIL Image object
                {"type": "text", "text": prompt},
            ]
        },
        {
            "role": "assistant",
            "content": [{"type": "text", "text": convert_npy_to_str(board_pos)}]
        }
    ]

def create_msg_move(img_path, pkl, prompt):
    move = pkl[0]

    image = Image.open(img_path)
    if image.mode != "RGB":
        image = image.convert("RGB")
    
    # Return complete conversation instead of separate prompt/labels
    return [
        {
            "role": "system",
            "content": [{"type": "text", "text": MAIN_SYSTEM_MESSAGE}],
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ]
        },
        {
            "role": "assistant", 
            "content": [
                {"type": "text", "text": f"\\boxed{{{move}}}"}
            ]
        }
    ]

#32,28,216

# image_token_id = processor.tokenizer.additional_special_tokens_ids[
#             processor.tokenizer.additional_special_tokens.index("<image>")]

# def collate_fn(examples):
#   texts = []
#   images = []
#   for example in examples:
#       image = example["image"]
#       if image.mode != 'RGB':
#         image = image.convert('RGB')
#       question = example["question"]
#       answer = example["multiple_choice_answer"]
#       messages = [
#           {
#               "role": "user",
#               "content": [
#                   {"type": "text", "text": "Answer briefly."},
#                   {"type": "image"},
#                   {"type": "text", "text": question}
#               ]
#           },
#           {
#               "role": "assistant",
#               "content": [
#                   {"type": "text", "text": answer}
#               ]
#           }
#       ]
#       text = processor.apply_chat_template(messages, add_generation_prompt=False)
#       texts.append(text.strip())
#       images.append([image])

#   batch = processor(text=texts, images=images, return_tensors="pt", padding=True)
#   labels = batch["input_ids"].clone()
#   labels[labels == processor.tokenizer.pad_token_id] = -100
#   labels[labels == image_token_id] = -100
#   batch["labels"] = labels

#   return batch

def collate_fn_v2(examples):
    texts = [processor.apply_chat_template(example, tokenize=False) for example in examples]

    image_inputs = []
    for example in examples:
        # Get image from user message content 
        user_message = example[1]  # The user message (after system message)
        for content_item in user_message["content"]:
            if content_item["type"] == "image":
                image = content_item["image"]  # PIL Image object
                if image.mode != "RGB":
                    image = image.convert("RGB")
                image_inputs.append([image])
                break

    batch = processor(text=texts, images=image_inputs, return_tensors="pt", padding=True)
    labels = batch["input_ids"].clone()
    
    # Get image token id for masking
    image_token_id = processor.tokenizer.additional_special_tokens_ids[
        processor.tokenizer.additional_special_tokens.index("<image>")
    ]
    
    # Mask padding tokens and image tokens
    labels[labels == processor.tokenizer.pad_token_id] = -100
    labels[labels == image_token_id] = -100
    
    # SIMPLIFIED MASKING: Find assistant response tokens and only train on those
    for i, example in enumerate(examples):
        assistant_response = example[2]["content"][0]["text"]  # Assistant's response text
        
        # Tokenize just the assistant response to find its tokens
        assistant_tokens = processor.tokenizer.encode(assistant_response, add_special_tokens=False)
        assistant_tensor = torch.tensor(assistant_tokens)
        
        # Find where these tokens appear in the full sequence
        full_tokens = batch["input_ids"][i]
        
        # Simple sliding window to find the assistant response
        assistant_start = None
        for start_idx in range(len(full_tokens) - len(assistant_tensor) + 1):
            if torch.equal(full_tokens[start_idx:start_idx + len(assistant_tensor)], assistant_tensor):
                assistant_start = start_idx
                break
        
        if assistant_start is not None:
            # Mask everything except the assistant response
            labels[i, :assistant_start] = -100
            # Keep the assistant response tokens for training
        else:
            # If we can't find the assistant response, mask everything (safety)
            labels[i, :] = -100
    
    batch["labels"] = labels
    return batch

#! For seperate
def collate_fn(examples):
    texts = [processor.apply_chat_template(example, tokenize=False) for example in examples]

    image_inputs = []
    for example in examples:
        # Get image from user message content (now it's already a PIL Image)
        user_message = example[1]  # The user message (after system message)
        for content_item in user_message["content"]:
            if content_item["type"] == "image":
                image = content_item["image"]  # This is now a PIL Image object
                if image.mode != "RGB":
                    image = image.convert("RGB")
                image_inputs.append([image])
                break

    batch = processor(text=texts, images=image_inputs, return_tensors="pt", padding=True)
    labels = batch["input_ids"].clone()
    
    # Get image token id for masking
    image_token_id = processor.tokenizer.additional_special_tokens_ids[
        processor.tokenizer.additional_special_tokens.index("<image>")
    ]
    
    # Mask padding tokens and image tokens
    labels[labels == processor.tokenizer.pad_token_id] = -100
    labels[labels == image_token_id] = -100
    
    # Mask the system and user parts (everything before assistant response)
    for i, example in enumerate(examples):
        # Get system + user part (everything except assistant response)
        system_user_part = example[:-1]  # Everything except the last (assistant) message
        
        # Apply chat template to system + user part only  
        system_user_text = processor.apply_chat_template(system_user_part, tokenize=False)
        
        # Get the image for this example (now it's already a PIL Image)
        user_message = example[1]
        example_image = None
        for content_item in user_message["content"]:
            if content_item["type"] == "image":
                example_image = content_item["image"]  # Already a PIL Image
                if example_image.mode != "RGB":
                    example_image = example_image.convert("RGB")
                break
        
        # Tokenize system + user part to find its length
        system_user_inputs = processor(
            text=[system_user_text],
            images=[[example_image]] if example_image else None,
            return_tensors="pt",
            padding=True,
        )
        
        system_user_length = system_user_inputs["input_ids"].shape[1]
        
        # Mask everything before the assistant response
        labels[i, :system_user_length] = -100
    
    batch["labels"] = labels
    return batch



# model_path = "HuggingFaceTB/SmolVLM2-2.2B-Instruct"
model_path = args.model_name
processor = AutoProcessor.from_pretrained(model_path)
model = Idefics3ForConditionalGeneration.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    _attn_implementation="flash_attention_2",
    device_map="auto"
)

train_dataset = return_dataset_images_move(args.dataset_name,create_msg_move,system_message_move)

# train_dataset = return_dataset_images_move(args.dataset_name,create_msg_move,system_message_move)

eval_dataset = [train_dataset[0]]

# Configure training arguments
training_args = SFTConfig(
    output_dir=args.save_name,  # Directory to save the model
    num_train_epochs=50,  # Number of training epochs
    per_device_train_batch_size=args.per_device_batch_size,  # Batch size for training
    gradient_accumulation_steps=args.grad_accum_steps,  # Steps to accumulate gradients
    optim="adamw_torch_fused",  # Optimizer type
    learning_rate=5e-5,  # Learning rate for training
    lr_scheduler_type="constant",  # Type of learning rate scheduler
    # Logging and evaluation
    logging_steps=2,  # Steps interval for logging
    save_strategy="steps",  # Strategy for saving the model
    # save_steps=10000,  # Steps interval for saving
    bf16=True,  # Use bfloat16 precision
    tf32=True,  # Use TensorFloat-32 precision
    warmup_ratio=0.03,  # Ratio of total steps for warmup
    # Hub and reporting
    #! eval_strategy="steps",  # Evaluate every N steps
    #! eval_steps=10,  # Change this to whatever N you want
    per_device_eval_batch_size=1,  # Single eval example

    report_to="wandb",  
    dataset_kwargs={"skip_prepare_dataset": True}, # Additional dataset options
)


wandb.init(
    project="chess_vllm",  # change this
    name=args.save_name,  # change this
    config=training_args,
)

# training_args.remove_unused_columns = False  # Keep unused columns in dataset
training_args.remove_unused_columns = False  # Keep unused columns in dataset
# wandb.init(
#     project="qwen2-7b-instruct-trl-sft-ChartQA",  # change this
#     name="qwen2-7b-instruct-trl-sft-ChartQA",  # change this
#     config=training_args,
# )

 
# print(train_dataset[0])

#! Comment this out

# num_samples_to_inspect = 4
# sample_examples_for_collate = [train_dataset[i] for i in range(num_samples_to_inspect)]

# batch_output = collate_fn(sample_examples_for_collate)

# print(batch_output['input_ids'].shape)
# token_ids = batch_output['input_ids'][0]

# token_ids_labels = batch_output['labels'][0]

# tokenizer = AutoTokenizer.from_pretrained(model_path)
# for i, token_id in enumerate(token_ids):
#     print("----")
#     print(i, token_id, tokenizer.decode(token_id))
#     if token_ids_labels[i] != -100:
#         print(i, token_ids_labels[i], tokenizer.decode(token_ids_labels[i]))
#     else:
#         print(i, token_ids_labels[i])
#     print("----")

# assert False

# print(tokenizer.decode(token_ids))
# Inspect input_ids

class EvalOutputCallback(TrainerCallback):
    def __init__(self, processor, eval_example):
        self.processor = processor
        self.eval_example = eval_example
        self.eval_data = []  # Store all eval results
    
    def on_evaluate(self, args, state, control, model=None, **kwargs):
        if model is None:
            return
            
        # Get system + user messages for generation (exclude assistant response)
        prompt_msgs = self.eval_example[:-1]  # Everything except assistant response
        
        model.eval()
        with torch.no_grad():
            inputs = self.processor.apply_chat_template(
                prompt_msgs,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            ).to(model.device, dtype=torch.bfloat16)
            
            prompt_length = inputs['input_ids'].shape[1]
            
            generated_ids = model.generate(**inputs, do_sample=False, max_new_tokens=256)
            generated_ids = generated_ids[:, prompt_length:]
            
            generated_texts = self.processor.batch_decode(
                generated_ids,
                skip_special_tokens=True,
            )
            
        # Print the output
        print(f"\n=== EVAL OUTPUT (Step {state.global_step}) ===")
        print(f"Generated: {generated_texts[0]}")
        print(f"Expected: {self.eval_example[2]['content'][0]['text']}")  # Assistant message content
        print("=" * 50)
        
        # Accumulate data
        generated_text = generated_texts[0]
        expected_text = self.eval_example[2]['content'][0]['text']  # Assistant message content
        self.eval_data.append([state.global_step, generated_text, expected_text])
        
        # Log accumulated table
        if wandb.run is not None:
            wandb.log({
                "eval_results": wandb.Table(
                    columns=["Step", "Generated", "Expected"],
                    data=self.eval_data
                )
            })

# Create the callback
eval_callback = EvalOutputCallback(processor, train_dataset[0])

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    # eval_dataset=eval_dataset,
    data_collator=collate_fn,
    # callbacks=[eval_callback],  # Add the callback here
)

trainer = accelerator.prepare(trainer)
trainer.train()

processor.save_pretrained(training_args.output_dir)