[![Python 3](https://img.shields.io/badge/Python-3.x-blue?style=flat&logo=python&logoColor=white)]()
[![License](https://img.shields.io/badge/License-MIT-green.svg)]()

# ChessMates: Testing how good VLMs are at chess

<img width="661" alt="image" src="https://github.com/user-attachments/assets/4d4c72c3-f66d-4085-8817-ce9f2dc304a2" />


# What can the repo do?

[![Dataset](https://img.shields.io/badge/🎲_Dataset-Generation-blue)]()
[![Evaluation](https://img.shields.io/badge/🤖_Model-Evaluation-green)]()
[![Training](https://img.shields.io/badge/🔄_Model-Training-orange)]()
[![Analysis](https://img.shields.io/badge/📊_Performance-Analysis-purple)]()

1. Generate chess dataset for predicting the best move and describing board positions.

2. Evaluate OpenAI (and compatible open-source models via OpenRouter) on the datasets.

3. Train Qwen and SmolVLM on the datasets.

4. Evaluate Qwen and SmolVLM on the datasets.

---


# Installation requirements

Install dependencies:
```bash
pip install -r requirements.txt
```

<details>
<summary>Stockfish Installation Instructions</summary>

To install stockfish (to generate the best moves) follow the following instructions: 

### Download Stockfish 17.1 SOURCE CODE (not binary)
```bash
mkdir stockfish_engine
cd stockfish_engine
wget https://github.com/official-stockfish/Stockfish/archive/refs/tags/sf_17.1.zip
unzip sf_17.1.zip
cd Stockfish-sf_17.1/src
```
### Compile the stable release
```bash
make clean
make -j build ARCH=x86-64-modern
```

### Test it (if this runs it means u are good)
```bash
./stockfish
```

</details>

---

## Generate dataset of chess board images and 8x8 matrix of board positions

```bash
python generate_random.py --save_folder dataset_folder --dataset_size 1024
```

- Generates N images each of size 384x384 and a corresponding 8x8 npy array. 

- To change size of images go to `chess_ui.py` and change `self.setFixedSize(384,384)`

## Generate dataset of chess board images and best move for that position

```bash
python generate_best_move.py --dataset_size 1024 --dataset_name dataset_folder --num_attempts 1200
```


- Generate `dataset_size` images and saves a a single `best_moves.pkl` file made of the 10 best moves for that position and a `color.pkl` file showing which color should play the next move.

- num_attemps is a hyperparameter to control the number of attempts to generate a valid position (a position with atleast 1 valid move). Generally a value just above `dataset_size` is good.

---

# Evaluating models

## Closed source / API models (GPT4 or any openrouter model like Pixtral)

### OpenAI

#### For desctibing the board position task

```bash
python test_openai.py --dataset dataset_folder | tee openai_results.txt
```

- Will save the results in `openai_results.txt`. The reason we save it in a text file is because often the outputs are not neatly formatted and manual post processing is required for fair evaluation.

#### For predicting the best move task

```bash
python test_openai_best.py --folder dataset_folder | tee openai_results_best.txt
```

---

# Training local models

### SmolVLM

```bash
python sft_train.py --model_name "HuggingFaceTB/SmolVLM2-2.2B-Instruct" --save_name "save_weights_here/" --dataset_name "dataset_name/" --task "describe_board"
```

- `task` can be `describe_board` or `best_move`


### Qwen

For simplicity there are 2 scripts for training Qwen. One for describing the board position and one for predicting the best move.

```bash
python sft_qwen.py --model "Qwen/Qwen2-VL-7B-Instruct" --dataset_path "dataset_name/" --limit 1024 --name "save_name/"
```

```bash
python sft_qwen_move.py --model "Qwen/Qwen2-VL-7B-Instruct" --dataset_path "dataset_name/" --limit 1024 --name "save_name/"
```

---

# Evaluating local models

### SmolVLM

```bash
python run_eval_smol.py --model "saved_weights/" --dataset_path "dataset_name/" --limit 128 --task "describe_board"
```

### Qwen

```bash
python run_eval_qwen.py --model "saved_weights/" --dataset_path "dataset_name/" --limit 128
```

```bash
python run_eval_qwen_move.py --model "saved_weights/" --dataset_path "dataset_name/" --limit 128
```

--- 

# Getting accuracy numbers

All the evaluation scripts above print the output. Therefore they should be used as `python command | tee text_results.txt`

This is because outputs often need manual formatting. To get the accuracy numbers once you have the `text_results.txt` use the following script:

### For describe board task

1) Clean/Process
```bash
python parse_text_describe.py --input "text_results.txt" --output "output.npy"
```

This will process the output.txt and also print what lines can't be processed so that you can manually clean them and rerun the script. Results are saved in `output.npy`

2) Print accuracy: 
```bash
python get_accuracy.py --processed_output "output.npy" --test_dataset "dataset_name/" --task "describe_board"
```

---

### For best move task

1) Clean/Process: 
```bash
python parse_best_move.py --input "text_results.txt" --output "output.pkl"
```

2) Print accuracy: 
```bash
python get_accuracy.py --processed_output "output.pkl" --test_dataset "dataset_name/" --task "best_move"
```
---

## Can img-to-img model generate how the board will look like after the best move?

```bash
python generate_img_diffusion.py --input_img "input_chess_board.png" --save_img "output_chess_board.png"
```

- `input_img` is the image of the board position.
- `output_img` is the image of the board position after the best move (generated by diffusion xl)





















