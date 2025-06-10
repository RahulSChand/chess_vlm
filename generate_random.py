from chess_updated import Chess
from PyQt5.QtWidgets import QApplication
import sys
import os
import numpy as np
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Generate random chess positions and screenshots')
    parser.add_argument('--save_folder', type=str, default='dataset_random_384_large_v2',
                      help='Folder to save the generated data')
    parser.add_argument('--dataset_size', type=int, default=1024,
                      help='Number of samples to generate')
    return parser.parse_args()

os.environ["QT_QPA_PLATFORM"] = "offscreen"

def main():
    args = parse_args()
    app = QApplication([]) 
    chess = Chess()

    os.makedirs(args.save_folder, exist_ok=True)

    for i in range(args.dataset_size):
        a = chess.get_screenshot()
        board_pos,_ = chess.set_new_position()
        print(board_pos)
        board_pos_np = np.array(board_pos)
        a = chess.get_screenshot()
        a.save(f"{args.save_folder}/screenshot_{i}.png", "png")
        np.save(f"{args.save_folder}/board_pos_{i}.npy", board_pos_np)

if __name__ == "__main__":
    main()
