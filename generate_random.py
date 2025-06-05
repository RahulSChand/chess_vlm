from chess_updated import Chess
from PyQt5.QtWidgets import QApplication
import sys
import os
import numpy as np
os.environ["QT_QPA_PLATFORM"] = "offscreen"

app = QApplication([]) 

chess = Chess()

for i in range(1024):
    a = chess.get_screenshot()
    board_pos,_ = chess.set_new_position()
    print(board_pos)
    board_pos_np = np.array(board_pos)
    a = chess.get_screenshot()
    a.save(f"dataset_random_384_large_v2/screenshot_{i}.png", "png")
    np.save(f"dataset_random_384_large_v2/board_pos_{i}.npy", board_pos_np)
