from chess_updated import Chess
from PyQt5.QtWidgets import QApplication
import sys
import os
import numpy as np
from stockfish import Stockfish
import chess
import pickle
from tqdm import tqdm
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Generate chess positions with best moves using Stockfish')
    parser.add_argument('--dataset_size', type=int, default=1024,
                      help='Number of samples to generate')
    parser.add_argument('--dataset_name', type=str, default='best_moves_128_with_matrix_train_1024_filter',
                      help='Name of the dataset folder')
    parser.add_argument('--num_attempts', type=int, default=1200,
                      help='Maximum number of attempts to generate valid positions')
    return parser.parse_args()

def uci_to_san(fen, uci_move):
    """Convert UCI move to SAN notation"""
    board = chess.Board(fen)
    move = chess.Move.from_uci(uci_move)
    return board.san(move)

def get_best_k_moves_from_fen(fen,k=10):
    """Get the best 5 moves from a FEN position"""
    
    # Path to your stockfish binary
    stockfish_path = "./stockfish_engine/Stockfish-sf_17.1/src/stockfish"
    
    # Initialize Stockfish with custom settings for better analysis
    stockfish = Stockfish(
        path=stockfish_path,
        depth=15,  # Increase depth for better analysis
        parameters={
            "Threads": 2,        # Use more threads if you have them
            "Hash": 512,         # Use more memory for better analysis
            "MultiPV": k         # This is key - analyze top 5 moves
        }
    )
    
    # Set the position
    stockfish.set_fen_position(fen)
    
    # Get top 5 moves
    top_moves = stockfish.get_top_moves(k)
    
    return top_moves

def check_whose_turn(fen):
    """Check whose turn it is from FEN"""
    fen_parts = fen.split(' ')
    turn = fen_parts[1]  # 'w' or 'b'
    return "White" if turn == 'w' else "Black"

def main():
    args = parse_args()
    os.environ["QT_QPA_PLATFORM"] = "offscreen"

    app = QApplication([])
    chess_game = Chess()

    temp_stockfish = Stockfish(path="./stockfish_engine/Stockfish-sf_17.1/src/stockfish")

    board_pos_np = []
    color_arr = []

    os.makedirs(args.dataset_name, exist_ok=True)

    count = 0

    for i in tqdm(range(args.num_attempts)):
        a = chess_game.get_screenshot()
        matrix_board, fen = chess_game.set_legal_position()

        temp_stockfish.set_fen_position(fen)

        top_moves = get_best_k_moves_from_fen(fen)
        color = check_whose_turn(fen)
        
        san_moves = [uci_to_san(fen, move['Move']) for move in top_moves]

        if len(san_moves) < 1:
            continue

        count += 1
        board_pos_np.append(san_moves)
        color_arr.append(color)
        
        matrix_np = np.array(matrix_board[0])
        print(matrix_np)

        a = chess_game.get_screenshot()
        a.save(f"{args.dataset_name}/screenshot_{i}.png", "png")
        np.save(f"{args.dataset_name}/board_pos_{i}.npy", matrix_np)

        if count == args.dataset_size:
            break

    with open(f'{args.dataset_name}/best_moves.pkl', 'wb') as f:
        pickle.dump(board_pos_np, f)

    with open(f'{args.dataset_name}/color.pkl', 'wb') as f:
        pickle.dump(color_arr, f)

if __name__ == "__main__":
    main()