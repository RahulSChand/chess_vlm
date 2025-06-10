import random
import re

import chess
import chess.engine
from PyQt5.QtWidgets import QMainWindow
from test import generate_random_legal_position
from chess_utils.chess_ui import ChessUI

WIN = 101
LOSE = 102
TIE = 103
INVALID_MOVE = 104
IN_PROGRESS = 105
MAX_TRIAL_REACHED = 106
ERROR = 107

class ChessLogic():
    """Pure logic for Chess game."""

    def __init__(self):
        self.game_cfg = None
        self.user_is_white = True
        self.board = chess.Board()
        
        self.turn = 'white' if self.user_is_white else 'black'
        self.moves_history = []

    def get_game_status(self):
        return self.status

    def reset_board(self):
        """Reset the board to initial state."""
        self.board = chess.Board()
        self.status = 0
        self.turn = 'white' if self.user_is_white else 'black'
        self.moves_history = []

    def get_random_state(self):
        """Generate a random game state."""
        self.reset_board()
        num_moves = random.randint(5, 55)
        for _ in range(num_moves):
            legal_moves = list(self.board.legal_moves)
            if not legal_moves:
                break
            move = random.choice(legal_moves)
            self.board.push(move)
            self.moves_history.append(move)

        piece_to_numeric = {
            chess.PAWN: 1,
            chess.KNIGHT: 2,
            chess.BISHOP: 3,
            chess.ROOK: 4,
            chess.QUEEN: 5,
            chess.KING: 6,
            None: 0
        }
        board_matrix = [[0 for _ in range(8)] for _ in range(8)]
        for i in range(64):
            piece = self.board.piece_at(i)
            if piece:
                value = piece_to_numeric[piece.piece_type]
                if piece.color == chess.BLACK:
                    value = -value
                board_matrix[7 - (i // 8)][i % 8] = value

        self._update_game_status()
        return board_matrix

    def parse_e2e(self, lmm_output):
        """Parse e2e output to a move in SAN format."""
        match = re.search(
            r'Movement:\s*([a-hA-H][1-8][a-hA-H][1-8]|[a-hA-H][1-8]|O-O|O-O-O|(?:N|B|R|Q|K)?[a-hA-H]?[1-8]?x?[a-hA-H][1-8](?:=[QRNB])?|(?:N|B|R|Q|K)[a-hA-H][1-8])',  # noqa
            lmm_output,
            re.IGNORECASE)
        if match:
            return match.group(1)
        return INVALID_MOVE


class ChessRenderer(QMainWindow):
    """Renderer for Chess UI."""

    def __init__(self, logic):
        super().__init__()
        self.logic = logic
        self.ui = ChessUI(self, user_is_white=self.logic.user_is_white)
        self.setCentralWidget(self.ui)
        self.ui.position = self.logic.board
        self.ui.reset_board()

    def get_screenshot(self):
        """Generate screenshot of the current board."""
        self.ui.position = self.logic.board
        self.ui.refresh_from_state()
        screenshot = self.ui.grab()
        return screenshot


# @GAME_REGISTRY.register('chess')
class Chess():
    

    def __init__(self):
        self.renderer = None
        # self.engine = chess.engine.SimpleEngine.popen_uci(
        #     '/usr/games/stockfish')
        # self.engine = None
        self.logic = ChessLogic()

    def set_random_pawns(self, num_pawns, board_size=8):
        
        import numpy as np
    
        self.logic.board = chess.Board(fen=None)  # Empty board

        # Ensure we don't try to place more pawns than squares available
        total_squares = board_size * board_size
        if num_pawns > total_squares:
            raise ValueError(f"Cannot place {num_pawns} pawns on a {board_size}x{board_size} board")
        
        # Create empty board matrix
        board_matrix = np.zeros((board_size, board_size), dtype=int)
        
        # Get all possible positions
        all_positions = [(row, col) for row in range(board_size) for col in range(board_size)]
        
        # Randomly select positions for pawns
        pawn_positions = random.sample(all_positions, num_pawns)
        
        # Place pawns (1s) at selected positions
        for row, col in pawn_positions:
            board_matrix[row, col] = 1
        
        # Always update the chess board visualization (map to 8x8 if needed)
        for row, col in pawn_positions:
            # For smaller boards, map to top-left corner of 8x8 board
            if row < 8 and col < 8:
                square_index = chess.square(col, 7 - row)
                self.logic.board.set_piece_at(square_index, chess.Piece(chess.PAWN, chess.BLACK))
        
        # Update renderer if it exists
        if self.renderer is None:
            self.renderer = ChessRenderer(self.logic)
        self.renderer.ui.position = self.logic.board
        self.renderer.ui.refresh_from_state()
        
        return board_matrix
    def return_actual_board(self, fen):

        piece_to_numeric = {
            chess.PAWN: 1, chess.KNIGHT: 2, chess.BISHOP: 3,
            chess.ROOK: 4, chess.QUEEN: 5, chess.KING: 6
        }
        
        # Initialize an 8x8 matrix with zeros
        board_matrix = [[0 for _ in range(8)] for _ in range(8)]

        # Populate the matrix based on the board state
        # Iterate m_row from 0 (rank 8) to 7 (rank 1)
        # Iterate m_col from 0 (file a) to 7 (file h)
        for m_row in range(8):
            for m_col in range(8):
                # Convert matrix coordinates to chess square index
                # chess.square(file_index, rank_index)
                # file_index = m_col
                # rank_index = 7 - m_row (since m_row 0 is 8th rank, which is rank_index 7)
                square_index = chess.square(m_col, 7 - m_row)
                piece = self.logic.board.piece_at(square_index)

                if piece:
                    value = piece_to_numeric[piece.piece_type]
                    if piece.color == chess.BLACK: # Black pieces are negative
                        value = -value
                    board_matrix[m_row][m_col] = value
        
        # Update UI if renderer exists
        if self.renderer is None:
            self.renderer = ChessRenderer(self.logic)
        self.renderer.ui.set_new_position(fen) # The UI update still uses FEN

        return board_matrix, fen


    def set_legal_position(self):
        self.logic.reset_board()
        fen = generate_random_legal_position(self.logic.board)
        return self.return_actual_board(fen), fen




    def generate_random_position(self):
        """Generate a completely random chess position with all 32 pieces placed randomly."""
        # Clear the logic board (not self.board)
        self.logic.board = chess.Board(fen=None)  # Empty board
        
        # Define the exact pieces for a full chess set (16 per side)
        white_pieces = [
            chess.KING,           # 1 king
            chess.QUEEN,          # 1 queen  
            chess.ROOK, chess.ROOK,           # 2 rooks
            chess.BISHOP, chess.BISHOP,       # 2 bishops
            chess.KNIGHT, chess.KNIGHT,       # 2 knights
            chess.PAWN, chess.PAWN, chess.PAWN, chess.PAWN,     # 8 pawns
            chess.PAWN, chess.PAWN, chess.PAWN, chess.PAWN
        ]
        
        black_pieces = [
            chess.KING,           # 1 king
            chess.QUEEN,          # 1 queen
            chess.ROOK, chess.ROOK,           # 2 rooks
            chess.BISHOP, chess.BISHOP,       # 2 bishops  
            chess.KNIGHT, chess.KNIGHT,       # 2 knights
            chess.PAWN, chess.PAWN, chess.PAWN, chess.PAWN,     # 8 pawns
            chess.PAWN, chess.PAWN, chess.PAWN, chess.PAWN
        ]
        
        # Verify we have exactly 16 pieces per side
        assert len(white_pieces) == 16
        assert len(black_pieces) == 16
        
        # Get all 64 squares and shuffle them
        all_squares = list(range(64))
        random.shuffle(all_squares)
        
        # Place all white pieces (first 16 squares)
        for i, piece_type in enumerate(white_pieces):
            square = all_squares[i]
            self.logic.board.set_piece_at(square, chess.Piece(piece_type, chess.WHITE))
            
        # Place all black pieces (next 16 squares)  
        for i, piece_type in enumerate(black_pieces):
            square = all_squares[i + 16]  # Offset by 16 to use squares 16-31
            self.logic.board.set_piece_at(square, chess.Piece(piece_type, chess.BLACK))
            
        # Set a random turn
        self.logic.board.turn = random.choice([chess.WHITE, chess.BLACK])
        
        return self.logic.board.fen()

    def __del__(self):
        """Cleanup engine resources."""
        if hasattr(self, 'engine'):
            self.engine.quit()

    def get_screenshot(self):
        if self.renderer is None:
            self.renderer = ChessRenderer(self.logic)
        return self.renderer.get_screenshot()
    

    def get_screenshot_custom(self, board_size=8):
        """
        Get a screenshot cropped to the specified board size.
        For boards smaller than 8x8, it crops to show only the top-left portion.
        """
        if self.renderer is None:
            self.renderer = ChessRenderer(self.logic)
        
        # Get full 8x8 screenshot
        screenshot = self.renderer.get_screenshot()
        
        if board_size == 8:
            return screenshot
        
        # Convert to QPixmap for cropping
        pixmap = screenshot
        
        # Calculate crop dimensions
        # The UI is 384x384 pixels with 10x10 grid (8x8 board + 2 rows/cols for labels)
        # Each grid cell is approximately 384/10 = 38.4 pixels
        cell_size = 384 / 10
        
        # New code (no labels):
        crop_width = int(cell_size * board_size)  # Just the board squares
        crop_height = int(cell_size * board_size)  # Just the board squares
        start_x = int(cell_size)  # Skip the left label column
        start_y = int(cell_size)  # Skip the top label row
        
        cropped_pixmap = pixmap.copy(start_x, start_y, crop_width, crop_height)
        
        return cropped_pixmap
    
    def set_new_position(self):

        # This will update self.logic.board to a new random position and return its FEN
        fen = self.generate_random_position()

        return self.return_actual_board(fen)

        # Define the mapping from piece type and color to integer
        

    def input_move(self, move):
        return self.logic.input_move(move)

    def get_game_status(self):
        return self.logic.get_game_status()

    def get_random_state(self):
        return self.logic.get_random_state()

    def get_rule_state(self):
        return self.logic.get_rule_state()