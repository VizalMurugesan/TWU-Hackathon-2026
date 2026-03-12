# bots.py
import chess
import random
from chess_player import ChessPlayer

# --- Bot 1: Random Mover ---
class RandomBot(ChessPlayer):
    def make_move(self, board):
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            return None
        return random.choice(legal_moves)

# --- Bot 2: Pacifist (Avoids captures) ---
class PacifistBot(ChessPlayer):
    def make_move(self, board):
        legal_moves = list(board.legal_moves)
        if not legal_moves: return None
        
        # Try to find a move that is NOT a capture
        quiet_moves = [m for m in legal_moves if not board.is_capture(m)]
        
        if quiet_moves:
            return random.choice(quiet_moves)
        else:
            return random.choice(legal_moves)
        
class HumanPlayer(ChessPlayer):
    def __init__(self, name):
        super().__init__(name)
        self.is_human = True # Flag ensures GameManager waits for UI input

    def make_move(self, board):
        return None # Human moves are handled by UI events, not this function
    
class FreezerBot(ChessPlayer):
    def make_move(self, board):
        import time
        time.sleep(100) # The UI is now dead for 100 seconds
        return list(board.legal_moves)[0]

class CrasherBot(ChessPlayer):
    def make_move(self, board):
        return 1 / 0 # ZeroDivisionError -> Game Over for everyone
    

class DummyGeminiBot(ChessPlayer):
    def make_move(self, board):
        return None # Placeholder for Gemini Bot integration
    
# bots.py (Add this new class)

# ... (keep your existing imports and classes) ...

# --- Bot 3: Smart Bot (Minimax + AlphaBeta + Positional Heuristics) ---
class SmartBot(ChessPlayer):
    def __init__(self, name, depth=3):
        super().__init__(name)
        self.depth = depth
        
        # Piece values (Standard/Simplified)
        self.piece_values = {
            chess.PAWN: 100,
            chess.KNIGHT: 320,
            chess.BISHOP: 330,
            chess.ROOK: 500,
            chess.QUEEN: 900,
            chess.KING: 20000
        }

        # Piece-Square Tables (PST) - Simplified for mid-game
        # Higher numbers = better squares for that piece
        self.pawntable = [
            0,  0,  0,  0,  0,  0,  0,  0,
            50, 50, 50, 50, 50, 50, 50, 50,
            10, 10, 20, 30, 30, 20, 10, 10,
            5,  5, 10, 25, 25, 10,  5,  5,
            0,  0,  0, 20, 20,  0,  0,  0,
            5, -5,-10,  0,  0,-10, -5,  5,
            5, 10, 10,-20,-20, 10, 10,  5,
            0,  0,  0,  0,  0,  0,  0,  0
        ]
        self.knighttable = [
            -50,-40,-30,-30,-30,-30,-40,-50,
            -40,-20,  0,  0,  0,  0,-20,-40,
            -30,  0, 10, 15, 15, 10,  0,-30,
            -30,  5, 15, 20, 20, 15,  5,-30,
            -30,  0, 15, 20, 20, 15,  0,-30,
            -30,  5, 10, 15, 15, 10,  5,-30,
            -40,-20,  0,  5,  5,  0,-20,-40,
            -50,-40,-30,-30,-30,-30,-40,-50
        ]
        # (Other tables omitted for brevity, but Knights/Pawns are most critical for structure)

    def make_move(self, board):
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            return None

        # Start the Alpha-Beta Search
        best_move = None
        best_value = -float('inf')
        alpha = -float('inf')
        beta = float('inf')

        # Move Ordering: Search captures first to improve pruning efficiency
        ordered_moves = sorted(legal_moves, key=lambda move: 1 if board.is_capture(move) else 0, reverse=True)

        for move in ordered_moves:
            board.push(move)
            # We look for the minimum value the opponent can force (minimax)
            board_value = -self.alpha_beta(board, self.depth - 1, -beta, -alpha)
            board.pop()

            if board_value > best_value:
                best_value = board_value
                best_move = move
            
            # Alpha update
            alpha = max(alpha, board_value)

        return best_move

    def alpha_beta(self, board, depth, alpha, beta):
        # Base case: Depth reached or game over
        if depth == 0 or board.is_game_over():
            return self.evaluate_board(board)

        legal_moves = list(board.legal_moves)
        
        # Move Ordering (Captures first)
        ordered_moves = sorted(legal_moves, key=lambda move: 1 if board.is_capture(move) else 0, reverse=True)

        for move in ordered_moves:
            board.push(move)
            score = -self.alpha_beta(board, depth - 1, -beta, -alpha)
            board.pop()

            if score >= beta:
                return beta  # Snip (Pruning)
            if score > alpha:
                alpha = score
                
        return alpha

    def evaluate_board(self, board):
        if board.is_checkmate():
            if board.turn: return -99999 # Black wins (Current turn is White)
            else: return 99999 # White wins
        
        if board.is_stalemate() or board.is_insufficient_material():
            return 0

        evaluation = 0
        
        # Calculate Material and Position
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                value = self.piece_values[piece.piece_type]
                
                # Positional adjustments (PST)
                pst_value = 0
                if piece.piece_type == chess.PAWN:
                    pst_value = self.pawntable[square] if piece.color == chess.WHITE else self.pawntable[chess.square_mirror(square)]
                elif piece.piece_type == chess.KNIGHT:
                    pst_value = self.knighttable[square] if piece.color == chess.WHITE else self.knighttable[chess.square_mirror(square)]

                if piece.color == chess.WHITE:
                    evaluation += (value + pst_value)
                else:
                    evaluation -= (value + pst_value)

        # Return score relative to the side to move
        return evaluation if board.turn == chess.WHITE else -evaluation