# hackathon_bot.py
#one of few files made with ai
import chess
import chess.polyglot
import torch
import torch.nn as nn
import numpy as np
import os
import sys

# ── Path setup ────────────────────────────────────────────────────────────────
# Assumes your Chess-Bot folder is next to the hackathon repo
# Change this path to wherever your files are
BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
BOOK_PATH    = os.path.join(BASE_DIR, "openingBook", "Perfect2023.bin")
WEIGHTS_PATH = os.path.join(BASE_DIR, "evaluator.pt")
SF_PATH      = os.path.join(BASE_DIR, "stockfish.exe")

# ── Import ChessPlayer from hackathon framework ───────────────────────────────
# This is the base class the framework requires
from chess_player import ChessPlayer


# ── Neural network — must match train.py exactly ──────────────────────────────
class Evaluator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(12 * 8 * 8, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Tanh(),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


# ── Board encoding — must match convert_pgn.py exactly ───────────────────────
def board_to_tensor(board):
    tensor = np.zeros((12, 8, 8), dtype=np.float32)
    piece_idx = {
        (chess.PAWN,   True):  0, (chess.PAWN,   False): 6,
        (chess.KNIGHT, True):  1, (chess.KNIGHT, False): 7,
        (chess.BISHOP, True):  2, (chess.BISHOP, False): 8,
        (chess.ROOK,   True):  3, (chess.ROOK,   False): 9,
        (chess.QUEEN,  True):  4, (chess.QUEEN,  False): 10,
        (chess.KING,   True):  5, (chess.KING,   False): 11,
    }
    for sq, p in board.piece_map().items():
        r, c = divmod(sq, 8)
        tensor[piece_idx[(p.piece_type, p.color)]][r][c] = 1.0
    return tensor

def encode_board(board):
    t = board_to_tensor(board)
    if board.turn == chess.BLACK:
        t = np.flip(t, axis=2).copy()
        t = t[[6,7,8,9,10,11,0,1,2,3,4,5]]
    return t


# ── Main bot class ────────────────────────────────────────────────────────────
class HackathonBot(ChessPlayer):
    """
    Your chess bot wrapped for the hackathon framework.
    Implements make_move(board) which the framework calls.
    Everything else is your existing search + evaluation.
    """

    def __init__(self, name="RLBot"):
        super().__init__(name)

        # Load neural network
        self.device = torch.device(
            "cuda"
        )
        self.model = Evaluator().to(self.device)

        try:
            self.model.load_state_dict(torch.load(
                WEIGHTS_PATH,
                map_location = self.device,
                weights_only = True
            ))
            self.model.eval()
            print(f"[{name}] Model loaded from {WEIGHTS_PATH}")
        except FileNotFoundError:
            print(f"[{name}] ERROR: evaluator.pt not found at {WEIGHTS_PATH}")
            raise

        # Evaluation cache — persists across moves in same game
        self.eval_cache = {}

        # Load Stockfish for move ordering
        try:
            import chess.engine
            self.sf_engine = chess.engine.SimpleEngine.popen_uci(SF_PATH)
            print(f"[{name}] Stockfish loaded")
        except Exception as e:
            self.sf_engine = None
            print(f"[{name}] WARNING: Stockfish not available — {e}")

        print(f"[{name}] Ready on {self.device}")

    def initialize(self, color):
        """
        Called by framework before each game
        color = chess.WHITE or chess.BLACK
        """
        self.color = color
        self.eval_cache.clear()   # fresh cache each game
        self.move_number = 0

    # ── Neural net evaluation ─────────────────────────────────────────────────
    def nn_score(self, board):
        if board.is_checkmate():             return -100000
        if board.is_stalemate():             return 0
        if board.is_insufficient_material(): return 0

        key = board.fen()
        if key in self.eval_cache:
            return self.eval_cache[key]

        t = encode_board(board)
        x = torch.tensor(t).unsqueeze(0).to(self.device)
        with torch.no_grad():
            score = self.model(x).item() * 1000

        self.eval_cache[key] = score
        return score

    # ── Quiescence search ─────────────────────────────────────────────────────
    def quiescence(self, board, alpha, beta, maximizing):
        stand_pat = self.nn_score(board)
        if maximizing:
            if stand_pat >= beta:   return beta
            alpha    = max(alpha, stand_pat)
            captures = [m for m in board.legal_moves if board.is_capture(m)]
            for move in captures:
                board.push(move)
                score = self.quiescence(board, alpha, beta, False)
                board.pop()
                alpha = max(alpha, score)
                if alpha >= beta: break
            return alpha
        else:
            if stand_pat <= alpha:  return alpha
            beta     = min(beta, stand_pat)
            captures = [m for m in board.legal_moves if board.is_capture(m)]
            for move in captures:
                board.push(move)
                score = self.quiescence(board, alpha, beta, True)
                board.pop()
                beta  = min(beta, score)
                if beta <= alpha: break
            return beta

    # ── Alpha-beta search ─────────────────────────────────────────────────────
    def alpha_beta(self, board, depth, alpha, beta, maximizing):
        if board.is_game_over():
            return self.nn_score(board)
        if depth == 0:
            return self.quiescence(board, alpha, beta, maximizing)

        moves = sorted(
            board.legal_moves,
            key=lambda m: (board.is_capture(m), board.gives_check(m)),
            reverse=True
        )
        if maximizing:
            value = -float('inf')
            for move in moves:
                board.push(move)
                value = max(value, self.alpha_beta(
                    board, depth-1, alpha, beta, False))
                board.pop()
                alpha = max(alpha, value)
                if beta <= alpha: break
            return value
        else:
            value = float('inf')
            for move in moves:
                board.push(move)
                value = min(value, self.alpha_beta(
                    board, depth-1, alpha, beta, True))
                board.pop()
                beta  = min(beta, value)
                if beta <= alpha: break
            return value

    # ── Stockfish move ordering ───────────────────────────────────────────────
    def get_move_order(self, board):
        if self.sf_engine is None:
            return sorted(
                board.legal_moves,
                key=lambda m: (board.is_capture(m), board.gives_check(m)),
                reverse=True
            )
        legal = list(board.legal_moves)
        if not legal:
            return []
        try:
            result  = self.sf_engine.analyse(
                board,
                chess.engine.Limit(depth=1),
                multipv=min(len(legal), 20)
            )
            ordered = [info["pv"][0] for info in result if "pv" in info]
            seen    = {m.uci() for m in ordered}
            for m in legal:
                if m.uci() not in seen:
                    ordered.append(m)
            return ordered
        except Exception:
            return sorted(
                legal,
                key=lambda m: (board.is_capture(m), board.gives_check(m)),
                reverse=True
            )

    # ── Opening book ──────────────────────────────────────────────────────────
    def get_book_move(self, board):
        try:
            with chess.polyglot.open_reader(BOOK_PATH) as reader:
                entry = max(
                    reader.find_all(board),
                    key=lambda e: e.weight
                )
                return entry.move
        except Exception:
            return None

    # ── Main move selection ───────────────────────────────────────────────────
    def search(self, board, depth=3):
        ordered_moves = self.get_move_order(board)
        if not ordered_moves:
            return None

        best, best_val = None, -float('inf')
        alpha          = -float('inf')

        for move in ordered_moves:
            board.push(move)
            val = self.alpha_beta(
                board, depth-1, alpha, float('inf'), False
            )
            board.pop()
            if val > best_val:
                best_val, best = val, move
            alpha = max(alpha, best_val)

        return best

    # ── THIS IS WHAT THE FRAMEWORK CALLS ─────────────────────────────────────
    def make_move(self, board):
        """
        Called by SecureBotWrapper for every move.
        Must return a chess.Move object.
        Must not modify board state permanently.
        Must not crash — all errors handled internally.
        """
        try:
            self.move_number += 1
            legal = list(board.legal_moves)

            if not legal:
                return None

            # 1. Opening book — first ~15 moves
            book_move = self.get_book_move(board)
            if book_move and book_move in board.legal_moves:
                return book_move

            # 2. Alpha-beta search with dynamic depth
            # Framework doesn't pass time remaining
            # so use move number as proxy for depth
            # Early game: depth 2 (many pieces, slower)
            # Endgame:    depth 3 (fewer pieces, faster)
            piece_count = len(board.piece_map())
            if piece_count > 20:
                depth = 2   # opening/middlegame — lots of pieces
            elif piece_count > 10:
                depth = 3   # middlegame/endgame
            else:
                depth = 4   # endgame — few pieces, fast search

            move = self.search(board, depth=depth)

            if move and move in board.legal_moves:
                return move

            # 3. Fallback — return first legal move
            # Should never reach here but framework
            # cannot crash so always have a fallback
            return legal[0]

        except Exception as e:
            # Framework logs crashes — return safe fallback
            print(f"[{self.name}] Error in make_move: {e}")
            legal = list(board.legal_moves)
            return legal[0] if legal else None

    def __del__(self):
        """Clean up Stockfish process when bot is garbage collected"""
        try:
            if self.sf_engine:
                self.sf_engine.quit()
        except Exception:
            pass