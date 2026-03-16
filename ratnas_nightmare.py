import torch
import chess
import random
from chess_player import ChessPlayer
from model import ChessNet
from dataset import board_to_tensor

class RatnasNightmare(ChessPlayer):
    def __init__(self, name):
        super().__init__(name)
        # torch.serialization.add_safe_globals([ChessNet])  # usually not needed with weights_only=False
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # self.net = ChessNet().to(self.device)
        # self.net.load_state_dict(torch.load("ratnasNightmare.pth", weights_only=False, map_location=self.device))
        self.net = torch.load("ratnasNightmare_final(2).pth", weights_only=False, map_location=self.device)
        self.net.to(self.device)  # make sure it's on correct device

    def predict_best_move(self, board: chess.Board) -> chess.Move:
        self.net.eval()
        with torch.no_grad():
            # 1. Convert board → tensor (B=1, C=12, H=8, W=8)
            x = torch.tensor(board_to_tensor(board), dtype=torch.float32
                             ).permute(2, 0, 1).unsqueeze(0).to(self.device)

            # 2. Forward → (1, 4096) logits
            logits = self.net(x)          # shape: (1, 4096)
            probs = torch.softmax(logits, dim=1)[0]   # (4096,)

        legal_moves = list(board.legal_moves)
        if not legal_moves:
            return None

        # Compute policy value for each legal move
        legal_probs = []
        for move in legal_moves:
            # flat index = from_square * 64 + to_square
            idx = move.from_square * 64 + move.to_square
            # ignore promotion for now — your net seems to have no promotion head
            p = probs[idx].item()
            legal_probs.append((p, move))

        if not legal_probs:
            # fallback — should not happen
            print("[WARNING] No legal moves could be scored (bug?) → random")
            return random.choice(legal_moves)

        # Pick the legal move with highest probability
        best_p, best_move = max(legal_probs, key=lambda x: x[0])

        if best_p > 0:   # just for debug/info
            print(f"Legal NN move (prob {best_p:.4f})")
        else:
            print("[WARNING] All legal moves have ~0 probability → picking highest anyway")

        return best_move

    def make_move(self, board: chess.Board) -> chess.Move | None:
        move = self.predict_best_move(board)
        if move is None:
            return None
        # Optional: you can still add a safety check
        if move not in board.legal_moves:
            print("[ERROR] Selected move is illegal — this should not happen")
            return random.choice(list(board.legal_moves))
        return move
