# written with AI
import numpy as np, torch
from torch.utils.data import IterableDataset
import chess.pgn
import random

# ------------------------------------------------------------------
def load_games_from_pgn(pgn_path: str):
    """Yield successive (board, move) pairs for every half‑move."""
    print("Reading games...")
    counter = 0

    with open(pgn_path, 'r', encoding='utf-8') as f:
        game = chess.pgn.read_game(f)
        while game is not None:
            board = chess.Board()
            for i, move in enumerate(game.mainline()):          # SAN moves
                yield board.copy(), move.move                    # 1st: board; 2nd: Move
                board.push(move.move)                              # advance half‑move

            counter += 1
            game = chess.pgn.read_game(f)

            if (counter >= 10000):
                game = None

    



# ------------------------------------------------------------------
def board_to_tensor(board):
    """Return an (8,8,12) numpy array for a `chess.Board`."""
    PIECE_MAP = {
        chess.PAWN:0,   chess.KNIGHT:1,  chess.BISHOP:2,
        chess.ROOK:3,  chess.QUEEN:4,   chess.KING:5,
        # black pieces share the same indices
    }
    PIECE_COLOUR_MAP = {
        chess.WHITE:0, chess.BLACK:1
    }
    arr = np.zeros((8,8,12), dtype=np.float32)
    for sq in board.piece_map():
        piece_type = board.piece_at(sq).piece_type
        piece_colour = board.piece_at(sq).color
        rank = chess.square_rank(sq)
        file = chess.square_file(sq)
        arr[rank,file,PIECE_MAP[piece_type]+(6*PIECE_COLOUR_MAP[piece_colour])] = 1.0
    return arr

# ------------------------------------------------------------------
def legal_move_to_int(move):
    """Encode a `chess.Move` as an integer in [0,4095]."""
    return move.from_square*64 + move.to_square


class ChessPolicyDataset(IterableDataset):
    """Dataset that streams from a PGN file."""
    def __init__(self, pgn_file: str, buffer_size=100000):
        super().__init__()
        self.pgn_file = pgn_file
        self.buffer_size = buffer_size

    def __iter__(self):
        game_iterator = load_games_from_pgn(self.pgn_file)
        
        buffer = []
        for board, move in game_iterator:
            buffer.append((board, move))
            if len(buffer) >= self.buffer_size:
                random.shuffle(buffer)
                for b, m in buffer:
                    X = board_to_tensor(b)
                    y = legal_move_to_int(m)
                    yield torch.from_numpy(X).permute(2,0,1), torch.tensor(y, dtype=torch.long)
                buffer = []

        # Yield remaining items in the buffer
        if len(buffer) > 0:
            random.shuffle(buffer)
            for b, m in buffer:
                X = board_to_tensor(b)
                y = legal_move_to_int(m)
                yield torch.from_numpy(X).permute(2,0,1), torch.tensor(y, dtype=torch.long)
