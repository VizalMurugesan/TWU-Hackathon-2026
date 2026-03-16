# Features:
#   - Proper Zobrist hashing (fast TT lookups)
#   - Transposition table with EXACT/LOWER/UPPER bound flags
#   - Negamax + Principal Variation Search (PVS)
#   - Iterative Deepening + Aspiration Windows
#   - Time management (hard 4.5s cutoff, soft limit)
#   - Move ordering: TT best move, MVV-LVA, SEE, Killer heuristic, History heuristic
#   - Null-move pruning (R = 2 + depth//4)
#   - Late Move Reductions (LMR)
#   - Quiescence search with delta pruning
#   - Evaluation: optional NNUE (HalfKP, 3-layer net) or classical eval
#   - Classical: Material + tapered PSTs, pawn structure, king safety, mobility, etc.
#   - Embedded opening book (major ECO lines)

import chess
import time
import math
import random
from chess_player import ChessPlayer

# ─────────────────────────────────────────────────────────────────────────────
# Zobrist hashing
# ─────────────────────────────────────────────────────────────────────────────

_rng = random.Random(0xC0FFEE_DEADBEEF)

# piece_type is 1-6 (PAWN=1..KING=6), color is 0/1 → index = (piece_type-1)*2 + color
_ZOB_PIECES = [[_rng.getrandbits(64) for _ in range(64)] for _ in range(12)]
_ZOB_SIDE   = _rng.getrandbits(64)
_ZOB_CASTLE = [_rng.getrandbits(64) for _ in range(16)]
_ZOB_EP     = [_rng.getrandbits(64) for _ in range(9)]   # files 0-7, index 8 = no ep

def _zobrist_hash(board: chess.Board) -> int:
    h = 0
    for sq, piece in board.piece_map().items():
        idx = (piece.piece_type - 1) * 2 + (0 if piece.color == chess.WHITE else 1)
        h ^= _ZOB_PIECES[idx][sq]
    if board.turn == chess.BLACK:
        h ^= _ZOB_SIDE
    h ^= _ZOB_CASTLE[board.castling_rights & 0xF]
    if board.ep_square is not None:
        h ^= _ZOB_EP[chess.square_file(board.ep_square)]
    else:
        h ^= _ZOB_EP[8]
    return h

# ─────────────────────────────────────────────────────────────────────────────
# Transposition Table
# ─────────────────────────────────────────────────────────────────────────────

TT_EXACT = 0
TT_LOWER = 1   # alpha-cutoff (score is a lower bound)
TT_UPPER = 2   # beta-cutoff  (score is an upper bound)

TT_SIZE = 1 << 22  # ~4M entries, ~200 MB in CPython dicts – use a plain dict instead
# We use a plain dict and cap at TT_SIZE by occasionally clearing; simple & fast.

INF = 10_000_000
MATE_SCORE = 9_000_000
MATE_THRESHOLD = MATE_SCORE - 1000

# ─────────────────────────────────────────────────────────────────────────────
# Opening Book (embedded Python dict: FEN -> list of UCI move strings)
# Keys use only the first 4 FEN fields (position + side + castling + ep)
# ─────────────────────────────────────────────────────────────────────────────

def _fen_key(board: chess.Board) -> str:
    """First 4 fields of FEN (ignores clock counts)."""
    return " ".join(board.fen().split()[:4])

# fmt: off
OPENING_BOOK: dict[str, list[str]] = {
    # ---- Start position ----
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq -": ["e2e4", "d2d4", "g1f3", "c2c4"],

    # ---- After 1.e4 ----
    "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3": ["e7e5", "c7c5", "e7e6", "c7c6", "d7d6"],

    # ---- After 1.d4 ----
    "rnbqkbnr/pppppppp/8/8/3P4/8/PPP1PPPP/RNBQKBNR b KQkq d3": ["d7d5", "g8f6", "e7e6", "c7c5"],

    # ---- After 1.Nf3 ----
    "rnbqkbnr/pppppppp/8/8/8/5N2/PPPPPPPP/RNBQKB1R b KQkq -": ["d7d5", "g8f6", "c7c5", "e7e6"],

    # ---- After 1.c4 (English) ----
    "rnbqkbnr/pppppppp/8/8/2P5/8/PP1PPPPP/RNBQKBNR b KQkq c3": ["e7e5", "c7c5", "g8f6", "e7e6"],

    # ---- 1.e4 e5 (Open games) ----
    "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq e6": ["g1f3", "f2f4", "b1c3"],
    # After 1.e4 e5 2.Nf3
    "rnbqkbnr/pppp1ppp/8/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq -": ["b8c6", "g8f6", "d7d6"],
    # After 1.e4 e5 2.Nf3 Nc6 (Ruy Lopez / Italian / Scotch)
    "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq -": ["f1b5", "f1c4", "d2d4"],
    # Ruy Lopez: 3.Bb5
    "r1bqkbnr/pppp1ppp/2n5/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R b KQkq -": ["a7a6", "g8f6", "f8c5"],
    # Italian: 3.Bc4
    "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq -": ["f8c5", "g8f6", "d7d6"],
    # Scotch: 3.d4
    "r1bqkbnr/pppp1ppp/2n5/4p3/3PP3/5N2/PPP2PPP/RNBQKB1R b KQkq d3": ["e5d4", "d7d6"],

    # ---- 1.e4 c5 (Sicilian) ----
    "rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR w KQkq c6": ["g1f3", "b1c3"],
    # Sicilian after 2.Nf3
    "rnbqkbnr/pp1ppppp/8/2p5/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq -": ["d7d6", "b8c6", "e7e6", "g7g6"],
    # Sicilian Najdorf setup: 1.e4 c5 2.Nf3 d6 3.d4 cxd4 4.Nxd4 Nf6 5.Nc3
    "rnbqkb1r/pp2pppp/3p1n2/8/3NP3/2N5/PPP2PPP/R1BQKB1R b KQkq -": ["a7a6", "g7g6", "e7e6"],

    # ---- 1.d4 d5 (Closed games) ----
    "rnbqkbnr/ppp1pppp/8/3p4/3P4/8/PPP1PPPP/RNBQKBNR w KQkq d6": ["c2c4", "g1f3", "b1c3"],
    # QGD setup: 1.d4 d5 2.c4
    "rnbqkbnr/ppp1pppp/8/3p4/2PP4/8/PP2PPPP/RNBQKBNR b KQkq c3": ["e7e6", "c7c6", "d5c4", "g8f6"],
    # After 2.c4 e6 (QGD)
    "rnbqkbnr/ppp2ppp/4p3/3p4/2PP4/8/PP2PPPP/RNBQKBNR w KQkq -": ["b1c3", "g1f3"],
    # After 2.c4 c6 (Slav)
    "rnbqkbnr/pp2pppp/2p5/3p4/2PP4/8/PP2PPPP/RNBQKBNR w KQkq -": ["g1f3", "b1c3"],

    # ---- 1.d4 Nf6 (Indian defences) ----
    "rnbqkb1r/pppppppp/5n2/8/3P4/8/PPP1PPPP/RNBQKBNR w KQkq -": ["c2c4", "g1f3"],
    # 1.d4 Nf6 2.c4
    "rnbqkb1r/pppppppp/5n2/8/2PP4/8/PP2PPPP/RNBQKBNR b KQkq c3": ["e7e6", "g7g6", "c7c5", "e7e5"],
    # King's Indian: 1.d4 Nf6 2.c4 g6
    "rnbqkb1r/pppppp1p/5np1/8/2PP4/8/PP2PPPP/RNBQKBNR w KQkq -": ["b1c3", "g1f3"],
    # Nimzo-Indian: 1.d4 Nf6 2.c4 e6
    "rnbqkb1r/pppp1ppp/4pn2/8/2PP4/8/PP2PPPP/RNBQKBNR w KQkq -": ["b1c3", "g1f3"],
    # 3.Nc3 → Nimzo-Indian
    "rnbqkb1r/pppp1ppp/4pn2/8/2PP4/2N5/PP2PPPP/R1BQKBNR b KQkq -": ["f8b4", "d7d5", "c7c5"],
}
# fmt: on

# ─────────────────────────────────────────────────────────────────────────────
# Piece values (centipawns)
# ─────────────────────────────────────────────────────────────────────────────

MG_PIECE_VAL = {
    chess.PAWN:   100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK:   500,
    chess.QUEEN:  950,
    chess.KING:   20000,
}
EG_PIECE_VAL = {
    chess.PAWN:   120,
    chess.KNIGHT: 300,
    chess.BISHOP: 320,
    chess.ROOK:   520,
    chess.QUEEN:  930,
    chess.KING:   20000,
}

# Phase weights (used to compute game phase 0=endgame .. 1=midgame)
PHASE_WEIGHTS = {
    chess.PAWN: 0, chess.KNIGHT: 1, chess.BISHOP: 1,
    chess.ROOK: 2, chess.QUEEN: 4, chess.KING: 0,
}
MAX_PHASE = 24  # 2*(4*1 + 4*1 + 4*2 + 2*4)

# ─────────────────────────────────────────────────────────────────────────────
# Piece-Square Tables  (from White's perspective, a1=0 .. h8=63)
# Two sets: midgame and endgame
# ─────────────────────────────────────────────────────────────────────────────

# Helpers: all PSTs are stored with rank 8 first (h8=7 last in row 0).
# python-chess square indices: a1=0, h1=7, a8=56, h8=63
# So index sq directly into a "a1..h8" layout table.

def _mk(rows):
    """Flatten 8 rows (rank-8 first) into an a1-indexed list."""
    flat = []
    for row in reversed(rows):   # reversed: row[0] becomes rank 1
        flat.extend(row)
    return flat

MG_PST = {
    chess.PAWN: _mk([
        [ 0,  0,  0,  0,  0,  0,  0,  0],
        [98,134, 61, 95, 68,126, 34,-11],
        [-6,  7, 26, 31, 65, 56, 25,-20],
        [-14, 13,  6, 21, 23, 12, 17,-23],
        [-27, -2, -5, 12, 17,  6, 10,-25],
        [-26, -4, -4,-10,  3,  3, 33,-12],
        [-35, -1,-20,-23,-15, 24, 38,-22],
        [ 0,  0,  0,  0,  0,  0,  0,  0],
    ]),
    chess.KNIGHT: _mk([
        [-167,-89,-34,-49, 61,-97,-15,-107],
        [ -73,-41, 72, 36, 23, 62,  7, -17],
        [ -47, 60, 37, 65, 84,129, 73,  44],
        [  -9, 17, 19, 53, 37, 69, 18,  22],
        [ -13,  4, 16, 13, 28, 19, 21,  -8],
        [ -23, -9, 12, 10, 19, 17, 25, -16],
        [ -29,-53,-12, -3, -1, 18,-14, -19],
        [-105,-21,-58,-33,-17,-28,-19, -23],
    ]),
    chess.BISHOP: _mk([
        [-29,  4,-82,-37,-25,-42,  7, -8],
        [-26, 16,-18,-13, 30, 59, 18,-47],
        [-16, 37, 43, 40, 35, 50, 37, -2],
        [ -4,  5, 19, 50, 37, 37,  7, -2],
        [ -6, 13, 13, 26, 34, 12, 10,  4],
        [  0, 15, 15, 15, 14, 27, 18, 10],
        [  4, 15, 16,  0,  7, 21, 33,  1],
        [-33,-3,-14,-21,-13,-12,-39,-21],
    ]),
    chess.ROOK: _mk([
        [ 32, 42, 32, 51, 63,  9, 31, 43],
        [ 27, 32, 58, 62, 80, 67, 26, 44],
        [ -5, 19, 26, 36, 17, 45, 61, 16],
        [-24,-11,  7, 26, 24, 35, -8,-20],
        [-36,-26,-12, -1,  9, -7,  6,-23],
        [-45,-25,-16,-17,  3,  0, -5,-33],
        [-44,-16,-20, -9, -1, 11, -6,-71],
        [-19,-13,  1, 17, 16,  7,-37,-26],
    ]),
    chess.QUEEN: _mk([
        [-28,  0, 29, 12, 59, 44, 43, 45],
        [-24,-39, -5,  1,-16, 57, 28, 54],
        [-13,-17,  7,  8, 29, 56, 47, 57],
        [-27,-27,-16,-16, -1, 17, -2,  1],
        [ -9,-26, -9,-10, -2, -4,  3, -3],
        [-14,  2,-11,  2, -2,  2, 14,  5],
        [-35, -8, 11,  2,  8, 15, -3,  1],
        [ -1,-18, -9, 10,-15,-25,-31,-50],
    ]),
    chess.KING: _mk([
        [-65, 23, 16,-15,-56,-34,  2, 13],
        [ 29, -1,-20, -7, -8, -4,-38,-29],
        [ -9, 24,  2,-16,-20,  6, 22,-22],
        [-17,-20,-12,-27,-30,-25,-14,-36],
        [-49, -1,-27,-39,-46,-44,-33,-51],
        [-14,-14,-22,-46,-44,-30,-15,-27],
        [  1,  7, -8,-64,-43,-16,  9,  8],
        [-15, 36, 12,-54,  8,-28, 24, 14],
    ]),
}

EG_PST = {
    chess.PAWN: _mk([
        [  0,  0,  0,  0,  0,  0,  0,  0],
        [178,173,158,134,147,132,165,187],
        [ 94,100, 85, 67, 56, 53, 82, 84],
        [ 32, 24, 13,  5, -2,  4, 17, 17],
        [ 13,  9, -3, -7, -7, -8,  3, -1],
        [  4,  7, -6,  1,  0, -5, -1, -8],
        [ 13,  8,  8, 10, 13,  0,  2, -7],
        [  0,  0,  0,  0,  0,  0,  0,  0],
    ]),
    chess.KNIGHT: _mk([
        [-58,-38,-13,-28,-31,-27,-63,-99],
        [-25, -8,-25, -2, -9,-25,-24,-52],
        [-24,-20, 10,  9, -1, -9,-19,-41],
        [-17,  3, 22, 22, 22, 11,  8,-18],
        [-18, -6, 16, 25, 16, 17,  4,-18],
        [-23, -3, -1, 15, 10, -3,-20,-22],
        [-42,-20,-10, -5, -2,-20,-23,-44],
        [-29,-51,-23,-15,-22,-18,-50,-64],
    ]),
    chess.BISHOP: _mk([
        [-14,-21,-11, -8, -7, -9,-17,-24],
        [ -8, -4,  7,-12, -3,-13, -4,-14],
        [  2, -8,  0, -1, -2,  6,  0,  4],
        [ -3,  9, 12,  9, 14, 10,  3,  2],
        [ -6,  3, 13, 19,  7, 10, -3, -9],
        [-12, -3,  8, 10, 13,  3, -7,-15],
        [-14,-18, -7, -1,  4, -9,-15,-27],
        [-23, -9,-23, -5, -9,-16, -5,-17],
    ]),
    chess.ROOK: _mk([
        [ 13, 10, 18, 15, 12, 12,  8,  5],
        [ 11, 13, 13, 11, -3,  3,  8,  3],
        [  7,  7,  7,  5,  4, -3, -5, -3],
        [  4,  3, 13,  1,  2,  1, -1,  2],
        [  3,  5,  8,  4, -5, -6, -8, -11],
        [ -4,  0,-5, -1, -7,-12, -8,-16],
        [-6, -6,  0,  2, -9, -9,-11, -3],
        [-9,  2,  3,-1,-5,-13, 4,-20],
    ]),
    chess.QUEEN: _mk([
        [ -9, 22, 22, 27, 27, 19, 10, 20],
        [-17, 20, 32, 41, 58, 25, 30,  0],
        [-20,  6,  9, 49, 47, 35, 19,  9],
        [  3, 22, 24, 45, 57, 40, 57, 36],
        [-18, 28, 19, 47, 31, 34, 39, 23],
        [-16,-27, 15,  6,  9, 17, 10,  5],
        [-22,-23,-30,-16,-16,-23,-36,-32],
        [-33,-28,-22,-43, -5,-32,-20,-41],
    ]),
    chess.KING: _mk([
        [-74,-35,-18,-18,-11, 15,  4,-17],
        [-12, 17, 14, 17, 17, 38, 23, 11],
        [ 10, 17, 23, 15, 20, 45, 44, 13],
        [ -8, 22, 24, 27, 26, 33, 26,  3],
        [-18, -4, 21, 24, 27, 23,  9,-11],
        [-19, -3, 11, 21, 23, 16,  7, -9],
        [-27,-11,  4, 13, 14,  4, -5,-17],
        [-53,-34,-21,-11,-28,-14,-24,-43],
    ]),
}

def _get_pst_value(piece: chess.Piece, sq: int, pst: dict) -> int:
    table = pst[piece.piece_type]
    if piece.color == chess.BLACK:
        sq = sq ^ 56   # flip rank for black
    return table[sq]

# ─────────────────────────────────────────────────────────────────────────────
# SEE (Static Exchange Evaluation) – simplified
# ─────────────────────────────────────────────────────────────────────────────

_SEE_VAL = {
    chess.PAWN: 100, chess.KNIGHT: 300, chess.BISHOP: 300,
    chess.ROOK: 500, chess.QUEEN: 900, chess.KING: 20000,
}

def _see(board: chess.Board, move: chess.Move) -> int:
    """Approximate SEE for a capture move. Returns material swap value."""
    to_sq = move.to_square
    victim = board.piece_at(to_sq)
    # En passant: victim is on a different square (same file, one rank from to_square)
    if victim is None and board.is_en_passant(move):
        ep_captured_rank = chess.square_rank(move.to_square) - 1 if board.turn == chess.WHITE else chess.square_rank(move.to_square) + 1
        victim_sq = chess.square(chess.square_file(move.to_square), ep_captured_rank)
        victim_val = _SEE_VAL[chess.PAWN]
    elif victim is None:
        return 0
    else:
        victim_sq = to_sq
        victim_val = _SEE_VAL.get(victim.piece_type, 0)
    gain = [0] * 32
    gain[0] = victim_val
    attacker_piece = board.piece_at(move.from_square)
    if attacker_piece is None:
        return 0
    d = 0
    # Simulate recaptures on to_sq (after push, our piece is on to_sq)
    b = board.copy(stack=False)
    b.push(move)
    while True:
        attackers = b.attackers(b.turn, to_sq)
        if not attackers:
            break
        d += 1
        # Pick lowest-value attacker
        min_val = INF
        min_sq = None
        for sq in attackers:
            p = b.piece_at(sq)
            if p and _SEE_VAL.get(p.piece_type, 0) < min_val:
                min_val = _SEE_VAL.get(p.piece_type, 0)
                min_sq = sq
        if min_sq is None:
            break
        gain[d] = min_val - gain[d-1]
        # simulate capture
        m = chess.Move(min_sq, to_sq)
        if b.is_legal(m):
            b.push(m)
        else:
            # promotion or complex? just break
            break
    # Negamax back through gains
    while d > 0:
        gain[d-1] = -max(-gain[d-1], gain[d])
        d -= 1
    return gain[0]

# ─────────────────────────────────────────────────────────────────────────────
# Evaluation helpers
# ─────────────────────────────────────────────────────────────────────────────

_FILE_MASKS = [chess.BB_FILES[f] for f in range(8)]
_RANK_MASKS = [chess.BB_RANKS[r] for r in range(8)]

# Cache for pawn structure evaluation: (white_pawns_bb, black_pawns_bb) -> score
_PAWN_HASH: dict[tuple[int, int], int] = {}

def _passed_pawn_bonus(sq: int, color: bool) -> int:
    """Bonus for a pawn (already verified as passed) based on advancement."""
    rank = chess.square_rank(sq)
    if color == chess.WHITE:
        advance = rank  # rank 1=1 .. rank 7=7
    else:
        advance = 7 - rank
    # 0, 5, 10, 20, 40, 60, 80 for ranks 1-7; cap index for robustness
    return [0, 5, 10, 20, 40, 60, 80][min(advance, 6)]

def _evaluate_pawns(board: chess.Board) -> int:
    """
    Returns score from WHITE's perspective.
    Passed pawn bonus, doubled pawn penalty, isolated pawn penalty.
    """
    white_pawns = board.pieces(chess.PAWN, chess.WHITE)
    black_pawns = board.pieces(chess.PAWN, chess.BLACK)

    key = (int(white_pawns), int(black_pawns))
    cached = _PAWN_HASH.get(key)
    if cached is not None:
        return cached

    score = 0

    for sq in white_pawns:
        f = chess.square_file(sq)
        r = chess.square_rank(sq)
        # Doubled
        file_mask = _FILE_MASKS[f]
        if chess.popcount(int(white_pawns) & file_mask) > 1:
            score -= 15
        # Isolated
        adj_mask = 0
        if f > 0: adj_mask |= _FILE_MASKS[f - 1]
        if f < 7: adj_mask |= _FILE_MASKS[f + 1]
        if not (int(white_pawns) & adj_mask):
            score -= 15
        # Passed
        ahead_mask = 0
        for rr in range(r + 1, 8):
            ahead_mask |= _FILE_MASKS[f] & _RANK_MASKS[rr]
            if f > 0: ahead_mask |= _FILE_MASKS[f-1] & _RANK_MASKS[rr]
            if f < 7: ahead_mask |= _FILE_MASKS[f+1] & _RANK_MASKS[rr]
        if not (int(black_pawns) & ahead_mask):
            # Blockader: enemy piece on same file ahead reduces bonus
            bonus = _passed_pawn_bonus(sq, chess.WHITE)
            for rr in range(r + 1, 8):
                blocker_sq = chess.square(f, rr)
                if board.piece_at(blocker_sq) and board.piece_at(blocker_sq).color == chess.BLACK:
                    bonus = bonus // 2  # blockaded
                    break
            score += bonus

    for sq in black_pawns:
        f = chess.square_file(sq)
        r = chess.square_rank(sq)
        file_mask = _FILE_MASKS[f]
        if chess.popcount(int(black_pawns) & file_mask) > 1:
            score += 15
        adj_mask = 0
        if f > 0: adj_mask |= _FILE_MASKS[f - 1]
        if f < 7: adj_mask |= _FILE_MASKS[f + 1]
        if not (int(black_pawns) & adj_mask):
            score += 15
        # Passed
        ahead_mask = 0
        for rr in range(r - 1, -1, -1):
            ahead_mask |= _FILE_MASKS[f] & _RANK_MASKS[rr]
            if f > 0: ahead_mask |= _FILE_MASKS[f-1] & _RANK_MASKS[rr]
            if f < 7: ahead_mask |= _FILE_MASKS[f+1] & _RANK_MASKS[rr]
        if not (int(white_pawns) & ahead_mask):
            bonus = _passed_pawn_bonus(sq, chess.BLACK)
            for rr in range(r - 1, -1, -1):
                blocker_sq = chess.square(f, rr)
                if board.piece_at(blocker_sq) and board.piece_at(blocker_sq).color == chess.WHITE:
                    bonus = bonus // 2
                    break
            score -= bonus

    _PAWN_HASH[key] = score
    return score


def _king_safety(board: chess.Board, color: bool) -> int:
    """Midgame king safety heuristic: pawn shield + attacker penalty."""
    score = 0
    king_sq = board.king(color)
    if king_sq is None:
        return 0
    king_file = chess.square_file(king_sq)
    king_rank = chess.square_rank(king_sq)
    opp = not color
    pawns = board.pieces(chess.PAWN, color)

    # Pawn shield: pawns on the same or one rank ahead on the 3 files near king
    shield_files = [f for f in [king_file-1, king_file, king_file+1] if 0 <= f <= 7]
    shield_rank = king_rank + (1 if color == chess.WHITE else -1)
    shield_count = 0
    for f in shield_files:
        if 0 <= shield_rank <= 7:
            sq = chess.square(f, shield_rank)
            if sq in pawns:
                shield_count += 1
    score += shield_count * 10

    # Count attackers in the king zone (3x3 box); weight by piece type
    attacker_penalty = 0
    attacker_weights = {chess.KNIGHT: 20, chess.BISHOP: 20, chess.ROOK: 40, chess.QUEEN: 80}
    for rank_off in [-1, 0, 1]:
        for file_off in [-2, -1, 0, 1, 2]:
            r = king_rank + rank_off
            f = king_file + file_off
            if 0 <= r <= 7 and 0 <= f <= 7:
                zone_sq = chess.square(f, r)
                if not board.is_attacked_by(opp, zone_sq):
                    continue
                # Sum weight for each opponent piece that attacks this zone square
                for attacker_sq in board.attackers(opp, zone_sq):
                    p = board.piece_at(attacker_sq)
                    if p:
                        attacker_penalty += attacker_weights.get(p.piece_type, 0)

    score -= min(attacker_penalty, 200)   # cap to avoid crazy swings

    # King in center penalty (higher score = safer; so subtract when king exposed)
    if king_file >= 2 and king_file <= 5:  # d/e/f files
        if color == chess.WHITE and king_rank <= 2:
            score -= 15
        elif color == chess.BLACK and king_rank >= 5:
            score -= 15

    # Open file toward king: opponent rook/queen on same file or adjacent
    for f_off in [-1, 0, 1]:
        kf = king_file + f_off
        if 0 <= kf <= 7:
            file_bb = _FILE_MASKS[kf]
            for sq in (board.pieces(chess.ROOK, opp) | board.pieces(chess.QUEEN, opp)):
                if file_bb & (1 << sq):
                    score -= 10
                    break

    return score


# ─────────────────────────────────────────────────────────────────────────────
# NNUE (Efficiently Updatable Neural Network) evaluator
# HalfKP-style: each non-king piece is a feature (piece_type, square, king_square).
# Two halves: white POV (white pieces + white king sq) and black POV (black pieces + black king sq).
# Net: 40960 features -> 64 -> 32 -> 1. Weights initialized small so output ~0 until trained.
# ─────────────────────────────────────────────────────────────────────────────

NNUE_HALF_SIZE = 5 * 64 * 64   # 20480 (piece_type 0..4, piece_sq, king_sq)
NNUE_FEATURE_SIZE = 2 * NNUE_HALF_SIZE  # 40960
NNUE_L1_SIZE = 32   # hidden size (smaller = faster init & inference in Python)
NNUE_L2_SIZE = 32
_NNUE_RNG = random.Random(0xE2026)  # fixed seed for reproducible untrained weights


def _nnue_feature_indices(board: chess.Board) -> list[int]:
    """Return list of active HalfKP feature indices (at most 32: 16 white + 16 black)."""
    indices: list[int] = []
    # Piece type index 0=pawn, 1=knight, 2=bishop, 3=rook, 4=queen (no king)
    pt_to_idx = {chess.PAWN: 0, chess.KNIGHT: 1, chess.BISHOP: 2, chess.ROOK: 3, chess.QUEEN: 4}

    w_king = board.king(chess.WHITE)
    b_king = board.king(chess.BLACK)
    if w_king is None or b_king is None:
        return indices

    # White pieces (from white's perspective): index in [0, NNUE_HALF_SIZE)
    for pt in (chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN):
        if pt not in pt_to_idx:
            continue
        idx_pt = pt_to_idx[pt]
        for sq in board.pieces(pt, chess.WHITE):
            i = idx_pt * 64 * 64 + sq * 64 + w_king
            indices.append(i)

    # Black pieces (from black's perspective): index in [NNUE_HALF_SIZE, NNUE_FEATURE_SIZE)
    for pt in (chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN):
        if pt not in pt_to_idx:
            continue
        idx_pt = pt_to_idx[pt]
        for sq in board.pieces(pt, chess.BLACK):
            i = NNUE_HALF_SIZE + idx_pt * 64 * 64 + sq * 64 + b_king
            indices.append(i)

    return indices


def _nnue_forward(feature_indices: list[int], l1_w: list[list[float]], l1_b: list[float],
                  l2_w: list[list[float]], l2_b: list[float],
                  l3_w: list[float], l3_b: float) -> float:
    """Forward pass: accumulator = sum of L1 rows for active features, then ReLU, L2, ReLU, L3."""
    # L1: accumulator (sum of rows)
    acc = [l1_b[j] for j in range(NNUE_L1_SIZE)]
    for idx in feature_indices:
        if 0 <= idx < len(l1_w):
            for j in range(NNUE_L1_SIZE):
                acc[j] += l1_w[idx][j]
    # ReLU
    for j in range(NNUE_L1_SIZE):
        acc[j] = max(0.0, acc[j])

    # L2
    hidden = [l2_b[j] for j in range(NNUE_L2_SIZE)]
    for j in range(NNUE_L2_SIZE):
        for k in range(NNUE_L1_SIZE):
            hidden[j] += acc[k] * l2_w[k][j]
    for j in range(NNUE_L2_SIZE):
        hidden[j] = max(0.0, hidden[j])

    # L3 (output)
    out = l3_b
    for j in range(NNUE_L2_SIZE):
        out += hidden[j] * l3_w[j]
    return out


def _nnue_init_weights():
    """Initialize NNUE weights with small values so untrained net outputs ~0."""
    scale1 = 0.02
    scale2 = 0.04
    scale3 = 0.5
    l1_w = [[_NNUE_RNG.gauss(0, scale1) for _ in range(NNUE_L1_SIZE)] for _ in range(NNUE_FEATURE_SIZE)]
    l1_b = [_NNUE_RNG.gauss(0, scale1) for _ in range(NNUE_L1_SIZE)]
    l2_w = [[_NNUE_RNG.gauss(0, scale2) for _ in range(NNUE_L2_SIZE)] for _ in range(NNUE_L1_SIZE)]
    l2_b = [_NNUE_RNG.gauss(0, scale2) for _ in range(NNUE_L2_SIZE)]
    l3_w = [_NNUE_RNG.gauss(0, scale3) for _ in range(NNUE_L2_SIZE)]
    l3_b = 0.0
    return (l1_w, l1_b, l2_w, l2_b, l3_w, l3_b)


# Lazy-init global NNUE weights (shared, read-only after init)
_nnue_weights: tuple | None = None


def _get_nnue_weights():
    global _nnue_weights
    if _nnue_weights is None:
        _nnue_weights = _nnue_init_weights()
    return _nnue_weights


def load_nnue_weights(filepath: str) -> bool:
    """
    Load NNUE weights from a binary file (floats in order: l1_w, l1_b, l2_w, l2_b, l3_w, l3_b).
    Returns True if loaded successfully, False otherwise. On success, sets global _nnue_weights
    and subsequent _evaluate_nnue calls use the new weights.
    """
    import struct
    global _nnue_weights
    try:
        with open(filepath, "rb") as f:
            data = f.read()
        n_floats = (NNUE_FEATURE_SIZE * NNUE_L1_SIZE + NNUE_L1_SIZE +
                    NNUE_L1_SIZE * NNUE_L2_SIZE + NNUE_L2_SIZE +
                    NNUE_L2_SIZE * 1 + 1)
        if len(data) < n_floats * 4:
            return False
        floats = list(struct.unpack(f"{n_floats}f", data[: n_floats * 4]))
        idx = 0
        l1_w = [[floats[idx + i * NNUE_L1_SIZE + j] for j in range(NNUE_L1_SIZE)]
                for i in range(NNUE_FEATURE_SIZE)]
        idx += NNUE_FEATURE_SIZE * NNUE_L1_SIZE
        l1_b = floats[idx: idx + NNUE_L1_SIZE]
        idx += NNUE_L1_SIZE
        l2_w = [[floats[idx + i * NNUE_L2_SIZE + j] for j in range(NNUE_L2_SIZE)]
                for i in range(NNUE_L1_SIZE)]
        idx += NNUE_L1_SIZE * NNUE_L2_SIZE
        l2_b = floats[idx: idx + NNUE_L2_SIZE]
        idx += NNUE_L2_SIZE
        l3_w = floats[idx: idx + NNUE_L2_SIZE]
        idx += NNUE_L2_SIZE
        l3_b = floats[idx]
        _nnue_weights = (l1_w, l1_b, l2_w, l2_b, l3_w, l3_b)
        return True
    except Exception:
        return False


def save_nnue_weights(filepath: str) -> bool:
    """Save current NNUE weights to a binary file. Returns True on success."""
    import struct
    w = _get_nnue_weights()
    l1_w, l1_b, l2_w, l2_b, l3_w, l3_b = w
    try:
        floats = []
        for row in l1_w:
            floats.extend(row)
        floats.extend(l1_b)
        for row in l2_w:
            floats.extend(row)
        floats.extend(l2_b)
        floats.extend(l3_w)
        floats.append(l3_b)
        with open(filepath, "wb") as f:
            f.write(struct.pack(f"{len(floats)}f", *floats))
        return True
    except Exception:
        return False


def _evaluate_nnue(board: chess.Board) -> int:
    """
    NNUE evaluation. Returns score in centipawns from SIDE TO MOVE perspective.
    Uses HalfKP feature set and a small 3-layer net. Untrained weights give ~0;
    replace with trained weights for strength.
    """
    if board.is_checkmate():
        return -MATE_SCORE
    if board.is_stalemate() or board.is_insufficient_material():
        return 0

    indices = _nnue_feature_indices(board)
    if not indices:
        return 0

    l1_w, l1_b, l2_w, l2_b, l3_w, l3_b = _get_nnue_weights()
    raw = _nnue_forward(indices, l1_w, l1_b, l2_w, l2_b, l3_w, l3_b)
    # Raw output is from White's perspective (positive = good for White). Convert to side-to-move.
    cp = int(round(raw))
    if board.turn == chess.WHITE:
        return cp
    return -cp


# Set to True to use NNUE; False to use classical eval.
# With untrained (random) weights NNUE outputs ~0 everywhere, so keep False until you load trained weights.
USE_NNUE = False


def _evaluate(board: chess.Board) -> int:
    """
    Full evaluation. Returns score in centipawns from the perspective of
    the SIDE TO MOVE (positive = good for side to move).
    Dispatches to NNUE or classical eval based on USE_NNUE.
    """
    if USE_NNUE:
        return _evaluate_nnue(board)
    return _evaluate_classical(board)


def _evaluate_classical(board: chess.Board) -> int:
    """
    Classical (hand-crafted) evaluation. Same contract as _evaluate.
    Used when USE_NNUE is False or for fallback.
    """
    if board.is_checkmate():
        return -MATE_SCORE
    if board.is_stalemate() or board.is_insufficient_material():
        return 0

    # Compute phase (0.0 = endgame, 1.0 = midgame)
    phase_sum = 0
    mg_score = 0
    eg_score = 0

    for sq, piece in board.piece_map().items():
        phase_sum += PHASE_WEIGHTS.get(piece.piece_type, 0)
        mg_val = (MG_PIECE_VAL[piece.piece_type]
                  + _get_pst_value(piece, sq, MG_PST))
        eg_val = (EG_PIECE_VAL[piece.piece_type]
                  + _get_pst_value(piece, sq, EG_PST))
        sign = 1 if piece.color == chess.WHITE else -1
        mg_score += sign * mg_val
        eg_score += sign * eg_val

    phase = min(phase_sum, MAX_PHASE) / MAX_PHASE   # 1.0 = full midgame

    # Tapered material + position
    score = int(phase * mg_score + (1.0 - phase) * eg_score)

    # Pawn structure
    score += _evaluate_pawns(board)

    # Bishop pair (midgame-weighted)
    if len(board.pieces(chess.BISHOP, chess.WHITE)) >= 2:
        score += int(30 * phase)
    if len(board.pieces(chess.BISHOP, chess.BLACK)) >= 2:
        score -= int(30 * phase)

    # Rook on open / semi-open file
    white_pawns = board.pieces(chess.PAWN, chess.WHITE)
    black_pawns = board.pieces(chess.PAWN, chess.BLACK)
    wp_int = int(white_pawns)
    bp_int = int(black_pawns)
    for sq in board.pieces(chess.ROOK, chess.WHITE):
        f = chess.square_file(sq)
        file_mask = _FILE_MASKS[f]
        if not (wp_int & file_mask) and not (bp_int & file_mask):
            score += 30   # open file
        elif not (wp_int & file_mask):
            score += 15   # semi-open
    for sq in board.pieces(chess.ROOK, chess.BLACK):
        f = chess.square_file(sq)
        file_mask = _FILE_MASKS[f]
        if not (wp_int & file_mask) and not (bp_int & file_mask):
            score -= 30
        elif not (bp_int & file_mask):
            score -= 15

    # Mobility: count attacked squares for bishops, rooks, queens
    if phase > 0.3:
        sliding_types = (chess.BISHOP, chess.ROOK, chess.QUEEN)
        w_mob = sum(
            chess.popcount(board.attacks_mask(sq))
            for pt in sliding_types
            for sq in board.pieces(pt, chess.WHITE)
        )
        b_mob = sum(
            chess.popcount(board.attacks_mask(sq))
            for pt in sliding_types
            for sq in board.pieces(pt, chess.BLACK)
        )
        score += (w_mob - b_mob) * 2

    # King safety (midgame only)
    if phase > 0.5:
        score += int(phase * _king_safety(board, chess.WHITE))
        score -= int(phase * _king_safety(board, chess.BLACK))

    score -= board.halfmove_clock // 2
    TEMPO_BONUS = 12
    return (score + TEMPO_BONUS) if board.turn == chess.WHITE else (-score + TEMPO_BONUS)


# ─────────────────────────────────────────────────────────────────────────────
# The main Bot class
# ─────────────────────────────────────────────────────────────────────────────

class KenYuDai(ChessPlayer):
    """
    strongest Python chess bot.
    Uses:
      - Negamax + PVS
      - Iterative deepening + aspiration windows
      - Transposition table (Zobrist, bound flags)
      - Move ordering: TT, MVV-LVA, SEE, killers, history
      - Null-move pruning  (R=2+depth//4)
      - Late Move Reductions (LMR)
      - Quiescence search
      - Rich evaluation with tapered PSTs, pawn structure, king safety,
        mobility, bishop pair, rook-on-open-file
      - Embedded opening book
    """

    TIME_LIMIT = 4.0      # Fallback soft limit (seconds) when no clock info is given
    HARD_LIMIT_SCALE = 4.0  # hard = soft * HARD_LIMIT_SCALE (used only in fallback/UCI mode)
    # When set_move_time_limit() receives an explicit budget from the game manager,
    # hard_limit is set to soft * 1.05 so the bot always finishes before the 5s kill threshold.

    def __init__(self, name=None, time_limit: float | None = None):
        if name is None:
            name = "KenYuDai"
        super().__init__(name)
        self._soft_limit = time_limit if time_limit is not None else self.TIME_LIMIT
        self._hard_limit = self._soft_limit * self.HARD_LIMIT_SCALE
        self.game_history: list[int] = []   # Zobrist hashes for all positions played so far
        self.last_own_move: chess.Move | None = None   # last move we played (for reversal penalty)
        self._reset_state()

    # Class-level constants kept together
    MAX_KILLERS = 2
    MAX_DEPTH = 64
    TT_MAX_SIZE = 1 << 21
    CONTEMPT = -20
    TIME_CHECK_INTERVAL = 2048

    def _reset_state(self):
        self.tt: dict = {}
        self.killers = [[None, None] for _ in range(self.MAX_DEPTH)]
        self.history = [[0] * 64 for _ in range(12)]
        self.nodes = 0
        self._start_time = 0.0

    def set_time_control(self, soft_sec: float, hard_sec: float) -> None:
        """Set soft and hard time limits for the next move.
        soft_sec: target time to stop between iterations.
        hard_sec: absolute hard cutoff (abort mid-search).
        """
        self._soft_limit = max(0.05, soft_sec)
        self._hard_limit = max(self._soft_limit, hard_sec)

    # Legacy compat (game_manager may call this)
    def set_move_time_limit(self, seconds: float | None) -> None:
        if seconds is None:
            self._soft_limit = self.TIME_LIMIT
            self._hard_limit = self.TIME_LIMIT * self.HARD_LIMIT_SCALE
        else:
            # game_manager kills the bot process exactly at `seconds` (bot_move_timeout).
            # We must finish BEFORE that wall, so keep both limits well under the budget:
            #   soft (stop starting new depth iterations) = 85% of budget
            #   hard (abort mid-search via _Timeout)      = 92% of budget
            # For a 5 s budget: soft = 4.25 s, hard = 4.6 s  →  safe margin before kill.
            self._soft_limit = max(0.05, seconds * 0.85)
            self._hard_limit = max(self._soft_limit, seconds * 0.92)

    def _piece_idx(self, piece: chess.Piece) -> int:
        return (piece.piece_type - 1) * 2 + (0 if piece.color == chess.WHITE else 1)

    def _soft_time_up(self) -> bool:
        """Check against soft limit: stop between depth iterations."""
        return (time.time() - self._start_time) >= self._soft_limit

    def _hard_time_up(self) -> bool:
        """Check against hard limit: abort immediately mid-search."""
        return (time.time() - self._start_time) >= self._hard_limit

    # Alias used by _negamax and _quiesce via the TIME_CHECK_INTERVAL gate
    def _time_up(self) -> bool:
        return self._hard_time_up()

    # ── Move ordering ────────────────────────────────────────────────────────

    def _mvv_lva(self, board: chess.Board, move: chess.Move) -> int:
        victim = board.piece_at(move.to_square)
        if victim is None and board.is_en_passant(move):
            victim = chess.Piece(chess.PAWN, not board.turn)  # captured pawn
        attacker = board.piece_at(move.from_square)
        if victim is None or attacker is None:
            return 0
        return _SEE_VAL.get(victim.piece_type, 0) * 10 - _SEE_VAL.get(attacker.piece_type, 0)

    def _order_moves(self, board: chess.Board, moves, ply: int, tt_move_uci: str | None,
                     last_own_move: chess.Move | None = None):
        scored = []
        # Detect the reversal square: if last_own_move was A→B, penalise B→A
        reversal_from = last_own_move.to_square if last_own_move else None
        reversal_to   = last_own_move.from_square if last_own_move else None
        for move in moves:
            score = 0
            uci = move.uci()
            if uci == tt_move_uci:
                score = 2_000_000
            elif board.is_capture(move):
                see_val = _see(board, move)
                if see_val >= 0:
                    score = 1_000_000 + self._mvv_lva(board, move)
                else:
                    score = -500_000 + see_val   # losing captures last
            elif move.promotion:
                score = 900_000
            else:
                # Quiet move
                if ply < self.MAX_DEPTH:
                    if move == self.killers[ply][0]:
                        score = 500_000
                    elif move == self.killers[ply][1]:
                        score = 499_000
                    else:
                        piece = board.piece_at(move.from_square)
                        if piece:
                            pidx = self._piece_idx(piece)
                            score = self.history[pidx][move.to_square]
                # Penalise immediately reversing the previous move
                if (reversal_from is not None
                        and move.from_square == reversal_from
                        and move.to_square == reversal_to):
                    score -= 300_000
            scored.append((score, move))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [m for _, m in scored]

    # ── Quiescence Search ────────────────────────────────────────────────────

    def _quiesce(self, board: chess.Board, alpha: int, beta: int, depth: int = 0) -> int:
        self.nodes += 1
        if (self.nodes & (self.TIME_CHECK_INTERVAL - 1)) == 0 and self._time_up():
            raise _Timeout()

        stand_pat = _evaluate(board)

        if stand_pat >= beta:
            return beta
        if stand_pat > alpha:
            alpha = stand_pat

        # Delta pruning: if even capturing a queen can't improve alpha, skip
        DELTA = 1000
        if stand_pat + DELTA < alpha:
            return alpha

        # Captures (including capture-promotions) and quiet queen promotions
        captures = [m for m in board.legal_moves if board.is_capture(m)]
        quiet_promos = [m for m in board.legal_moves if m.promotion == chess.QUEEN and not board.is_capture(m)]
        captures.sort(key=lambda m: self._mvv_lva(board, m), reverse=True)

        for move in captures:
            if _see(board, move) < -50:
                continue
            board.push(move)
            score = -self._quiesce(board, -beta, -alpha, depth + 1)
            board.pop()
            if score >= beta:
                return beta
            if score > alpha:
                alpha = score

        for move in quiet_promos:
            board.push(move)
            score = -self._quiesce(board, -beta, -alpha, depth + 1)
            board.pop()
            if score >= beta:
                return beta
            if score > alpha:
                alpha = score

        return alpha

    # ── Negamax PVS ─────────────────────────────────────────────────────────

    def _negamax(self, board: chess.Board, depth: int, alpha: int, beta: int, ply: int,
                 can_null: bool = True, pos_stack: list | None = None) -> int:
        self.nodes += 1
        if (self.nodes & (self.TIME_CHECK_INTERVAL - 1)) == 0 and self._time_up():
            raise _Timeout()

        alpha_orig = alpha

        # ── Repetition detection ─────────────────────────────────────────────
        h = _zobrist_hash(board)
        if pos_stack is not None and ply > 0:
            count_in_stack   = pos_stack.count(h)
            count_in_history = self.game_history.count(h)
            # Standard threefold repetition: current position would be at least the
            # third occurrence if it already appeared twice before.
            if count_in_stack + count_in_history >= 2:
                # Use side-to-move evaluation to decide whether to welcome the draw.
                static = _evaluate(board)
                # When badly behind, slightly prefer the draw; otherwise apply contempt.
                if static < -150:
                    return 10
                if static > 50:
                    return self.CONTEMPT
                return 0

        # ── TT lookup ───────────────────────────────────────────────────────
        tt_move_uci = None
        if h in self.tt:
            tt_depth, tt_score, tt_flag, tt_mv = self.tt[h]
            tt_move_uci = tt_mv
            if tt_depth >= depth:
                # Convert stored mate scores back to the current ply-relative form.
                if tt_score > MATE_THRESHOLD:
                    tt_score = tt_score - ply
                elif tt_score < -MATE_THRESHOLD:
                    tt_score = tt_score + ply
                if tt_flag == TT_EXACT:
                    return tt_score
                elif tt_flag == TT_LOWER:
                    alpha = max(alpha, tt_score)
                elif tt_flag == TT_UPPER:
                    beta = min(beta, tt_score)
                if alpha >= beta:
                    return tt_score

        if board.is_game_over():
            if board.is_checkmate():
                return -MATE_SCORE + ply   # prefer shorter mates
            return 0

        if depth <= 0:
            return self._quiesce(board, alpha, beta)

        in_check = board.is_check()

        # Reverse Futility Pruning (Static Null Move Pruning)
        static_eval = None
        if not in_check and abs(beta) < MATE_THRESHOLD:
            static_eval = _evaluate(board)
            if depth <= 5:
                rfp_margin = 120 * depth
                if static_eval - rfp_margin >= beta:
                    return static_eval

        # Null-move pruning
        R = 2 + depth // 4
        if (can_null and not in_check and depth >= 3
                and len(board.piece_map()) > 4):
            board.push(chess.Move.null())
            null_score = -self._negamax(board, depth - 1 - R, -beta, -beta + 1, ply + 1,
                                        can_null=False)
            board.pop()
            if null_score >= beta:
                # Verification search to avoid incorrect cutoffs in zugzwang-like positions.
                if depth - 1 - R > 0:
                    verify = -self._negamax(board, depth - 1 - R, -beta, -beta + 1, ply + 1,
                                            can_null=False, pos_stack=pos_stack)
                    if verify >= beta:
                        return verify
                else:
                    return null_score

        # Get + order moves
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            return 0   # stalemate (checkmate already caught above)
        # At the root, pass last_own_move to penalise immediate reversals
        root_last_move = self.last_own_move if ply == 0 else None
        ordered = self._order_moves(board, legal_moves, ply, tt_move_uci,
                                    last_own_move=root_last_move)

        # Futility Pruning flag
        futility_pruning = False
        if not in_check and depth <= 3 and static_eval is not None and abs(alpha) < MATE_THRESHOLD:
            futility_margin = 150 + 150 * depth
            if static_eval + futility_margin <= alpha:
                futility_pruning = True

        best_score = -INF
        best_move = None

        for i, move in enumerate(ordered):
            is_capture = board.is_capture(move)
            piece_for_history = board.piece_at(move.from_square)
            # gives_check for futility: python-chess documents gives_check(move) as "would this move give check"
            if futility_pruning and i > 0 and not is_capture and not move.promotion and not board.gives_check(move):
                continue

            # Late Move Pruning (LMP): prune very late quiet moves at shallow depth.
            if (depth <= 3 and i >= 6 and not in_check and not is_capture
                    and not move.promotion and not board.gives_check(move)):
                continue

            # SEE-based pruning of clearly losing captures at shallow depth.
            if is_capture and depth <= 4 and not board.gives_check(move):
                if _see(board, move) < 0:
                    continue

            next_stack = (pos_stack or []) + [h]  # push current position before the move
            board.push(move)
            gives_check = board.is_check()  # use actual state after push for LMR and extension

            # Search depth: extend by 1 when move gives check
            search_depth = depth - 1 + (1 if gives_check else 0)
            search_depth = min(search_depth, depth)  # don't exceed current depth

            # LMR: reduce late quiet moves (no capture, no promotion, no check)
            reduction = 0
            if (i >= 3 and depth >= 3 and not is_capture
                    and not move.promotion and not in_check and not gives_check):
                reduction = max(1, int(math.log(depth) * math.log(i + 1) / 2))
                reduction = min(reduction, depth - 2)

            # PVS
            if i == 0:
                score = -self._negamax(board, search_depth, -beta, -alpha, ply + 1,
                                       pos_stack=next_stack)
            else:
                score = -self._negamax(board, search_depth - reduction, -alpha - 1, -alpha, ply + 1,
                                       pos_stack=next_stack)
                if score > alpha and (score < beta or reduction > 0):
                    score = -self._negamax(board, search_depth, -beta, -alpha, ply + 1,
                                           pos_stack=next_stack)

            board.pop()

            if score > best_score:
                best_score = score
                best_move = move

            if score > alpha:
                alpha = score

            if alpha >= beta:
                # Beta cutoff – update killers + history (use piece saved before push)
                if not is_capture:
                    if ply < self.MAX_DEPTH:
                        self.killers[ply][1] = self.killers[ply][0]
                        self.killers[ply][0] = move
                    if piece_for_history:
                        pidx = self._piece_idx(piece_for_history)
                        self.history[pidx][move.to_square] += depth * depth
                break

        # Store in TT
        if best_score <= alpha_orig:
            flag = TT_UPPER
        elif best_score >= beta:
            flag = TT_LOWER
        else:
            flag = TT_EXACT

        # Encode mate scores independent of ply so they are comparable across depths.
        store_score = best_score
        if store_score > MATE_THRESHOLD:
            store_score = store_score + ply
        elif store_score < -MATE_THRESHOLD:
            store_score = store_score - ply

        # Simple replacement: prefer deeper entries
        if h not in self.tt or self.tt[h][0] <= depth:
            self.tt[h] = (depth, store_score, flag, best_move.uci() if best_move else None)

        # Cap TT size
        if len(self.tt) > self.TT_MAX_SIZE:
            # Remove ~10% entries using dict insertion order (approximate LRU) without
            # building an intermediate list.
            to_remove = max(1, self.TT_MAX_SIZE // 10)
            it = iter(self.tt)
            for _ in range(to_remove):
                try:
                    k = next(it)
                except StopIteration:
                    break
                self.tt.pop(k, None)

        return best_score

    # ── Root search with iterative deepening + aspiration windows ────────────

    def _search(self, board: chess.Board) -> chess.Move | None:
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            return None
        if len(legal_moves) == 1:
            return legal_moves[0]

        self._start_time = time.time()
        best_move = legal_moves[0]
        prev_score = 0
        best_move_stability = 0   # consecutive depths with the same best move
        prev_best_uci = None

        for depth in range(1, self.MAX_DEPTH):
            # Soft limit: don't START a new depth if we're already over budget
            if depth > 1 and self._soft_time_up():
                break

            # Aspiration window: depth-dependent to reduce re-searches.
            if depth >= 4:
                window = 16 + depth * depth
                alpha = prev_score - window
                beta  = prev_score + window
            else:
                window = None
                alpha = -INF
                beta  = INF

            try:
                while True:
                    try:
                        score = self._negamax(board, depth, alpha, beta, 0,
                                              pos_stack=list(self.game_history))
                    except _Timeout:
                        return best_move

                    if score <= alpha:
                        # Fail-low: widen window and re-search.
                        if window is not None:
                            window *= 2
                            alpha = prev_score - window
                            beta  = prev_score + window
                        else:
                            alpha = -INF
                            beta  = INF
                    elif score >= beta:
                        # Fail-high: widen window and re-search.
                        if window is not None:
                            window *= 2
                            alpha = prev_score - window
                            beta  = prev_score + window
                        else:
                            alpha = -INF
                            beta  = INF
                    else:
                        break   # score inside window

                # Extract best move from TT at root
                h = _zobrist_hash(board)
                if h in self.tt and self.tt[h][3]:
                    try:
                        candidate = chess.Move.from_uci(self.tt[h][3])
                        if candidate in legal_moves:
                            best_move = candidate
                    except Exception:
                        pass

                # Track best-move stability
                cur_uci = best_move.uci() if best_move else None
                if cur_uci == prev_best_uci:
                    best_move_stability += 1
                else:
                    best_move_stability = 0
                prev_best_uci = cur_uci

                prev_score = score
                elapsed = time.time() - self._start_time
                print(f"  depth={depth} score={score} move={best_move} "
                      f"nodes={self.nodes} time={elapsed:.2f}s", flush=True)

                # Early exit: same best move 3+ consecutive depths and >60% soft budget used
                if best_move_stability >= 3 and elapsed >= self._soft_limit * 0.60:
                    break

            except _Timeout:
                break

        return best_move

    # ── Public interface ─────────────────────────────────────────────────────

    def make_move(self, board: chess.Board) -> chess.Move | None:
        # Build game_history: Zobrist hashes for every position seen BEFORE
        # the current board state. Used by _negamax for repetition detection.
        # NOTE: do NOT include the current position — that would cause false
        #       "already repeated" hits at the root of every search.
        scratch = chess.Board()
        self.game_history = []
        for mv in board.move_stack:          # oldest-first, same as board's internal stack
            self.game_history.append(_zobrist_hash(scratch))
            scratch.push(mv)
        # (do not append _zobrist_hash(board) — that is the position we are searching)

        # Reset per-move state (keep TT across moves for same game)
        self.nodes = 0
        self.killers = [[None, None] for _ in range(self.MAX_DEPTH)]
        # Decay history (avoid stale values dominating, but keep useful info longer)
        self.history = [[(v * 3) // 4 for v in row] for row in self.history]

        legal_moves = list(board.legal_moves)
        if not legal_moves:
            return None

        # Opening book lookup (only returns if move is legal)
        key = _fen_key(board)
        if key in OPENING_BOOK:
            book_moves = OPENING_BOOK[key]
            random.shuffle(book_moves)
            for uci in book_moves:
                try:
                    m = chess.Move.from_uci(uci)
                    if m in legal_moves:
                        print(f"  [Book] {uci}", flush=True)
                        return m
                except Exception:
                    continue

        move = self._search(board)
        # Defensive: never return an illegal move (covers TT hash collision or any bug)
        if move is not None and move not in board.legal_moves:
            move = legal_moves[0]
        self.last_own_move = move   # remember for reversal penalty next turn
        return move


class _Timeout(Exception):
    """Internal signal to abort the search when time is up."""
    pass
