import chess
import time
from chess_player import ChessPlayer

# Piece values (Standard)
PIECE_VALUES = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    # King value is high to ensure it's not traded
    chess.KING: 20000 
}

# Simplified PeSTO's Evaluation Weights (Piece-Square Tables)
# Encourages pieces to develop to active squares and pawns to control the center.
MG_PAWN_TABLE = [
     0,  0,  0,  0,  0,  0,  0,  0,
    50, 50, 50, 50, 50, 50, 50, 50,
    10, 10, 20, 30, 30, 20, 10, 10,
     5,  5, 10, 25, 25, 10,  5,  5,
     0,  0,  0, 20, 20,  0,  0,  0,
     5, -5,-10,  0,  0,-10, -5,  5,
     5, 10, 10,-20,-20, 10, 10,  5,
     0,  0,  0,  0,  0,  0,  0,  0
]

MG_KNIGHT_TABLE = [
    -50,-40,-30,-30,-30,-30,-40,-50,
    -40,-20,  0,  0,  0,  0,-20,-40,
    -30,  0, 10, 15, 15, 10,  0,-30,
    -30,  5, 15, 20, 20, 15,  5,-30,
    -30,  0, 15, 20, 20, 15,  0,-30,
    -30,  5, 10, 15, 15, 10,  5,-30,
    -40,-20,  0,  5,  5,  0,-20,-40,
    -50,-40,-30,-30,-30,-30,-40,-50
]

MG_BISHOP_TABLE = [
    -20,-10,-10,-10,-10,-10,-10,-20,
    -10,  0,  0,  0,  0,  0,  0,-10,
    -10,  0,  5, 10, 10,  5,  0,-10,
    -10,  5,  5, 10, 10,  5,  5,-10,
    -10,  0, 10, 10, 10, 10,  0,-10,
    -10, 10, 10, 10, 10, 10, 10,-10,
    -10,  5,  0,  0,  0,  0,  5,-10,
    -20,-10,-10,-10,-10,-10,-10,-20
]

MG_ROOK_TABLE = [
      0,  0,  0,  0,  0,  0,  0,  0,
      5, 10, 10, 10, 10, 10, 10,  5,
     -5,  0,  0,  0,  0,  0,  0, -5,
     -5,  0,  0,  0,  0,  0,  0, -5,
     -5,  0,  0,  0,  0,  0,  0, -5,
     -5,  0,  0,  0,  0,  0,  0, -5,
     -5,  0,  0,  0,  0,  0,  0, -5,
      0,  0,  0,  5,  5,  0,  0,  0
]

MG_QUEEN_TABLE = [
    -20,-10,-10, -5, -5,-10,-10,-20,
    -10,  0,  0,  0,  0,  0,  0,-10,
    -10,  0,  5,  5,  5,  5,  0,-10,
     -5,  0,  5,  5,  5,  5,  0, -5,
      0,  0,  5,  5,  5,  5,  0, -5,
    -10,  5,  5,  5,  5,  5,  0,-10,
    -10,  0,  5,  0,  0,  0,  0,-10,
    -20,-10,-10, -5, -5,-10,-10,-20
]

MG_KING_TABLE = [
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -20,-30,-30,-40,-40,-30,-30,-20,
    -10,-20,-20,-20,-20,-20,-20,-10,
     20, 20,  0,  0,  0,  0, 20, 20,
     20, 30, 10,  0,  0, 10, 30, 20
]

PST = {
    chess.PAWN: MG_PAWN_TABLE,
    chess.KNIGHT: MG_KNIGHT_TABLE,
    chess.BISHOP: MG_BISHOP_TABLE,
    chess.ROOK: MG_ROOK_TABLE,
    chess.QUEEN: MG_QUEEN_TABLE,
    chess.KING: MG_KING_TABLE
}

class ChessChamp(ChessPlayer):
    """
    Supercharged Negamax Engine featuring:
    - Alpha-Beta Pruning
    - Quiescence Search (Solves the Horizon Effect)
    - Iterative Deepening (Dynamic Time Management)
    - Move Ordering (MVV-LVA for faster pruning)
    - Transposition Table caching (Memory for positions)
    """
    
    def __init__(self, name):
        super().__init__(name)
        # Using a dictionary to cache board evaluations (Transposition Table)
        self.tt = {}
        # Max search duration per turn in seconds (Safe margin below 5.0 to avoid timeouts)
        self.time_limit = 4.8 
        self.start_time = 0
        
    def evaluate_board(self, board):
        """Static Evaluation: Material Counting + Piece-Square Tables"""
        if board.is_checkmate():
            # If current turn is checkmated, it's terrible for them
            return -99999
            
        if board.is_stalemate() or board.is_insufficient_material() or board.is_repetition():
            return 0
            
        evaluation = 0
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                value = PIECE_VALUES[piece.piece_type]
                
                # Positional value from Piece-Square Table
                table = PST[piece.piece_type]
                # Flip table for black pieces
                pst_val = table[chess.square_mirror(square)] if piece.color == chess.BLACK else table[square]
                
                if piece.color == chess.WHITE:
                    evaluation += (value + pst_val)
                else:
                    evaluation -= (value + pst_val)
                    
        # Negamax always expects score relative to the side whose turn it is
        return evaluation if board.turn == chess.WHITE else -evaluation

    def score_move(self, board, move):
        """
        Move Ordering Heuristic - Crucial for fast Alpha-Beta Pruning.
        Orders captures based on MVV-LVA (Most Valuable Victim - Least Valuable Attacker)
        """
        score = 0
        if board.is_capture(move):
            victim = board.piece_at(move.to_square)
            attacker = board.piece_at(move.from_square)
            if victim and attacker:
                # E.g. Pawn taking Queen gets huge score: 900 - 100/100 = 899
                score = 10 * PIECE_VALUES[victim.piece_type] - PIECE_VALUES[attacker.piece_type]
        if move.promotion:
            score += PIECE_VALUES[move.promotion]
        return score

    def quiescence_search(self, board, alpha, beta):
        """
        Quiescence Search: Only looks at captures.
        If a sequence of captures is happening at depth zero, keep searching
        until the board "quiets down" to prevent walking into traps.
        """
        if time.time() - self.start_time > self.time_limit:
            raise TimeoutError()
            
        stand_pat = self.evaluate_board(board)
        if stand_pat >= beta:
            return beta
        if alpha < stand_pat:
            alpha = stand_pat
            
        # Only look at captures or promotions
        captures = [m for m in board.legal_moves if board.is_capture(m) or m.promotion]
        captures.sort(key=lambda m: self.score_move(board, m), reverse=True)
        
        for move in captures:
            board.push(move)
            score = -self.quiescence_search(board, -beta, -alpha)
            board.pop()
            
            if score >= beta:
                return beta
            if score > alpha:
                alpha = score
                
        return alpha

    def negamax(self, board, depth, alpha, beta, ply=0):
        """
        Main recursive Minimax algorithm (Negamax variant).
        Uses Alpha-Beta Pruning to skip bad branches.
        """
        if time.time() - self.start_time > self.time_limit:
            raise TimeoutError()
            
        alpha_orig = alpha
        
        # 1. Transposition Table Lookup
        board_fen = board.fen()
        tt_entry = self.tt.get(board_fen)
        
        if tt_entry is not None and tt_entry['depth'] >= depth:
            if tt_entry['flag'] == 'EXACT':
                return tt_entry['value']
            elif tt_entry['flag'] == 'LOWERBOUND':
                alpha = max(alpha, tt_entry['value'])
            elif tt_entry['flag'] == 'UPPERBOUND':
                beta = min(beta, tt_entry['value'])
                
            if alpha >= beta:
                return tt_entry['value']
                
        # 2. Base case: Evaluate the quiet position
        if depth == 0 or board.is_game_over():
            return self.quiescence_search(board, alpha, beta)
            
        # 3. Get and order moves
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            return 0 # Draw/Stalemate
            
        legal_moves.sort(key=lambda m: self.score_move(board, m), reverse=True)
        
        best_value = -float('inf')
        
        # 4. Search branches
        for move in legal_moves:
            board.push(move)
            board_val = -self.negamax(board, depth - 1, -beta, -alpha, ply + 1)
            board.pop()
            
            best_value = max(best_value, board_val)
            alpha = max(alpha, board_val)
            
            if alpha >= beta:
                break # Alpha-Beta Pruning (Snip!)
                
        # 5. Store result in Transposition Table
        tt_flag = 'EXACT'
        if best_value <= alpha_orig:
            tt_flag = 'UPPERBOUND'
        elif best_value >= beta:
            tt_flag = 'LOWERBOUND'
            
        self.tt[board_fen] = {
            'value': best_value,
            'depth': depth,
            'flag': tt_flag
        }
        
        return best_value

    def make_move(self, board):
        """
        Iterative Deepening Search.
        Instead of getting stuck forever searching depth 5, we search depth 1, then 2, then 3...
        until our timer runs out. This guarantees we use our time perfectly.
        """
        self.start_time = time.time()
        best_move_overall = None
        
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            return None
            
        # Only 1 legal move? Don't waste time thinking!
        if len(legal_moves) == 1:
            return legal_moves[0]
            
        try:
            # Iterative deepening loop
            for current_depth in range(1, 100): # Practically infinite depth
                best_value = -float('inf')
                alpha = -float('inf')
                beta = float('inf')
                best_move_this_depth = None
                
                # Order moves to search the best lines first
                legal_moves.sort(key=lambda m: self.score_move(board, m), reverse=True)
                
                for move in legal_moves:
                    board.push(move)
                    board_value = -self.negamax(board, current_depth - 1, -beta, -alpha, 1)
                    board.pop()
                    
                    if board_value > best_value:
                        best_value = board_value
                        best_move_this_depth = move
                        
                    alpha = max(alpha, board_value)
                    
                best_move_overall = best_move_this_depth
                
                # Optional debugging: Console logging thinking process
                # print(f"Depth {current_depth} complete. Move: {best_move_overall}, Eval: {best_value}")
                
        except TimeoutError:
            # Time is up! We return the best move found from the LAST fully completed depth.
            pass 
            
        # Fallback if somehow we found no move
        if best_move_overall is None:
            import random
            best_move_overall = random.choice(legal_moves)
            
        return best_move_overall
