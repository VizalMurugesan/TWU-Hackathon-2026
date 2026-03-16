# bot_template.py
# Template for participants to create their own chess bot
import time

import chess
from chess_player import ChessPlayer
import random


class Dabot(ChessPlayer):
    """
    Create your own chess bot by inheriting from ChessPlayer!

    To use your bot:
    1. Copy this file and rename it (e.g., my_awesome_bot.py)
    2. Replace 'YourCustomBot' with your bot's name
    3. Implement the make_move() method with your logic
    4. Import and use it in main.py or other files

    Example in main.py:
        from my_awesome_bot import MyAwesomeBot
        player1 = SecureBotWrapper(MyAwesomeBot, "My Bot Name")
    """

    def __init__(self, name):
        super().__init__(name)
        self.piece_values = {1: 100, 2: 320, 3: 330, 4: 500, 5: 900, 6: 0}
        self.depth = 3

    # evaluates the score of the piece
    def evaluate_material(self, board):
        score = 0
        our_color = not board.turn
        for piece_type, value in self.piece_values.items():
            score += len(board.pieces(piece_type, our_color)) * value
            score -= len(board.pieces(piece_type, board.turn)) * value
        return score

    def evaluate_compensation(self, board):
        score = 0
        our_color = not board.turn

        # Exposed enemy king
        opponent_king_sq = board.king(board.turn)
        king_file = chess.square_file(opponent_king_sq)
        pawns_on_file = [
            chess.square(king_file, r) for r in range(8)
            if board.piece_type_at(chess.square(king_file, r)) == chess.PAWN
        ]
        if not pawns_on_file:
            score += 200

        # Our piece activity
        for sq in chess.SQUARES:
            piece = board.piece_at(sq)
            if piece and piece.color == our_color:
                score += len(board.attacks(sq)) * 5

        # Available checks
        checks = [m for m in board.legal_moves if board.gives_check(m)]
        score += len(checks) * 30

        return score

    def evaluate_stealth(self, board):
        score = 0
        our_color = not board.turn

        # Penalize tactical sharpness — minimax handles that,
        # we want to build quiet threats on top
        captures_available = sum(1 for m in board.legal_moves if board.is_capture(m))
        score -= captures_available * 10

        # Reward advanced safe pawns
        for sq in chess.SQUARES:
            piece = board.piece_at(sq)
            if piece and piece.color == our_color and piece.piece_type == chess.PAWN:
                rank = chess.square_rank(sq)
                if not board.is_attacked_by(board.turn, sq):
                    score += rank * 15 if our_color == chess.WHITE else (7 - rank) * 15

        # Reward piece coordination
        for sq in chess.SQUARES:
            piece = board.piece_at(sq)
            if piece and piece.color == our_color:
                defenders = sum(
                    1 for s in chess.SQUARES
                    if board.piece_at(s)
                    and board.piece_at(s).color == our_color
                    and sq in board.attacks(s)
                )
                score += defenders * 20

        return score

    def evaluate_danger(self, board):
        danger = 0
        our_color = not board.turn

        # Our king exposure
        our_king_sq = board.king(our_color)
        king_file = chess.square_file(our_king_sq)
        pawns_on_file = [
            chess.square(king_file, r) for r in range(8)
            if board.piece_type_at(chess.square(king_file, r)) == chess.PAWN
        ]
        if not pawns_on_file:
            danger += 200

        # Our pieces under attack
        for sq in chess.SQUARES:
            piece = board.piece_at(sq)
            if piece and piece.color == our_color:
                if board.is_attacked_by(board.turn, sq):
                    danger += self.piece_values.get(piece.piece_type, 0)

        # Opponent's available checks
        opponent_checks = sum(
            1 for m in board.legal_moves if board.gives_check(m)
        )
        danger += opponent_checks * 40

        return danger

    def evaluate(self, board):
        if board.is_checkmate():
            return float('-inf')
        if board.is_stalemate() or board.is_insufficient_material():
            return 0

        our_color = not board.turn
        score = 0

        # Material — fast, no square looping
        for piece_type, value in self.piece_values.items():
            score += len(board.pieces(piece_type, our_color)) * value
            score -= len(board.pieces(piece_type, board.turn)) * value

        # King safety — just check if king is on open file
        our_king_sq = board.king(our_color)
        opp_king_sq = board.king(board.turn)

        our_king_file = chess.square_file(our_king_sq)
        opp_king_file = chess.square_file(opp_king_sq)

        # Penalize our exposed king
        if not any(
                board.piece_type_at(chess.square(our_king_file, r)) == chess.PAWN
                for r in range(8)
        ):
            score -= 150

        # Reward opponent's exposed king
        if not any(
                board.piece_type_at(chess.square(opp_king_file, r)) == chess.PAWN
                for r in range(8)
        ):
            score += 150

        # Mobility — more moves = better, cheap to compute
        score += len(list(board.legal_moves)) * 5

        return score

    def find_sacrifice(self, board):
        """
        Runs before minimax — if a strong sacrifice is available,
        return it immediately rather than letting minimax evaluate it
        as a simple material loss.
        """
        our_color = board.turn
        print(type(board))
        sacrifice_candidates = []

        for move in board.legal_moves:
            board.push(move)

            if board.is_attacked_by(board.turn, move.to_square):
                for reply in board.legal_moves:
                    if reply.to_square == move.to_square:
                        board.push(reply)

                        compensation = self.evaluate_compensation(board)
                        piece_given_up = self.piece_values.get(
                            board.piece_type_at(move.to_square), 0
                        )

                        if compensation > piece_given_up + 150:
                            sacrifice_candidates.append((move, compensation))

                        board.pop()
                        break

            board.pop()

        if not sacrifice_candidates:
            return None

        sacrifice_candidates.sort(key=lambda x: x[1], reverse=True)
        return sacrifice_candidates[0][0]

    def minimax(self, board, depth, alpha, beta, maximizing):
        """
        Search ahead `depth` moves using the combined evaluator.
        Alpha-beta pruning cuts branches that can't affect the result,
        allowing deeper search in the same time.
        """
        # Base case — score the position if we've hit the depth limit
        if depth == 0 or board.is_game_over():
            return self.evaluate(board)

        if time.time() - self.start_time > self.time_limit:
            return self.evaluate(board)

        if maximizing:
            best = float('-inf')
            for move in self.order_moves(board):  # ← ordered for better pruning
                board.push(move)
                score = self.minimax(board, depth - 1, alpha, beta, False)
                board.pop()

                best = max(best, score)
                alpha = max(alpha, best)
                if alpha >= beta:
                    break  # Prune — opponent won't allow this branch
            return best

        else:
            best = float('inf')
            for move in self.order_moves(board):
                board.push(move)
                score = self.minimax(board, depth - 1, alpha, beta, True)
                board.pop()

                best = min(best, score)
                beta = min(beta, best)
                if beta <= alpha:
                    break  # Prune — we won't allow this branch
            return best

        # ─────────────────────────────────────────
        # MOVE ORDERING
        # ─────────────────────────────────────────

    def order_moves(self, board):
        """
        Sort moves so alpha-beta pruning is more effective.
        Better moves searched first = more branches pruned.
        Order: checkmate → checks → captures (by value) → rest
        """

        def move_priority(move):
            score = 0
            # Reward captures by value of captured piece
            if board.is_capture(move):
                victim = board.piece_type_at(move.to_square)
                attacker = board.piece_type_at(move.from_square)
                if victim:
                    score += self.piece_values.get(victim, 0)
                # Prefer capturing with lower value pieces (MVV-LVA)
                if attacker:
                    score -= self.piece_values.get(attacker, 0) * 0.1
            # Reward promotions
            if move.promotion:
                score += 900
            return score

        # def move_priority(move):
        #     board.push(move)
        #
        #     if board.is_checkmate():
        #         score = 10000  # search checkmate first
        #     elif board.is_check():
        #         score = 5000
        #     else:
        #         score = 0
        #
        #     board.pop()
        #
        #     # On top of check bonus, add capture value
        #     if board.is_capture(move):
        #         captured_value = self.piece_values.get(
        #             board.piece_type_at(move.to_square), 0
        #         )
        #         score += captured_value
        #
        #     return score

        return sorted(board.legal_moves, key=move_priority, reverse=True)

    def make_move(self, board):
        """
        Args:
            board: chess.Board object representing the current game state

        Returns:
            chess.Move: Your chosen move, or None if no moves available

        Tips:
            - board.legal_moves gives you all valid moves
            - board.is_capture(move) checks if a move captures a piece
            - board.piece_at(square) gets the piece at a square (or None)
            - board.fen() returns the position in FEN notation
            - board.turn returns chess.WHITE or chess.BLACK
        """

        # Get all legal moves
        print(type(board))
        legal_moves = list(board.legal_moves)

        if not legal_moves:
            return None
# time limit because the algorithm can take so long
        self.start_time = time.time()
        self.time_limit = 3.0
# make sure that it finds sacrifices after 10 moves
        if board.fullmove_number > 10:
            sacrifice = self.find_sacrifice(board)
            if sacrifice:
                return sacrifice

        # ── Stage 2: Minimax search ──────────────────────────────
        # Run full search using the blended evaluator
        best_move = legal_moves[0]
        best_score = float('-inf')
        alpha = float('-inf')
        beta = float('inf')

        for move in self.order_moves(board):
            if time.time() - self.start_time > self.time_limit:
                print("Time limit hit, returning best so far")
                break
            board.push(move)
            score = self.minimax(board, self.depth - 1, alpha, beta, False)
            board.pop()

            if score > best_score:
                best_score = score
                best_move = move

            alpha = max(alpha, best_score)

        return best_move

        # piece_values = {1: 100, 2: 320, 3: 330, 4: 500, 5: 900, 6: 0}

        # safemoves = []
        # captures = [m for m in legal_moves if board.is_capture(m)]
        # fin = []
        #
        # # IMPLEMENT YOUR LOGIC HERE!
        # # This prioritizes moves, it prioritizes safe captures, then safe moves, then any legal moves
        #
        # for i in legal_moves:
        #     board.push(i)
        #     safe = not board.is_attacked_by(not board.turn, i.to_square)
        #     board.pop()
        #
        #     if safe:
        #         safemoves.append(i)
        #
        # if safemoves:
        #     safe_captures = [m for m in safemoves if m in captures]
        #     if safe_captures:
        #         safe_captures.sort(
        #             key=lambda m: piece_values.get(board.piece_type_at(m.to_square), 0),
        #             reverse=True
        #         )
        #         return safe_captures[0] #take the most valuable piece
        #     else:
        #         fin.extend(safemoves)
        #         return random.choice(fin)
        # else:
        #     return random.choice(legal_moves)

    # return random.choice(legal_moves)


# ===== EXAMPLE IMPLEMENTATIONS =====

class DefensiveBot(ChessPlayer):
    """Prefers to move pieces to safe squares and avoid captures"""

    def make_move(self, board):
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            return None

        import random

        # Try to find a move that doesn't leave our piece hanging
        safe_moves = []
        for move in legal_moves:
            board.push(move)
            # Check if our piece is under attack after this move
            is_safe = not board.is_attacked_by(not board.turn, move.to_square)

            board.pop()

            if is_safe:
                safe_moves.append(move)

        if safe_moves:
            return random.choice(safe_moves)
        else:
            return random.choice(legal_moves)


class AggressiveBot(ChessPlayer):
    """Prioritizes capturing opponent pieces"""

    def make_move(self, board):
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            return None

        import random

        # Separate captures and quiet moves
        captures = [m for m in legal_moves if board.is_capture(m)]

        # Prefer captures, but make regular moves if none available
        if captures:
            return random.choice(captures)
        else:
            return random.choice(legal_moves)
