import chess
import time
import settings
from game_recorder import GameRecord

class GameManager:
    def __init__(self, white_player, black_player):
        self.board = chess.Board()
        self.white_player = white_player
        self.black_player = black_player
        self.white_player.initialize(chess.WHITE)
        self.black_player.initialize(chess.BLACK)

        # Start secure processes if they are bots
        if hasattr(self.white_player, 'start_process'): self.white_player.start_process()
        if hasattr(self.black_player, 'start_process'): self.black_player.start_process()

        self.is_paused = True 
        
        # Timing
        self.last_real_time = time.time()
        self.white_time = settings.GAME_TIME_LIMIT
        self.black_time = settings.GAME_TIME_LIMIT
        
        # Async State
        self.pending_move_bot = None  # The bot currently thinking
        self.pending_move_start_time = 0
        self.bot_move_timeout = 300.0   # HARD LIMIT: Kill bot if it takes > 5s per move
        
        self.game_over_reason = ""
        self.time_history = []
        
        # Game Recording
        self.game_record = GameRecord(white_player.name, black_player.name) 

    def update(self):
        current_time = time.time()
        dt = current_time - self.last_real_time
        self.last_real_time = current_time

        if self.is_paused or self.board.is_game_over() or self.game_over_reason:
            return

        # 1. Manage Clocks
        if self.board.turn == chess.WHITE:
            self.white_time -= dt
            if self.white_time <= 0: self._timeout(chess.BLACK)
        else:
            self.black_time -= dt
            if self.black_time <= 0: self._timeout(chess.WHITE)

        # 2. Get Current Player
        current_player = self.white_player if self.board.turn == chess.WHITE else self.black_player

        # --- ASYNC BOT LOGIC ---
        
        # Scenario A: We are already waiting for a bot
        if self.pending_move_bot:
            # Check for timeout (Anti-Hogging Security)
            time_spent = current_time - self.pending_move_start_time
            if time_spent > self.bot_move_timeout:
                bot = self.pending_move_bot
                self.game_over_reason = f"Forfeit: {bot.name} timed out"
                bot.kill()
                return

            # Poll for result
            try:
                move = self.pending_move_bot.check_result() # Non-blocking check
                if move:
                    # Move received!
                    if move in self.board.legal_moves:
                        self._apply_move(move)
                        self.pending_move_bot = None
                    else:
                        self.game_over_reason = f"Illegal Move: {move}"
            except Exception as e:
                # Bot Crashed (1/0, etc)
                print(e)
                self.game_over_reason = f"Forfeit: {current_player.name} Crashed"
                self.pending_move_bot.kill()

        # Scenario B: No one is thinking, we need to start a turn
        else:
            if current_player.is_human:
                return # Wait for UI click
            
            # Start the Bot Process
            # Only start if we have enough time left on the clock
            if hasattr(current_player, 'request_move'):
                current_player.request_move(self.board.copy())
                self.pending_move_bot = current_player
                self.pending_move_start_time = current_time
            else:
                # Fallback for old synchronous bots (not recommended)
                move = current_player.make_move(self.board.copy())
                if move: self._apply_move(move)

    def _timeout(self, winner_color):
        winner = "White" if winner_color == chess.WHITE else "Black"
        self.white_time = max(0, self.white_time)
        self.black_time = max(0, self.black_time)
        self.game_over_reason = f"{winner} Wins (Timeout)"

    def _apply_move(self, move):
        self.time_history.append((self.white_time, self.black_time))
        self.board.push(move)
        if self.board.turn == chess.BLACK: 
            self.white_time += settings.INCREMENT_SEC
        else: 
            self.black_time += settings.INCREMENT_SEC
        
        # Record the move in the game record
        self.game_record.add_move(move.uci(), self.white_time, self.black_time)

    def try_manual_move(self, start_sq, end_sq):
        # Human moves bypass the queue system
        if self.pending_move_bot: return False # Can't move while bot thinks
        
        move = chess.Move(start_sq, end_sq)
        piece = self.board.piece_at(start_sq)
        if piece and piece.piece_type == chess.PAWN:
            rank_to = chess.square_rank(end_sq)
            if (piece.color == chess.WHITE and rank_to == 7) or (piece.color == chess.BLACK and rank_to == 0):
                 move = chess.Move(start_sq, end_sq, promotion=chess.QUEEN)

        if move in self.board.legal_moves:
            self._apply_move(move)
            return True
        return False

    def undo_move(self):
        # Kill any thinking bots before undoing state
        if self.pending_move_bot:
            self.pending_move_bot.kill()
            self.pending_move_bot.start_process() # Restart fresh
            self.pending_move_bot = None

        if len(self.board.move_stack) > 0:
            self.board.pop()
            if self.time_history:
                t = self.time_history.pop()
                self.white_time, self.black_time = t
            self.is_paused = True
            self.game_over_reason = ""
            
    def reset_game(self):
        # Stop any pending bot operations
        if self.pending_move_bot:
            self.pending_move_bot.kill()
            self.pending_move_bot = None
        
        # Kill and cleanly restart both players
        for player in [self.white_player, self.black_player]:
            if hasattr(player, 'kill'): 
                try:
                    player.kill()
                except Exception as e:
                    print(f"Warning: Error killing {player.name}: {e}")
            
            if hasattr(player, 'start_process'):
                try:
                    player.start_process()
                except Exception as e:
                    print(f"Warning: Error restarting {player.name}: {e}")
            
        # Reset board state
        self.board.reset()
        self.white_time = settings.GAME_TIME_LIMIT
        self.black_time = settings.GAME_TIME_LIMIT
        self.time_history.clear()
        self.game_over_reason = ""
        self.is_paused = True
        
        # Reset the game record for new game
        self.game_record = GameRecord(self.white_player.name, self.black_player.name)