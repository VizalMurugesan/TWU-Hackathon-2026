import pygame
import chess
import settings
from game_manager import GameManager

class ChessUI:
    def __init__(self, white_bot, black_bot):
        pygame.init()
        self.screen = pygame.display.set_mode((settings.WIDTH, settings.HEIGHT))
        pygame.display.set_caption("Chess Bot Tournament System")
        self.clock = pygame.time.Clock()
        
        # --- DYNAMIC FONT SIZING ---
        sq_size = settings.SQ_SIZE
        
        self.piece_font = pygame.font.SysFont("segoe ui symbol", int(sq_size * 0.85))
        self.timer_font = pygame.font.SysFont("consolas", int(sq_size * 0.45), bold=True)
        self.header_font = pygame.font.SysFont("arial", int(sq_size * 0.35), bold=True)
        self.ui_font = pygame.font.SysFont("arial", int(sq_size * 0.25))

        self.manager = GameManager(white_bot, black_bot)
        self.selected_square = None

    def get_square_under_mouse(self, pos):
        x, y = pos
        if y >= settings.BOARD_HEIGHT: return None 
        col = x // settings.SQ_SIZE
        row = y // settings.SQ_SIZE
        return chess.square(col, 7 - row)

    def handle_click(self, pos):
        # --- LOGIC FIX START ---
        # 1. If game is Paused, we always allow manual edits/clicks.
        if self.manager.is_paused:
            can_click = True
        else:
            # 2. If game is Running, we only allow clicks if it's a Human's turn.
            if self.manager.board.turn == chess.WHITE:
                can_click = self.manager.white_player.is_human
            else:
                can_click = self.manager.black_player.is_human

        # If we are not allowed to click (e.g., game running and it's Bot's turn), stop here.
        if not can_click:
            return 
        # --- LOGIC FIX END ---

        clicked_sq = self.get_square_under_mouse(pos)
        if clicked_sq is None: return 

        if self.selected_square is None:
            piece = self.manager.board.piece_at(clicked_sq)
            # Only allow selecting your own pieces!
            if piece and piece.color == self.manager.board.turn:
                self.selected_square = clicked_sq
        else:
            if clicked_sq == self.selected_square:
                self.selected_square = None
            else:
                # Try to move
                success = self.manager.try_manual_move(self.selected_square, clicked_sq)
                if success:
                    self.selected_square = None 
                else:
                    # If move failed (illegal), check if they clicked a different friendly piece
                    piece = self.manager.board.piece_at(clicked_sq)
                    if piece and piece.color == self.manager.board.turn:
                        self.selected_square = clicked_sq
                    else:
                        self.selected_square = None

    def format_time(self, seconds):
        mins = int(seconds // 60)
        secs = int(seconds % 60)
        millis = int((seconds - int(seconds)) * 100) 
        return f"{mins:02}:{secs:02}.{millis:02}"

    def draw_board(self):
        for r in range(settings.DIMENSION):
            for c in range(settings.DIMENSION):
                color = settings.LIGHT_SQ if (r + c) % 2 == 0 else settings.DARK_SQ
                rect = pygame.Rect(c*settings.SQ_SIZE, r*settings.SQ_SIZE, settings.SQ_SIZE, settings.SQ_SIZE)
                pygame.draw.rect(self.screen, color, rect)
                
                if self.selected_square is not None:
                    sel_file = chess.square_file(self.selected_square)
                    sel_rank = chess.square_rank(self.selected_square)
                    if c == sel_file and r == (7 - sel_rank):
                        pygame.draw.rect(self.screen, settings.HIGHLIGHT_COLOR, rect, 5)

                if self.manager.board.move_stack:
                    last_move = self.manager.board.peek()
                    if (c == chess.square_file(last_move.from_square) and r == 7-chess.square_rank(last_move.from_square)) or \
                       (c == chess.square_file(last_move.to_square) and r == 7-chess.square_rank(last_move.to_square)):
                       pygame.draw.rect(self.screen, (200, 200, 50), rect, 4)

    def draw_pieces(self):
        board = self.manager.board
        for i in range(64):
            piece = board.piece_at(i)
            if piece:
                col = chess.square_file(i)
                row = 7 - chess.square_rank(i) 
                symbol = settings.PIECE_SYMBOLS[piece.symbol()]
                text_surface = self.piece_font.render(symbol, True, (0,0,0))
                text_rect = text_surface.get_rect(center=(col*settings.SQ_SIZE + settings.SQ_SIZE//2, row*settings.SQ_SIZE + settings.SQ_SIZE//2))
                self.screen.blit(text_surface, text_rect)

    def draw_ui(self):
        pygame.draw.rect(self.screen, settings.UI_BG, (0, settings.BOARD_HEIGHT, settings.WIDTH, settings.HEIGHT - settings.BOARD_HEIGHT))
        
        w_time_str = self.format_time(self.manager.white_time)
        b_time_str = self.format_time(self.manager.black_time)

        w_color = (0, 255, 0) if self.manager.board.turn == chess.WHITE else (150, 150, 150)
        b_color = (0, 255, 0) if self.manager.board.turn == chess.BLACK else (150, 150, 150)
        
        # Calculate dynamic positions based on board height
        name_y = settings.BOARD_HEIGHT + int(settings.SQ_SIZE * 0.15)
        timer_y = settings.BOARD_HEIGHT + int(settings.SQ_SIZE * 0.55)
        status_y = settings.BOARD_HEIGHT + int(settings.SQ_SIZE * 1.2)
        
        # Draw Names
        self.screen.blit(self.header_font.render(f"White: {self.manager.white_player.name}", True, w_color), (20, name_y))
        
        # Align Black's name to the right side dynamically
        b_text_surf = self.header_font.render(f"Black: {self.manager.black_player.name}", True, b_color)
        b_x = settings.WIDTH - b_text_surf.get_width() - 20
        self.screen.blit(b_text_surf, (b_x, name_y))
        
        # Draw Timers
        self.screen.blit(self.timer_font.render(w_time_str, True, (255, 255, 255)), (20, timer_y))
        
        b_timer_surf = self.timer_font.render(b_time_str, True, (255, 255, 255))
        b_timer_x = settings.WIDTH - b_timer_surf.get_width() - 20
        self.screen.blit(b_timer_surf, (b_timer_x, timer_y))

        # Game Status
        if self.manager.game_over_reason:
            status_text = f"GAME OVER: {self.manager.game_over_reason}"
            color = (255, 50, 50)
        elif self.manager.board.is_game_over():
            res = self.manager.board.result()
            status_text = f"GAME OVER: {res}"
            color = (255, 50, 50)
        elif self.manager.is_paused:
            status_text = "PAUSED (Manual Mode)"
            color = (100, 200, 255)
        else:
            status_text = "RUNNING"
            color = (100, 255, 100)

        self.screen.blit(self.ui_font.render(status_text, True, color), (20, status_y))
        
        # Instructions aligned right
        instr_text = "Space: Run | R: Reset"
        instr_surf = self.ui_font.render(instr_text, True, (200,200,200))
        instr_x = settings.WIDTH - instr_surf.get_width() - 20
        self.screen.blit(instr_surf, (instr_x, status_y))

    def run(self):
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        self.manager.is_paused = not self.manager.is_paused
                        self.selected_square = None
                    if event.key == pygame.K_r:
                        self.manager.reset_game()
                        self.selected_square = None
                    if event.key == pygame.K_LEFT:
                        self.manager.undo_move()
                        self.selected_square = None

                if event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1:
                        self.handle_click(pygame.mouse.get_pos())

            self.manager.update()
            
            self.screen.fill(settings.UI_BG)
            self.draw_board()
            self.draw_pieces()
            self.draw_ui()
            pygame.display.flip()
            self.clock.tick(settings.FPS)

        pygame.quit()