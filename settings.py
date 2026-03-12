# settings.py

# Window configuration
WIDTH, HEIGHT = 600, 700
BOARD_HEIGHT = 550
DIMENSION = 8
SQ_SIZE = BOARD_HEIGHT // DIMENSION
FPS = 120

# Time control
GAME_TIME_LIMIT = 600.0  
INCREMENT_SEC = 5.0      

# Visuals
LIGHT_SQ = (240, 217, 181) 
DARK_SQ = (181, 136, 99)
HIGHLIGHT_COLOR = (0, 255, 255)
UI_BG = (40, 40, 40)
TEXT_COLOR = (255, 255, 255)

PIECE_SYMBOLS = {
    'P': '♙', 'R': '♖', 'N': '♘', 'B': '♗', 'Q': '♕', 'K': '♔',
    'p': '♟', 'r': '♜', 'n': '♞', 'b': '♝', 'q': '♛', 'k': '♚'
}