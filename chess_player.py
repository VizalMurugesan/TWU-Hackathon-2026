# chess_player.py
import chess

class ChessPlayer:
    def __init__(self, name):
        self.name = name
        self.color = None
        self.is_human = False  # <--- ADD THIS FLAG

    def initialize(self, color):
        self.color = color

    def make_move(self, board):
        pass