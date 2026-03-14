# elo.py
"""
Runs 100 games between PacifistBot and SmartBot and estimates relative Elo ratings.
Based on typical structure in main.py but extended for batch play + Elo calc.
"""

import chess
import random
import math
from typing import Tuple

# Framework imports (adjust if any import error — these match the repo structure)
from chess_player import ChessPlayer
from secure_bot import SecureBotWrapper
from game_manager import GameManager
from bots import PacifistBot, SmartBot


def play_single_game(white_bot: ChessPlayer, black_bot: ChessPlayer) -> Tuple[str, str]:
    """
    Play one full game using GameManager (preferred way if it works).
    Returns (result, pgn) where result is "1-0", "0-1" or "1/2-1/2".
    """
    try:
        manager = GameManager(white=white_bot, black=black_bot)
        outcome = manager.play_game()  # This should run until game over
        result = outcome.result()      # chess python result string
        # If GameManager exposes PGN-like object:
        pgn = str(outcome.game) if hasattr(outcome, 'game') else ""
        return result, pgn
    except AttributeError:
        # Fallback: manual loop if GameManager doesn't have play_game() or outcome
        print("GameManager fallback: manual board loop")
        board = chess.Board()
        current_player = white_bot

        while not board.is_game_over():
            move = current_player.make_move(board)
            if move not in board.legal_moves:
                raise ValueError("Illegal move returned by bot")
            board.push(move)
            current_player = black_bot if current_player == white_bot else white_bot

        result = board.result()
        pgn = chess.pgn.Game.from_board(board).accept(chess.pgn.StringExporter())
        return result, pgn


def estimate_elo_difference(score: float, n_games: int) -> float:
    """
    Very simple logistic Elo difference approximation.
    score = points for player A (0..1), including 0.5 per draw
    Returns Elo(A) - Elo(B)
    """
    if score <= 0:
        return -800.0
    if score >= 1:
        return +800.0
    return 400.0 * math.log10(score / (1.0 - score))


def run_elo_evaluation(num_games: int = 100):
    # Instantiate bots (wrapped for safety/timeouts as per repo convention)
    pacifist = SecureBotWrapper(PacifistBot, name="PacifistBot")
    smart    = SecureBotWrapper(SmartBot, name="SmartBot")

    # Track results from PacifistBot's perspective
    pacifist_points = 0.0
    wins_p = 0
    wins_s = 0
    draws  = 0

    print(f"Starting {num_games} games: PacifistBot vs SmartBot")
    print("Color alternation enabled\n")

    for i in range(num_games):
        # Alternate who is white
        if i % 2 == 0:
            white, black = pacifist, smart
            white_name, black_name = "PacifistBot", "SmartBot"
        else:
            white, black = smart, pacifist
            white_name, black_name = "SmartBot", "PacifistBot"

        print(f"Game {i+1}/{num_games} • {white_name} (white) vs {black_name} (black) ... ", end="", flush=True)

        result, _ = play_single_game(white, black)  # we ignore pgn for speed

        if result == "1-0":
            winner = white_name
        elif result == "0-1":
            winner = black_name
        else:
            winner = "draw"

        if winner == "PacifistBot":
            wins_p += 1
            pacifist_points += 1.0
            print("Pacifist wins")
        elif winner == "SmartBot":
            wins_s += 1
            pacifist_points += 0.0
            print("SmartBot wins")
        else:
            draws += 1
            pacifist_points += 0.5
            print("Draw")

    # Final stats
    pacifist_score = pacifist_points / num_games
    smart_score    = 1.0 - pacifist_score

    elo_diff = estimate_elo_difference(pacifist_score, num_games)

    print("\n" + "="*60)
    print(f"Results after {num_games} games:")
    print(f"  PacifistBot wins : {wins_p:3d}  ({wins_p/num_games*100:5.1f}%)")
    print(f"  SmartBot    wins : {wins_s:3d}  ({wins_s/num_games*100:5.1f}%)")
    print(f"  Draws            : {draws:3d}  ({draws/num_games*100:5.1f}%)")
    print("-"*60)
    print(f"PacifistBot score : {pacifist_score:.3f}")
    print(f"SmartBot    score : {smart_score:.3f}")
    print()
    print("Approximate Elo ratings (starting from 1200):")
    print(f"  SmartBot    Elo ≈ {1200 + elo_diff:4.0f}")
    print(f"  PacifistBot Elo ≈ {1200 - elo_diff:4.0f}")
    print(f"  Elo difference  : SmartBot +{elo_diff:.0f} over PacifistBot")
    print("="*60)


if __name__ == "__main__":
    random.seed(42)           # reproducible
    run_elo_evaluation(100)   # change number here if desired
