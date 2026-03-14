# main.py
import multiprocessing
from chess_ui import ChessUI
# Make sure to import SmartBot from bots.py
from bots import RandomBot, PacifistBot, HumanPlayer, FreezerBot, CrasherBot, SmartBot
from maia1900_bot import Maia1900Bot
from secure_bot import SecureBotWrapper

if __name__ == "__main__":
    multiprocessing.freeze_support()
    
    print("--- CHESS TOURNAMENT MODE ---")
    print("1. Bot vs Bot (MaiaBot-1900 vs SmartBot)")
    print("2. Human vs Bot (You vs SmartBot)")
    print("3. Human vs Human")
    
    choice = input("Select Mode (1, 2, or 3): ")

    if choice == "2":
        player1 = HumanPlayer("You")
        # We wrap SmartBot to keep the UI responsive while it thinks
        player2 = SecureBotWrapper(SmartBot, "Gemini SmartBot")
        
    elif choice == "3": # <--- FIXED: changed 'if' to 'elif'
        player1 = HumanPlayer("Player1")
        player2 = HumanPlayer("Player2")
        
    else:
        # Default: Watch two bots fight
        player1 = SecureBotWrapper(Maia1900Bot, "MaiaBot-1900")
        player2 = SecureBotWrapper(SmartBot, "Gemini SmartBot")

    # Launch the game
    ui = ChessUI(white_bot=player1, black_bot=player2)
    ui.run()
