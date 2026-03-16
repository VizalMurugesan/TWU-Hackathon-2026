# main.py
import multiprocessing
from chess_ui import ChessUI
# Make sure to import SmartBot from bots.py
from bots import RandomBot, PacifistBot, HumanPlayer, FreezerBot, CrasherBot, SmartBot
from secure_bot import SecureBotWrapper
from chesschamp import ChessChamp 
from KenYuDai_bot import KenYuDai
from hackathon_bot import HackathonBot
from Dabot import Dabot
from ratnas_nightmare import RatnasNightmare

if __name__ == "__main__":
    multiprocessing.freeze_support()
    
    print("--- CHESS TOURNAMENT MODE ---")
    print("1. Bot vs Bot (Pacifist vs SmartBot)")
    print("2. Human vs Bot (You vs SmartBot)")
    print("3. Human vs Human")
    
    choice = input("Select Mode (1, 2, or 3): ")

    #ChessBots
    ASquare = SecureBotWrapper(ChessChamp, "Chess Champ")
    YuKenDai = SecureBotWrapper(KenYuDai, "KenYuDai")
    OnARL = SecureBotWrapper(HackathonBot, "OnARL hackathonbot")
    GrandMaster_Inferno = SecureBotWrapper(Dabot, "Dabot")
    Memory_Shortage_Algoritm = SecureBotWrapper(RatnasNightmare, "Ratna's Nightmare")


    if choice == "2":
        player1 = HumanPlayer("You")
        # We wrap SmartBot to keep the UI responsive while it thinks
        player2 = ASquare
        
    elif choice == "3": # <--- FIXED: changed 'if' to 'elif'
        player1 = HumanPlayer("Player1")
        player2 = HumanPlayer("Player2")
        
    else:
        # Default: Watch two bots fight
        player1 =  ASquare
        player2 =  Memory_Shortage_Algoritm

    # Launch the game
    ui = ChessUI(white_bot=player1, black_bot=player2)
    ui.run()