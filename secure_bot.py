import multiprocessing
import chess
import time
import traceback
from chess_player import ChessPlayer

# This function runs in the SEPARATE process
def bot_worker(bot_cls, name, input_queue, output_queue):
    try:
        # Instantiate the bot inside the isolated process
        bot_instance = bot_cls(name)
        bot_instance.initialize(chess.WHITE) # Color doesn't strictly matter for calculation here
        
        while True:
            # 1. Wait for a board FEN (blocking wait inside this process only)
            task = input_queue.get()
            if task == "STOP":
                break
            
            fen = task
            board = chess.Board(fen)
            
            try:
                # 2. Run the potentially dangerous user code
                move = bot_instance.make_move(board)
                
                # 3. Send back the result safe and sound
                if move:
                    output_queue.put(("MOVE", move.uci()))
                else:
                    output_queue.put(("ERROR", "Bot returned None"))
                    
            except Exception as e:
                # Catch logic crashes (e.g. 1/0)
                error_msg = traceback.format_exc()
                output_queue.put(("CRASH", error_msg))

    except Exception as e:
        output_queue.put(("CRASH", f"Worker Setup Failed: {e}"))

class SecureBotWrapper(ChessPlayer):
    def __init__(self, target_bot_cls, name):
        super().__init__(name)
        self.target_bot_cls = target_bot_cls
        self.process = None
        self.input_queue = multiprocessing.Queue()
        self.output_queue = multiprocessing.Queue()
        self.is_thinking = False

    def start_process(self):
        if self.process and self.process.is_alive():
            return
            
        # Spawn the worker
        self.process = multiprocessing.Process(
            target=bot_worker, 
            args=(self.target_bot_cls, self.name, self.input_queue, self.output_queue)
        )
        self.process.daemon = True # Kills subprocess if main app dies
        self.process.start()

    def request_move(self, board):
        """Non-blocking request to start thinking"""
        # Clean out old messages
        while not self.output_queue.empty():
            self.output_queue.get()
            
        # Send FEN (Safe string representation)
        self.input_queue.put(board.fen())
        self.is_thinking = True

    def check_result(self):
        """
        Called every frame by GameManager.
        Returns: 
           None -> Still thinking
           (str) -> UCI move string
           Exception -> If crash occurred
        """
        try:
            # Check if data is available without blocking
            if not self.output_queue.empty():
                msg_type, content = self.output_queue.get_nowait()
                self.is_thinking = False
                
                if msg_type == "MOVE":
                    return chess.Move.from_uci(content)
                elif msg_type == "CRASH":
                    raise RuntimeError(f"Bot Crashed: {content}")
                elif msg_type == "ERROR":
                    raise RuntimeError(f"Bot Logic Error: {content}")
                    
        except multiprocessing.queues.Empty:
            pass
            
        return None

    def kill(self):
        if self.process and self.process.is_alive():
            self.process.terminate()
            self.process.join()
            self.is_thinking = False