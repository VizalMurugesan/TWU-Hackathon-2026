import chess
from chess_player import ChessPlayer
import subprocess
import time
import os
import sys
import shutil

class Maia1900Bot(ChessPlayer):
    """
    Maia-1900 via Nix-packaged lc0 (nixpkgs#lc0).
    No explicit backend (uses lc0 default that worked in manual test).
    """

    def __init__(self, *args, **kwargs):
        # Accept any extra args/kwargs from SecureBotWrapper / framework
        super().__init__(*args, **kwargs)
        self._engine = None
        self._started = False
        self.lc0_path = self._find_lc0()

    def _find_lc0(self):
        candidates = [
            shutil.which("lc0"),
            "/run/current-system/sw/bin/lc0",
        ]
        for cand in [c for c in candidates if c]:
            if os.path.isfile(cand) and os.access(cand, os.X_OK):
                print(f"[Maia-1900] Found lc0 at: {cand}", file=sys.stderr)
                return cand

        raise FileNotFoundError(
            "[Maia-1900] Could not find 'lc0' executable.\n"
            "Make sure you run with: nix shell nixpkgs#lc0 --command python main.py\n"
            "or have lc0 in your PATH."
        )

    def _ensure_engine(self):
        if self._started:
            return True

        weights = os.path.abspath("./maia-1900.pb.gz")

        if not os.path.isfile(weights):
            print(f"[Maia-1900] Missing weights file: {weights}", file=sys.stderr)
            return False

        try:
            cmd = [
                self.lc0_path,
                f"--weights={weights}",
                "--threads=1",
                "--nncache_size=200000",
                "--move-time=40000",
                "--logfile=maia_lc0.log"
            ]

            print(f"[Maia-1900] Launching lc0: {' '.join(cmd)}", file=sys.stderr)

            self._engine = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,   # merge errors into stdout
                text=True,
                bufsize=1,
                universal_newlines=True,
                cwd=os.getcwd()
            )

            time.sleep(3.0)  # Give time for weights loading

            self._write("uci")
            resp = self._read_until("uciok", max_lines=200)

            print(f"[Maia-1900] Raw lc0 response after 'uci':\n{resp}\n", file=sys.stderr)

            if "uciok" not in resp.lower():
                # Try to read more output
                extra = ""
                for _ in range(15):
                    line = self._read_line()
                    if line:
                        extra += line + "\n"
                if extra:
                    print(f"[Maia-1900] Additional output:\n{extra}", file=sys.stderr)
                raise RuntimeError("Did not receive 'uciok' from lc0")

            self._started = True
            print("[Maia-1900] Engine initialized successfully!", file=sys.stderr)
            return True

        except Exception as e:
            print(f"[Maia-1900] Engine startup failed: {str(e)}", file=sys.stderr)
            self._kill_engine()
            return False

    def _write(self, text):
        if self._engine and self._engine.poll() is None:
            try:
                self._engine.stdin.write(text + "\n")
                self._engine.stdin.flush()
            except:
                pass

    def _read_line(self):
        if self._engine and self._engine.poll() is None:
            try:
                return self._engine.stdout.readline().rstrip()
            except:
                return ""
        return ""

    def _read_until(self, keyword, max_lines=200):
        lines = []
        for _ in range(max_lines):
            line = self._read_line()
            if line:
                lines.append(line)
            if keyword.lower() in line.lower():
                break
            time.sleep(0.02)
        return "\n".join(lines)

    def _kill_engine(self):
        if self._engine:
            try:
                self._write("quit")
                self._engine.terminate()
                self._engine.wait(4.0)
            except:
                pass
            self._engine = None
        self._started = False

    def make_move(self, board: chess.Board):
        if board.is_game_over():
            return None

        if not self._ensure_engine():
            print("[Maia-1900] Engine not available → random fallback move", file=sys.stderr)
            moves = list(board.legal_moves)
            return moves[0] if moves else None

        try:
            self._write("ucinewgame")
            time.sleep(0.1)
            self._write(f"position fen {board.fen()}")
            self._write("go movetime 4000")

            output = self._read_until("bestmove", max_lines=300)

            for line in output.splitlines():
                if line.startswith("bestmove"):
                    parts = line.split()
                    if len(parts) > 1:
                        uci = parts[1]
                        try:
                            move = chess.Move.from_uci(uci)
                            if move in board.legal_moves:
                                return move
                        except ValueError:
                            pass

        except Exception as e:
            print(f"[Maia-1900] Error during make_move: {str(e)}", file=sys.stderr)

        # Fallback
        moves = list(board.legal_moves)
        return moves[0] if moves else None
