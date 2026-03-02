"""Thread-safe pool of Stockfish engines for parallel evaluation."""

from __future__ import annotations

import queue
import threading
from dataclasses import dataclass, field

import chess
import chess.engine


@dataclass
class StockfishConfig:
    binary_path: str = "stockfish"
    depth: int = 12
    threads: int = 1
    hash_mb: int = 64
    num_engines: int = 4


MATE_SCORE = 10_000


class StockfishPool:
    """Pool of Stockfish engine instances for parallel evaluation.

    Usage::

        pool = StockfishPool(StockfishConfig(num_engines=4))
        score = pool.evaluate(board)
        top5 = pool.top_moves(board, n=5)
        pool.close()

    Thread-safe: multiple threads can call evaluate/top_moves concurrently.
    """

    def __init__(self, cfg: StockfishConfig | None = None):
        cfg = cfg or StockfishConfig()
        self.cfg = cfg
        self._pool: queue.Queue[chess.engine.SimpleEngine] = queue.Queue()

        for _ in range(cfg.num_engines):
            engine = chess.engine.SimpleEngine.popen_uci(cfg.binary_path)
            engine.configure({"Threads": cfg.threads, "Hash": cfg.hash_mb})
            self._pool.put(engine)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate(self, board: chess.Board, depth: int | None = None) -> int:
        """Evaluate a position. Returns centipawns from white's perspective.

        Mates are scored as +/- MATE_SCORE.
        """
        depth = depth or self.cfg.depth
        engine = self._acquire()
        try:
            info = engine.analyse(board, chess.engine.Limit(depth=depth))
            return self._score_to_cp(info["score"].white())
        finally:
            self._release(engine)

    def top_moves(
        self,
        board: chess.Board,
        n: int = 5,
        depth: int | None = None,
    ) -> list[tuple[str, int]]:
        """Return top-N moves with centipawn scores (white perspective).

        Returns list of (uci_move, centipawns).
        """
        depth = depth or self.cfg.depth
        n = min(n, board.legal_moves.count())
        if n == 0:
            return []

        engine = self._acquire()
        try:
            infos = engine.analyse(
                board, chess.engine.Limit(depth=depth), multipv=n,
            )
            results = []
            for info in infos:
                move_uci = info["pv"][0].uci()
                score_cp = self._score_to_cp(info["score"].white())
                results.append((move_uci, score_cp))
            return results
        finally:
            self._release(engine)

    def close(self):
        """Shut down all engines."""
        while not self._pool.empty():
            try:
                engine = self._pool.get_nowait()
                engine.quit()
            except queue.Empty:
                break

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _acquire(self) -> chess.engine.SimpleEngine:
        return self._pool.get()

    def _release(self, engine: chess.engine.SimpleEngine):
        self._pool.put(engine)

    @staticmethod
    def _score_to_cp(score: chess.engine.PovScore | chess.engine.Cp | chess.engine.Mate) -> int:
        """Convert a python-chess score to centipawns."""
        if score.is_mate():
            mate_in = score.mate()
            return MATE_SCORE if mate_in > 0 else -MATE_SCORE
        return score.score()
