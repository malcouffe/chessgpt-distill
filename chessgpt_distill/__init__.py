"""ChessGPT-RL -- evaluation, distillation, and RL fine-tuning for ChessGPT."""

from .stockfish import StockfishPool  # noqa: F401
from .evaluation import (  # noqa: F401
    stockfish_agreement,
    average_centipawn_loss,
    legality_rate,
)
