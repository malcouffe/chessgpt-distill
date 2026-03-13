"""Evaluation metrics for ChessGPT playing strength."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import chess
import torch
import torch.nn.functional as F

from chessgpt import ChessGPT, UCITokenizer

from .stockfish import StockfishPool


@dataclass
class Position:
    """A test position with move history and metadata."""

    move_history: list[str]  # UCI moves leading to this position
    phase: str  # "opening", "middlegame", "endgame"
    expected_move: str | None = None  # human move played (optional)
    elo: int | None = None


@dataclass
class EvalResult:
    """Aggregated evaluation results."""

    agreement_top1: float  # fraction of model top-1 matching SF top-1
    agreement_top5: float  # fraction of model top-1 in SF top-5
    avg_cpl: float  # average centipawn loss
    legality_rate: float  # fraction of model top-1 that are legal
    count: int  # number of positions evaluated

    # Per-phase breakdown
    cpl_by_phase: dict[str, float]
    agreement_top1_by_phase: dict[str, float]

    def __str__(self) -> str:
        lines = [
            f"Positions evaluated : {self.count}",
            f"Stockfish agreement : top-1 = {self.agreement_top1:.1%}, top-5 = {self.agreement_top5:.1%}",
            f"Average CPL         : {self.avg_cpl:.1f}",
            f"Legality rate       : {self.legality_rate:.1%}",
            "",
            "Per-phase breakdown:",
        ]
        for phase in ("opening", "middlegame", "endgame"):
            cpl = self.cpl_by_phase.get(phase, float("nan"))
            agr = self.agreement_top1_by_phase.get(phase, float("nan"))
            lines.append(f"  {phase:12s} : CPL = {cpl:6.1f}, agreement = {agr:.1%}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Core metric functions
# ---------------------------------------------------------------------------


def _build_legal_mask(
    board: chess.Board,
    tokenizer: UCITokenizer,
    device: torch.device,
) -> torch.Tensor:
    """Boolean mask (vocab_size,) where True = legal move."""
    mask = torch.zeros(tokenizer.vocab_size, dtype=torch.bool, device=device)
    for mv in board.legal_moves:
        tid = tokenizer.move_to_id.get(mv.uci())
        if tid is not None:
            mask[tid] = True
    return mask


@torch.no_grad()
def _get_model_top1(
    model: ChessGPT,
    tokenizer: UCITokenizer,
    move_history: list[str],
    device: torch.device,
) -> tuple[str | None, bool]:
    """Get the model's greedy top-1 move (no legal masking).

    Returns (move_uci_or_None, is_legal).
    """
    # Encode history
    ids = [tokenizer.BOS_ID]
    for mv in move_history:
        tid = tokenizer.move_to_id.get(mv)
        if tid is not None:
            ids.append(tid)

    # Crop to max context
    max_ctx = model.config.max_seq_len
    if len(ids) > max_ctx:
        ids = ids[-max_ctx:]

    input_ids = torch.tensor([ids], dtype=torch.long, device=device)

    with torch.amp.autocast(device_type=device.type, enabled=(device.type == "cuda")):
        logits, _, _ = model(input_ids)

    logits = logits[0, -1, :]  # (V,)

    # Exclude special tokens
    logits[tokenizer.PAD_ID] = float("-inf")
    logits[tokenizer.BOS_ID] = float("-inf")
    logits[tokenizer.EOS_ID] = float("-inf")

    top1_id = torch.argmax(logits).item()
    move_uci = tokenizer.id_to_move.get(top1_id)

    if move_uci is None:
        return None, False

    # Check legality
    board = chess.Board()
    for mv in move_history:
        board.push_uci(mv)

    try:
        chess_move = chess.Move.from_uci(move_uci)
        is_legal = chess_move in board.legal_moves
    except ValueError:
        is_legal = False

    return move_uci, is_legal


def stockfish_agreement(
    model: ChessGPT,
    tokenizer: UCITokenizer,
    positions: list[Position],
    pool: StockfishPool,
    device: torch.device,
    sf_depth: int = 15,
) -> tuple[float, float]:
    """Compute Stockfish agreement (top-1 and top-5).

    Returns (agreement_top1, agreement_top5).
    """
    top1_match = 0
    top5_match = 0
    total = 0

    for pos in positions:
        board = chess.Board()
        for mv in pos.move_history:
            board.push_uci(mv)

        if board.is_game_over():
            continue

        model_move, _ = _get_model_top1(model, tokenizer, pos.move_history, device)
        if model_move is None:
            total += 1
            continue

        sf_top = pool.top_moves(board, n=5, depth=sf_depth)
        sf_moves = [m for m, _ in sf_top]

        if sf_moves and model_move == sf_moves[0]:
            top1_match += 1
        if model_move in sf_moves:
            top5_match += 1
        total += 1

    if total == 0:
        return 0.0, 0.0
    return top1_match / total, top5_match / total


def average_centipawn_loss(
    model: ChessGPT,
    tokenizer: UCITokenizer,
    positions: list[Position],
    pool: StockfishPool,
    device: torch.device,
    sf_depth: int = 15,
) -> float:
    """Compute average centipawn loss of model's top-1 move vs Stockfish best."""
    total_cpl = 0.0
    count = 0

    for pos in positions:
        board = chess.Board()
        for mv in pos.move_history:
            board.push_uci(mv)

        if board.is_game_over():
            continue

        model_move, is_legal = _get_model_top1(
            model, tokenizer, pos.move_history, device,
        )
        if model_move is None or not is_legal:
            # Illegal move = max penalty
            total_cpl += 500.0
            count += 1
            continue

        # Get best move eval
        sf_top = pool.top_moves(board, n=1, depth=sf_depth)
        if not sf_top:
            continue
        best_cp = sf_top[0][1]

        # Get model move eval
        board.push_uci(model_move)
        model_cp = -pool.evaluate(board, depth=sf_depth)  # negate: opponent's perspective
        board.pop()

        # CPL from the perspective of the side to move
        if board.turn == chess.WHITE:
            cpl = best_cp - model_cp
        else:
            cpl = model_cp - best_cp

        total_cpl += max(cpl, 0)  # CPL is non-negative by definition
        count += 1

    return total_cpl / count if count > 0 else 0.0


def legality_rate(
    model: ChessGPT,
    tokenizer: UCITokenizer,
    positions: list[Position],
    device: torch.device,
) -> float:
    """Fraction of positions where the model's greedy move (no masking) is legal."""
    legal = 0
    total = 0

    for pos in positions:
        board = chess.Board()
        for mv in pos.move_history:
            board.push_uci(mv)

        if board.is_game_over():
            continue

        _, is_legal = _get_model_top1(model, tokenizer, pos.move_history, device)
        if is_legal:
            legal += 1
        total += 1

    return legal / total if total > 0 else 0.0


MoveSelector = Callable[
    [ChessGPT, UCITokenizer, list[str], chess.Board, torch.device],
    str | None,
]


def full_evaluation(
    model: ChessGPT,
    tokenizer: UCITokenizer,
    positions: list[Position],
    pool: StockfishPool,
    device: torch.device,
    sf_depth: int = 15,
    move_selector: MoveSelector | None = None,
) -> EvalResult:
    """Run all evaluation metrics on a set of positions.

    If move_selector is provided, it is used instead of the default greedy
    top-1 (no legal masking). The selector receives (model, tokenizer,
    move_history, board, device) and returns a UCI move string or None.
    """
    # Per-position accumulators
    top1_matches = 0
    top5_matches = 0
    total_cpl = 0.0
    legal_count = 0
    total = 0

    # Per-phase accumulators
    phase_cpl: dict[str, list[float]] = {"opening": [], "middlegame": [], "endgame": []}
    phase_top1: dict[str, list[bool]] = {"opening": [], "middlegame": [], "endgame": []}

    for pos in positions:
        board = chess.Board()
        for mv in pos.move_history:
            board.push_uci(mv)

        if board.is_game_over():
            continue

        total += 1

        if move_selector is not None:
            model_move = move_selector(model, tokenizer, pos.move_history, board, device)
            # Custom selectors are assumed to return legal moves
            is_legal = model_move is not None
        else:
            # Default: greedy top-1 without legal masking
            model_move, is_legal = _get_model_top1(
                model, tokenizer, pos.move_history, device,
            )

        if is_legal:
            legal_count += 1

        # Stockfish top-5
        sf_top = pool.top_moves(board, n=5, depth=sf_depth)
        sf_moves = [m for m, _ in sf_top]

        t1_match = (model_move is not None and sf_moves and model_move == sf_moves[0])
        t5_match = (model_move is not None and model_move in sf_moves)

        if t1_match:
            top1_matches += 1
        if t5_match:
            top5_matches += 1

        # CPL
        if model_move is not None and is_legal and sf_top:
            best_cp = sf_top[0][1]
            board.push_uci(model_move)
            model_cp = -pool.evaluate(board, depth=sf_depth)
            board.pop()

            if board.turn == chess.WHITE:
                cpl = best_cp - model_cp
            else:
                cpl = model_cp - best_cp
            cpl = max(cpl, 0)
        else:
            cpl = 500.0  # penalty for illegal/missing moves

        total_cpl += cpl

        # Phase tracking
        phase = pos.phase
        if phase in phase_cpl:
            phase_cpl[phase].append(cpl)
            phase_top1[phase].append(t1_match)

    if total == 0:
        return EvalResult(
            agreement_top1=0.0, agreement_top5=0.0, avg_cpl=0.0,
            legality_rate=0.0, count=0,
            cpl_by_phase={}, agreement_top1_by_phase={},
        )

    cpl_by_phase = {}
    agr_by_phase = {}
    for phase in ("opening", "middlegame", "endgame"):
        vals = phase_cpl.get(phase, [])
        cpl_by_phase[phase] = sum(vals) / len(vals) if vals else 0.0
        matches = phase_top1.get(phase, [])
        agr_by_phase[phase] = sum(matches) / len(matches) if matches else 0.0

    return EvalResult(
        agreement_top1=top1_matches / total,
        agreement_top5=top5_matches / total,
        avg_cpl=total_cpl / total,
        legality_rate=legal_count / total,
        count=total,
        cpl_by_phase=cpl_by_phase,
        agreement_top1_by_phase=agr_by_phase,
    )
