#!/usr/bin/env python3
"""Evaluate a ChessGPT checkpoint on a test position set.

Usage:
    python scripts/evaluate.py \
        --checkpoint /path/to/step_120000.pt \
        --test_set data/test_positions.jsonl \
        --stockfish_path stockfish \
        --sf_depth 15 \
        --num_engines 4 \
        --max_positions 0  # 0 = all
"""

from __future__ import annotations

import argparse
import json
import sys
import time

import torch

from chessgpt import ChessGPT, ChessGPTConfig, UCITokenizer, resolve_device
from chessgpt_distill.evaluation import Position, full_evaluation
from chessgpt_distill.stockfish import StockfishConfig, StockfishPool


# ---------------------------------------------------------------------------
# Checkpoint loading (same logic as scripts/generate.py in ChessGPT repo)
# ---------------------------------------------------------------------------

def load_checkpoint(checkpoint_path: str, device: torch.device):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    if "config" in ckpt and "model" in ckpt.get("config", {}):
        model_d = ckpt["config"]["model"]
    else:
        model_d = ckpt.get("train_config", {})

    tokenizer = UCITokenizer()

    model_cfg = ChessGPTConfig(
        vocab_size=tokenizer.vocab_size,
        d_model=int(model_d.get("d_model", 256)),
        n_layers=int(model_d.get("n_layers", 8)),
        n_heads=int(model_d.get("n_heads", 8)),
        d_ff=int(model_d.get("d_ff", 1024)),
        max_seq_len=int(model_d.get("max_seq_len", 256)),
        dropout=float(model_d.get("dropout", 0.0)),
    )

    model = ChessGPT(model_cfg).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    step = ckpt.get("step", "?")
    return model, tokenizer, step


# ---------------------------------------------------------------------------
# Test set loading
# ---------------------------------------------------------------------------

def load_test_set(path: str, max_positions: int = 0) -> list[Position]:
    positions = []
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            positions.append(Position(
                move_history=d["move_history"],
                phase=d["phase"],
                expected_move=d.get("expected_move"),
                elo=d.get("elo"),
            ))
            if 0 < max_positions <= len(positions):
                break
    return positions


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Evaluate ChessGPT on test positions")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--test_set", type=str, default="data/test_positions.jsonl")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--stockfish_path", type=str, default="stockfish")
    parser.add_argument("--sf_depth", type=int, default=15)
    parser.add_argument("--num_engines", type=int, default=4)
    parser.add_argument("--max_positions", type=int, default=0,
                        help="Limit number of positions (0 = all)")
    args = parser.parse_args()

    device = resolve_device(args.device)
    print(f"Device: {device}")

    # Load model
    print(f"Loading checkpoint: {args.checkpoint}")
    model, tokenizer, step = load_checkpoint(args.checkpoint, device)
    print(f"  Step: {step}, params: {model.count_parameters():,}")

    # Load test set
    print(f"Loading test set: {args.test_set}")
    positions = load_test_set(args.test_set, args.max_positions)
    print(f"  Positions: {len(positions)}")

    phase_counts = {}
    for pos in positions:
        phase_counts[pos.phase] = phase_counts.get(pos.phase, 0) + 1
    for phase, count in sorted(phase_counts.items()):
        print(f"  {phase}: {count}")

    # Stockfish pool
    sf_cfg = StockfishConfig(
        binary_path=args.stockfish_path,
        depth=args.sf_depth,
        num_engines=args.num_engines,
    )
    print(f"Starting Stockfish pool ({sf_cfg.num_engines} engines, depth {sf_cfg.depth})")

    with StockfishPool(sf_cfg) as pool:
        print("\nRunning evaluation...")
        t0 = time.time()
        result = full_evaluation(
            model=model,
            tokenizer=tokenizer,
            positions=positions,
            pool=pool,
            device=device,
            sf_depth=args.sf_depth,
        )
        elapsed = time.time() - t0

    print(f"\n{'=' * 60}")
    print(f"EVALUATION RESULTS (step {step})")
    print(f"{'=' * 60}")
    print(result)
    print(f"\nTime: {elapsed:.1f}s ({elapsed / max(result.count, 1):.2f}s/position)")


if __name__ == "__main__":
    main()
