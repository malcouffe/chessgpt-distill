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
import sys
import time

import torch
from datasets import load_dataset as hf_load_dataset

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
    # Strip torch.compile's '_orig_mod.' prefix if present
    state_dict = ckpt["model_state_dict"]
    state_dict = {k.removeprefix("_orig_mod."): v for k, v in state_dict.items()}
    info = model.load_state_dict(state_dict, strict=False)
    if info.missing_keys:
        print(f"  WARNING: missing keys: {info.missing_keys}")
    if info.unexpected_keys:
        print(f"  WARNING: unexpected keys: {info.unexpected_keys}")
    model.eval()

    step = ckpt.get("step", "?")
    return model, tokenizer, step


# ---------------------------------------------------------------------------
# Test set loading
# ---------------------------------------------------------------------------

def load_test_set(path: str, max_positions: int = 0) -> list[Position]:
    ds = hf_load_dataset(path, split="train")
    positions = []
    for d in ds:
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
    parser.add_argument("--test_set", type=str, default="malcouffe/chessgpt-test-positions",
                        help="HF Hub dataset repo name")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--stockfish_path", type=str, default="stockfish")
    parser.add_argument("--sf_depth", type=int, default=15)
    parser.add_argument("--num_engines", type=int, default=4)
    parser.add_argument("--max_positions", type=int, default=0,
                        help="Limit number of positions (0 = all)")
    parser.add_argument("--search", type=str, default="greedy",
                        choices=["greedy", "greedy_legal", "mcts", "mcts_sf"],
                        help="Move selection strategy")
    parser.add_argument("--mcts_simulations", type=int, default=100)
    parser.add_argument("--mcts_cpuct", type=float, default=1.5)
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

    # Build move selector
    move_selector = None
    if args.search == "greedy_legal":
        from chessgpt_distill.search import search_greedy_legal
        move_selector = search_greedy_legal
    elif args.search == "mcts":
        from chessgpt_distill.search import mcts_search
        from functools import partial
        move_selector = partial(
            mcts_search,
            num_simulations=args.mcts_simulations,
            c_puct=args.mcts_cpuct,
        )
    elif args.search == "mcts_sf":
        # mcts_sf needs the pool — will be set inside the context manager
        pass

    with StockfishPool(sf_cfg) as pool:
        if args.search == "mcts_sf":
            from chessgpt_distill.search import mcts_search_sf
            from functools import partial
            move_selector = partial(
                mcts_search_sf,
                pool=pool,
                sf_depth=3,
                num_simulations=args.mcts_simulations,
                c_puct=args.mcts_cpuct,
            )

        print(f"\nRunning evaluation (search={args.search})...")
        t0 = time.time()
        result = full_evaluation(
            model=model,
            tokenizer=tokenizer,
            positions=positions,
            pool=pool,
            device=device,
            sf_depth=args.sf_depth,
            move_selector=move_selector,
        )
        elapsed = time.time() - t0

    print(f"\n{'=' * 60}")
    print(f"EVALUATION RESULTS (step {step})")
    print(f"{'=' * 60}")
    print(result)
    print(f"\nTime: {elapsed:.1f}s ({elapsed / max(result.count, 1):.2f}s/position)")


if __name__ == "__main__":
    main()
