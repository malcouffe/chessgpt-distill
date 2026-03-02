#!/usr/bin/env python3
"""Generate a Stockfish-annotated dataset for distillation training.

For each game, samples multiple positions and records the top-N Stockfish
moves with centipawn evaluations plus all legal move token IDs.
One JSONL record per game with multiple annotated positions.

Usage:
    python scripts/generate_sf_data.py \
        --output malcouffe/chessgpt-sf-distill \
        --num_positions 5000000 \
        --sf_depth 18 \
        --num_engines 8 \
        --sample_every 8
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import time

import chess
from datasets import Dataset, load_dataset

from chessgpt import UCITokenizer
from chessgpt_distill.stockfish import StockfishConfig, StockfishPool


def annotate_game(
    pool: StockfishPool,
    tokenizer: UCITokenizer,
    moves_uci: list[str],
    sample_every: int,
    top_n: int,
    sf_depth: int,
    max_seq_len: int,
) -> tuple[dict | None, int]:
    """Annotate multiple sampled positions from a single game.

    Returns (record, num_positions) where record is a dict with tokens and
    annotations, or (None, 0) if no valid annotations were produced.
    """
    # Build full token sequence
    tokens = [tokenizer.BOS_ID]
    for mv in moves_uci:
        tid = tokenizer.move_to_id.get(mv)
        if tid is not None:
            tokens.append(tid)

    # Crop to max_seq_len (keep most recent moves)
    if len(tokens) > max_seq_len:
        offset = len(tokens) - max_seq_len
        tokens = tokens[-max_seq_len:]
    else:
        offset = 0

    annotations = []
    board = chess.Board()

    for ply, mv_uci in enumerate(moves_uci):
        # Check if this ply should be sampled
        if ply >= 4 and ply < len(moves_uci) - 1 and (ply - 4) % sample_every == 0:
            if not board.is_game_over() and board.legal_moves.count() > 0:
                # Token index for this position (ply+1 because of BOS)
                token_pos = ply + 1 - offset
                if 0 <= token_pos < len(tokens):
                    top_moves = pool.top_moves(board, n=top_n, depth=sf_depth)

                    sf_targets = []
                    for move_uci, score_cp in top_moves:
                        tid = tokenizer.move_to_id.get(move_uci)
                        if tid is not None:
                            sf_targets.append({
                                "token_id": tid,
                                "move": move_uci,
                                "score_cp": score_cp,
                            })

                    legal_move_ids = []
                    for legal_mv in board.legal_moves:
                        tid = tokenizer.move_to_id.get(legal_mv.uci())
                        if tid is not None:
                            legal_move_ids.append(tid)

                    if sf_targets and legal_move_ids:
                        annotations.append({
                            "position_index": token_pos,
                            "sf_targets": sf_targets,
                            "legal_move_ids": legal_move_ids,
                        })

        board.push_uci(mv_uci)

    if not annotations:
        return None, 0

    record = {"tokens": tokens, "annotations": annotations}
    return record, len(annotations)


def main():
    parser = argparse.ArgumentParser(description="Generate Stockfish-annotated dataset")
    parser.add_argument("--output", type=str, default="malcouffe/chessgpt-sf-distill",
                        help="HF Hub repo name (e.g. malcouffe/chessgpt-sf-distill)")
    parser.add_argument("--num_positions", type=int, default=5_000_000)
    parser.add_argument("--min_elo", type=int, default=1800)
    parser.add_argument("--sample_every", type=int, default=8,
                        help="Sample a position every N plies within each game")
    parser.add_argument("--top_n", type=int, default=10,
                        help="Number of Stockfish top moves to record")
    parser.add_argument("--sf_depth", type=int, default=18)
    parser.add_argument("--max_seq_len", type=int, default=256)
    parser.add_argument("--num_engines", type=int, default=8)
    parser.add_argument("--stockfish_path", type=str, default="stockfish")
    parser.add_argument("--datasets", nargs="+", default=[
        "malcouffe/lichess-standard-rated-2025-07-uci",
        "malcouffe/lichess-standard-rated-2025-08-uci",
        "malcouffe/lichess-standard-rated-2025-09-uci",
        "malcouffe/lichess-standard-rated-2025-10-uci",
        "malcouffe/lichess-standard-rated-2025-11-uci",
        "malcouffe/lichess-standard-rated-2025-12-uci",
        "malcouffe/lichess-standard-rated-2026-01-uci",
    ])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    tokenizer = UCITokenizer()

    sf_cfg = StockfishConfig(
        binary_path=args.stockfish_path,
        depth=args.sf_depth,
        num_engines=args.num_engines,
    )

    local_path = os.path.join(tempfile.gettempdir(), "sf_distill.jsonl")

    print(f"Target: {args.num_positions:,} positions")
    print(f"Stockfish: depth={args.sf_depth}, top_n={args.top_n}, engines={args.num_engines}")
    print(f"Max seq len: {args.max_seq_len}")
    print(f"Datasets: {len(args.datasets)}")
    print(f"HF Hub repo: {args.output}")

    count = 0
    games = 0
    t0 = time.time()

    with StockfishPool(sf_cfg) as pool, open(local_path, "w") as fout:
        for ds_name in args.datasets:
            if count >= args.num_positions:
                break

            print(f"\nLoading {ds_name} ...")
            ds = load_dataset(ds_name, split="train", streaming=True)

            for example in ds:
                if count >= args.num_positions:
                    break

                white_elo = example.get("white_elo", 0)
                black_elo = example.get("black_elo", 0)
                if white_elo < args.min_elo or black_elo < args.min_elo:
                    continue

                moves_uci = example.get("moves", "").strip().split()
                if len(moves_uci) < 10:
                    continue

                record, n_positions = annotate_game(
                    pool, tokenizer, moves_uci,
                    sample_every=args.sample_every,
                    top_n=args.top_n,
                    sf_depth=args.sf_depth,
                    max_seq_len=args.max_seq_len,
                )

                if record is not None:
                    fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                    count += n_positions
                    games += 1

                    if count % 1000 < n_positions:
                        elapsed = time.time() - t0
                        rate = count / elapsed
                        eta = (args.num_positions - count) / rate if rate > 0 else 0
                        print(
                            f"  {count:>10,} / {args.num_positions:,} "
                            f"({count / args.num_positions:.1%}) "
                            f"| {games:,} games "
                            f"| {rate:.0f} pos/s "
                            f"| ETA {eta / 3600:.1f}h",
                        )

    elapsed = time.time() - t0
    print(f"\nDone: {count:,} positions from {games:,} games in {elapsed / 3600:.1f}h")

    # Upload to Hugging Face Hub
    print(f"\nUploading to {args.output} ...")
    hf_dataset = Dataset.from_json(local_path)
    hf_dataset.push_to_hub(args.output)
    print(f"Uploaded to https://huggingface.co/datasets/{args.output}")
    os.remove(local_path)


if __name__ == "__main__":
    main()
