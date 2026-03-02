#!/usr/bin/env python3
"""Generate a test set of chess positions from Lichess games for evaluation.

Extracts positions at various game phases from high-ELO games
and uploads them to Hugging Face Hub.

Usage:
    python scripts/generate_test_set.py \
        --output malcouffe/chessgpt-test-positions \
        --num_positions 10000 \
        --min_elo 2200
"""

from __future__ import annotations

import argparse
import json
import os
import random
import tempfile

from datasets import Dataset, load_dataset

# Phase boundaries (in half-moves / plies)
OPENING_END = 30     # moves 1-15
MIDDLEGAME_END = 70  # moves 16-35
# Anything beyond = endgame


def classify_phase(ply: int) -> str:
    if ply < OPENING_END:
        return "opening"
    elif ply < MIDDLEGAME_END:
        return "middlegame"
    else:
        return "endgame"


def extract_positions_from_game(
    moves_uci: str,
    white_elo: int,
    black_elo: int,
    sample_every: int = 5,
) -> list[dict]:
    """Extract positions from a game, sampling every N plies."""
    tokens = moves_uci.strip().split()
    if len(tokens) < 10:
        return []

    positions = []
    for ply in range(4, len(tokens) - 1, sample_every):
        # Skip very early moves (first 2 full moves)
        phase = classify_phase(ply)
        positions.append({
            "move_history": tokens[:ply],
            "expected_move": tokens[ply],
            "phase": phase,
            "elo": min(white_elo, black_elo),
            "ply": ply,
        })

    return positions


def main():
    parser = argparse.ArgumentParser(description="Generate test positions from Lichess games")
    parser.add_argument("--output", type=str, default="malcouffe/chessgpt-test-positions",
                        help="HF Hub repo name (e.g. malcouffe/chessgpt-test-positions)")
    parser.add_argument("--num_positions", type=int, default=10_000)
    parser.add_argument("--min_elo", type=int, default=2200)
    parser.add_argument("--sample_every", type=int, default=5,
                        help="Sample a position every N plies within each game")
    parser.add_argument("--datasets", nargs="+", default=[
        "malcouffe/lichess-standard-rated-2025-09-uci",
    ])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    # Target distribution: 50% opening, 30% middlegame, 20% endgame
    targets = {
        "opening": int(args.num_positions * 0.5),
        "middlegame": int(args.num_positions * 0.3),
        "endgame": args.num_positions - int(args.num_positions * 0.5) - int(args.num_positions * 0.3),
    }
    collected: dict[str, list[dict]] = {"opening": [], "middlegame": [], "endgame": []}

    print(f"Target: {targets}")
    print(f"Loading datasets: {args.datasets}")
    print(f"Filtering ELO >= {args.min_elo}")

    for ds_name in args.datasets:
        # Check if we already have enough
        if all(len(collected[p]) >= targets[p] for p in targets):
            break

        print(f"  Loading {ds_name} ...")
        ds = load_dataset(ds_name, split="train", streaming=True)

        for example in ds:
            # Filter by ELO
            white_elo = example.get("white_elo", 0)
            black_elo = example.get("black_elo", 0)
            if white_elo < args.min_elo or black_elo < args.min_elo:
                continue

            moves_uci = example.get("moves_uci") or example.get("moves", "")
            positions = extract_positions_from_game(
                moves_uci, white_elo, black_elo,
                sample_every=args.sample_every,
            )

            for pos in positions:
                phase = pos["phase"]
                if len(collected[phase]) < targets[phase]:
                    collected[phase].append(pos)

            # Progress
            total = sum(len(v) for v in collected.values())
            if total % 1000 == 0 and total > 0:
                status = ", ".join(f"{p}: {len(collected[p])}/{targets[p]}" for p in targets)
                print(f"  Collected {total}/{args.num_positions} ({status})")

            if all(len(collected[p]) >= targets[p] for p in targets):
                break

    # Combine, shuffle
    all_positions = []
    for phase in collected:
        all_positions.extend(collected[phase][:targets[phase]])

    random.shuffle(all_positions)

    # Write to temp file then upload to HF Hub
    local_path = os.path.join(tempfile.gettempdir(), "test_positions.jsonl")
    with open(local_path, "w") as f:
        for pos in all_positions:
            f.write(json.dumps(pos, ensure_ascii=False) + "\n")

    total = len(all_positions)
    phase_counts = {}
    for pos in all_positions:
        phase_counts[pos["phase"]] = phase_counts.get(pos["phase"], 0) + 1

    print(f"\nGenerated {total} positions")
    for phase, count in sorted(phase_counts.items()):
        print(f"  {phase}: {count}")

    print(f"\nUploading to {args.output} ...")
    hf_dataset = Dataset.from_json(local_path)
    hf_dataset.push_to_hub(args.output)
    print(f"Uploaded to https://huggingface.co/datasets/{args.output}")
    os.remove(local_path)


if __name__ == "__main__":
    main()
