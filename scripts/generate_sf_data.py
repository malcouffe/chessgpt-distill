#!/usr/bin/env python3
"""Generate a Stockfish-annotated dataset for distillation training.

For each game, samples multiple positions and records the top-N Stockfish
moves with centipawn evaluations plus all legal move token IDs.
One JSONL record per game with multiple annotated positions.

Uses multiprocessing (not threads) to bypass the GIL and fully utilize
all CPU cores for Stockfish evaluation.

Usage:
    python scripts/generate_sf_data.py \
        --output malcouffe/chessgpt-sf-distill \
        --num_positions 5000000 \
        --sf_depth 12 \
        --num_workers 96 \
        --sample_every 8
"""

from __future__ import annotations

import argparse
import json
import os
import random
import signal
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import chess
import chess.engine
from datasets import Dataset, load_dataset

from chessgpt import UCITokenizer

# ---------------------------------------------------------------------------
# Per-worker Stockfish engine (one per process, no sharing)
# ---------------------------------------------------------------------------

_worker_engine: chess.engine.SimpleEngine | None = None
_worker_sf_path: str = ""


def _init_worker(sf_path: str):
    """Initialize a Stockfish engine in this worker process."""
    global _worker_engine, _worker_sf_path
    _worker_sf_path = sf_path
    _worker_engine = chess.engine.SimpleEngine.popen_uci(sf_path)
    _worker_engine.configure({"Threads": 1, "Hash": 64})


MATE_SCORE = 10_000


def _score_to_cp(score) -> int:
    if score.is_mate():
        return MATE_SCORE if score.mate() > 0 else -MATE_SCORE
    return score.score()


def annotate_game(
    moves_uci: list[str],
    move_to_id: dict[str, int],
    bos_id: int,
    sample_every: int,
    top_n: int,
    sf_depth: int,
    max_seq_len: int,
) -> tuple[dict | None, int]:
    """Annotate multiple sampled positions from a single game.

    Runs in a worker process with its own Stockfish engine.
    """
    global _worker_engine
    engine = _worker_engine

    # Build full token sequence
    tokens = [bos_id]
    for mv in moves_uci:
        tid = move_to_id.get(mv)
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
        if ply >= 4 and ply < len(moves_uci) - 1 and (ply - 4) % sample_every == 0:
            if not board.is_game_over() and board.legal_moves.count() > 0:
                token_pos = ply + 1 - offset
                if 0 <= token_pos < len(tokens):
                    n = min(top_n, board.legal_moves.count())
                    if n > 0:
                        infos = engine.analyse(
                            board, chess.engine.Limit(depth=sf_depth), multipv=n,
                        )
                        sf_targets = []
                        for info in infos:
                            move_uci_str = info["pv"][0].uci()
                            score_cp = _score_to_cp(info["score"].white())
                            tid = move_to_id.get(move_uci_str)
                            if tid is not None:
                                sf_targets.append({
                                    "token_id": tid,
                                    "move": move_uci_str,
                                    "score_cp": score_cp,
                                })

                        legal_move_ids = []
                        for legal_mv in board.legal_moves:
                            tid = move_to_id.get(legal_mv.uci())
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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Generate Stockfish-annotated dataset")
    parser.add_argument("--output", type=str, default="malcouffe/chessgpt-sf-distill",
                        help="HF Hub repo name (e.g. malcouffe/chessgpt-sf-distill)")
    parser.add_argument("--num_positions", type=int, default=5_000_000)
    parser.add_argument("--min_elo", type=int, default=1000)
    parser.add_argument("--sample_every", type=int, default=8,
                        help="Sample a position every N plies within each game")
    parser.add_argument("--top_n", type=int, default=5,
                        help="Number of Stockfish top moves to record")
    parser.add_argument("--sf_depth", type=int, default=12)
    parser.add_argument("--max_seq_len", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=96,
                        help="Number of parallel worker processes (each with its own SF engine)")
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
    # Extract serializable data for worker processes (can't pickle tokenizer)
    move_to_id = dict(tokenizer.move_to_id)
    bos_id = tokenizer.BOS_ID

    local_path = os.path.join(tempfile.gettempdir(), "sf_distill.jsonl")

    print(f"Target: {args.num_positions:,} positions")
    print(f"Stockfish: depth={args.sf_depth}, top_n={args.top_n}")
    print(f"Workers: {args.num_workers} (multiprocessing, 1 SF engine per worker)")
    print(f"Max seq len: {args.max_seq_len}")
    print(f"Datasets: {len(args.datasets)}")
    print(f"HF Hub repo: {args.output}")

    count = 0
    games = 0
    scanned = 0
    filtered = 0
    t0 = time.time()
    last_log = t0

    _shutdown_requested = False

    def _handle_signal(signum, frame):
        nonlocal _shutdown_requested
        if not _shutdown_requested:
            _shutdown_requested = True
            print(f"\n  [{signal.Signals(signum).name}] Graceful shutdown ...", flush=True)

    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)

    with open(local_path, "w") as fout:
        executor = ProcessPoolExecutor(
            max_workers=args.num_workers,
            initializer=_init_worker,
            initargs=(args.stockfish_path,),
        )
        pending_futures: dict = {}

        def _drain_completed():
            nonlocal count, games
            done = [f for f in pending_futures if f.done()]
            for f in done:
                del pending_futures[f]
                record, n_positions = f.result()
                if record is not None:
                    fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                    count += n_positions
                    games += 1

        def _log_progress(force: bool = False):
            nonlocal last_log
            now = time.time()
            if not force and now - last_log < 10:
                return
            last_log = now
            elapsed = now - t0
            rate = count / elapsed if elapsed > 0 else 0
            eta = (args.num_positions - count) / rate if rate > 0 else 0
            print(
                f"  {count:>10,} / {args.num_positions:,} "
                f"({count / args.num_positions:.1%}) "
                f"| {games:,} games "
                f"| scanned {scanned:,} (kept {filtered:,}) "
                f"| in-flight {len(pending_futures)} "
                f"| {rate:.0f} pos/s "
                f"| ETA {eta / 3600:.1f}h",
                flush=True,
            )

        # Open all dataset streams and interleave randomly
        print(f"\nOpening {len(args.datasets)} dataset streams...", flush=True)
        streams = []
        for ds_name in args.datasets:
            print(f"  Loading {ds_name} ...", flush=True)
            ds = load_dataset(ds_name, split="train", streaming=True)
            streams.append(iter(ds))
        print(f"  {len(streams)} streams ready", flush=True)

        rng = random.Random(args.seed)
        active = list(range(len(streams)))

        while active and count < args.num_positions and not _shutdown_requested:
            # Pick a random stream
            idx = rng.choice(active)
            try:
                example = next(streams[idx])
            except StopIteration:
                active.remove(idx)
                print(f"  Stream {args.datasets[idx]} exhausted, {len(active)} remaining", flush=True)
                continue

            scanned += 1

            if scanned % 1000 == 0:
                _drain_completed()
                _log_progress()

            white_elo = example.get("white_elo", 0)
            black_elo = example.get("black_elo", 0)
            if white_elo < args.min_elo or black_elo < args.min_elo:
                continue

            moves_uci = (example.get("moves_uci") or example.get("moves", "")).strip().split()
            if len(moves_uci) < 10:
                continue

            filtered += 1

            fut = executor.submit(
                annotate_game, moves_uci, move_to_id, bos_id,
                sample_every=args.sample_every,
                top_n=args.top_n,
                sf_depth=args.sf_depth,
                max_seq_len=args.max_seq_len,
            )
            pending_futures[fut] = True

            # Throttle: if too many in-flight, drain some
            while len(pending_futures) >= args.num_workers * 3:
                _drain_completed()
                _log_progress()
                if len(pending_futures) >= args.num_workers * 3:
                    time.sleep(0.1)

        # Drain all remaining futures
        print(f"\nDraining {len(pending_futures)} remaining futures...", flush=True)
        for fut in as_completed(pending_futures):
            record, n_positions = fut.result()
            if record is not None:
                fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                count += n_positions
                games += 1
        _log_progress(force=True)

        executor.shutdown(wait=True)

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
