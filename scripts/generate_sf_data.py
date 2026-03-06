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
from multiprocessing import Pool

import chess
import chess.engine
from datasets import Dataset, load_dataset

from chessgpt import UCITokenizer

# ---------------------------------------------------------------------------
# Per-worker globals — initialised once per process, never re-serialised
# ---------------------------------------------------------------------------

_worker_engine: chess.engine.SimpleEngine | None = None
_worker_move_to_id: dict[str, int] = {}
_worker_bos_id: int = 0
_worker_sample_every: int = 8
_worker_top_n: int = 5
_worker_sf_depth: int = 12
_worker_max_seq_len: int = 256


def _init_worker(sf_path: str, move_to_id: dict, bos_id: int,
                 sample_every: int, top_n: int, sf_depth: int, max_seq_len: int):
    """Initialise one Stockfish engine + all constant data per worker process."""
    import signal as _signal
    _signal.signal(_signal.SIGINT, _signal.SIG_IGN)   # let the main process handle Ctrl-C

    global _worker_engine, _worker_move_to_id, _worker_bos_id
    global _worker_sample_every, _worker_top_n, _worker_sf_depth, _worker_max_seq_len
    _worker_engine = chess.engine.SimpleEngine.popen_uci(sf_path)
    _worker_engine.configure({"Threads": 1, "Hash": 64})
    _worker_move_to_id = move_to_id
    _worker_bos_id = bos_id
    _worker_sample_every = sample_every
    _worker_top_n = top_n
    _worker_sf_depth = sf_depth
    _worker_max_seq_len = max_seq_len


MATE_SCORE = 10_000


def _score_to_cp(score) -> int:
    if score.is_mate():
        return MATE_SCORE if score.mate() > 0 else -MATE_SCORE
    return score.score()


def annotate_game(moves_uci: list[str]) -> tuple[dict | None, int]:
    """Annotate multiple sampled positions from a single game.

    Runs in a worker process with its own Stockfish engine.
    All constant parameters come from process-local globals set by _init_worker,
    so nothing beyond the move list is serialised per task.
    """
    engine = _worker_engine
    move_to_id = _worker_move_to_id
    bos_id = _worker_bos_id
    sample_every = _worker_sample_every
    top_n = _worker_top_n
    sf_depth = _worker_sf_depth
    max_seq_len = _worker_max_seq_len

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
    last_move_ply = len(moves_uci) - 1

    for ply, mv_uci in enumerate(moves_uci):
        if ply >= 4 and ply < last_move_ply and (ply - 4) % sample_every == 0:
            if not board.is_game_over():
                token_pos = ply + 1 - offset
                if 0 <= token_pos < len(tokens):
                    # Collect legal moves once — avoids three separate generator passes
                    legal_moves = list(board.legal_moves)
                    n = min(top_n, len(legal_moves))
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

                        legal_move_ids = [
                            tid for lm in legal_moves
                            if (tid := move_to_id.get(lm.uci())) is not None
                        ]

                        if sf_targets and legal_move_ids:
                            annotations.append({
                                "position_index": token_pos,
                                "sf_targets": sf_targets,
                                "legal_move_ids": legal_move_ids,
                            })

        board.push_uci(mv_uci)

    if not annotations:
        return None, 0

    return {"tokens": tokens, "annotations": annotations}, len(annotations)


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
    # Extract serializable data — passed once per worker via initializer, not per task
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

    _shutdown = False

    def _handle_signal(signum, frame):
        nonlocal _shutdown
        if not _shutdown:
            _shutdown = True
            print(f"\n  [{signal.Signals(signum).name}] Graceful shutdown ...", flush=True)

    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)

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

    def _game_source():
        """Yield moves_uci lists from randomly interleaved streams (main process)."""
        nonlocal scanned, filtered
        while active and not _shutdown:
            idx = rng.choice(active)
            try:
                example = next(streams[idx])
            except StopIteration:
                active.remove(idx)
                print(
                    f"  Stream {args.datasets[idx]} exhausted, "
                    f"{len(active)} remaining",
                    flush=True,
                )
                continue

            scanned += 1

            white_elo = example.get("white_elo", 0)
            black_elo = example.get("black_elo", 0)
            if white_elo < args.min_elo or black_elo < args.min_elo:
                continue

            moves_uci = (
                example.get("moves_uci") or example.get("moves", "")
            ).strip().split()
            if len(moves_uci) < 10:
                continue

            filtered += 1
            yield moves_uci

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
            f"| {rate:.0f} pos/s "
            f"| ETA {eta / 3600:.1f}h",
            flush=True,
        )

    with open(local_path, "w") as fout:
        pool = Pool(
            processes=args.num_workers,
            initializer=_init_worker,
            initargs=(
                args.stockfish_path, move_to_id, bos_id,
                args.sample_every, args.top_n, args.sf_depth, args.max_seq_len,
            ),
        )
        try:
            # imap_unordered keeps all workers busy with no manual throttling,
            # no polling loop, and no sleep() calls.
            # chunksize=2 amortises IPC cost without over-buffering.
            for record, n_positions in pool.imap_unordered(
                annotate_game, _game_source(), chunksize=2
            ):
                if record is not None:
                    fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                    count += n_positions
                    games += 1

                _log_progress()

                if count >= args.num_positions or _shutdown:
                    break
        finally:
            pool.terminate()
            pool.join()

    _log_progress(force=True)
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
