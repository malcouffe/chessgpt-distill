#!/usr/bin/env python3
"""Fine-tune ChessGPT via Stockfish distillation (KL-divergence on soft targets).

Loads a pre-trained checkpoint and fine-tunes using Stockfish top-N move
distributions as soft targets. This lifts the quality ceiling imposed by
human game data.

Usage:
    python scripts/train_distill.py --config configs/distill_h100.yaml
    python scripts/train_distill.py --config configs/distill_h100.yaml \
        --set optim.learning_rate=1e-5 schedule.max_steps=30000
"""

from __future__ import annotations

import argparse
import math
import os
import signal
import sys
import time
from dataclasses import asdict, dataclass, field

import torch
import torch.nn.functional as F
from datasets import load_dataset as hf_load_dataset
from torch.utils.data import DataLoader, Dataset

from chessgpt import (
    UCITokenizer,
    ChessGPTConfig,
    ChessGPT,
    get_lr,
    set_lr,
    set_seed,
)
from chessgpt.config import (
    ModelConfig,
    OptimConfig,
    ScheduleConfig,
    LoggingConfig,
    HubConfig,
    load_config,
    _build_dataclass,
)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class DistillDataConfig:
    train_path: str = "data/sf_distill.jsonl"
    val_ratio: float = 0.01  # fraction of data for validation
    batch_size: int = 256
    max_seq_len: int = 256
    num_workers: int = 4
    prefetch_factor: int = 2
    sf_temperature: float = 150.0  # temperature for softening SF scores
    ce_alpha: float = 0.0  # weight for CE loss on SF top-1 move (0 = KL only)
    value_alpha: float = 0.5  # weight for value MSE loss
    seed: int = 42


@dataclass
class DistillConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DistillDataConfig = field(default_factory=DistillDataConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    schedule: ScheduleConfig = field(default_factory=ScheduleConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    hub: HubConfig = field(default_factory=HubConfig)
    device: str = "auto"
    use_amp: bool = True
    amp_dtype: str = "bf16"
    compile: bool = False
    compile_mode: str = "default"
    gradient_checkpointing: bool = False
    pretrained_checkpoint: str = ""  # path to pre-trained ChessGPT checkpoint
    resume: str = ""  # path to distillation checkpoint to resume from
    early_stopping_patience: int = 0  # 0 = disabled, N = stop after N evals without improvement


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class StockfishDistillDataset(Dataset):
    """Dataset of game fragments with multiple Stockfish-annotated positions.

    Each sample is loaded from JSONL with format:
    {
        "tokens": [1, 244, 385, ...],
        "annotations": [
            {
                "position_index": 3,
                "sf_targets": [{"token_id": 100, "score_cp": 50}, ...],
                "legal_move_ids": [100, 234, 567, ...]
            },
            ...
        ]
    }
    """

    def __init__(self, data: list[dict], max_seq_len: int, vocab_size: int, sf_temperature: float):
        self.data = data
        self.max_seq_len = max_seq_len
        self.vocab_size = vocab_size
        self.sf_temperature = sf_temperature

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        tokens = item["tokens"]

        # Crop to max_seq_len (keep most recent)
        if len(tokens) > self.max_seq_len:
            offset = len(tokens) - self.max_seq_len
            tokens = tokens[-self.max_seq_len:]
        else:
            offset = 0

        seq_len = len(tokens)

        # Pad to max_seq_len
        pad_len = self.max_seq_len - seq_len
        input_ids = tokens + [0] * pad_len  # 0 = PAD

        # Build per-position targets
        position_indices = []
        target_probs_list = []
        legal_masks_list = []
        value_targets_list = []

        for ann in item["annotations"]:
            pos_idx = ann["position_index"] - offset
            if pos_idx < 0 or pos_idx >= seq_len:
                continue  # annotation was cropped out

            sf_targets = ann["sf_targets"]
            legal_ids = ann["legal_move_ids"]

            # Build target distribution over legal moves only
            target_logits = torch.full((self.vocab_size,), float("-inf"))

            # Baseline for legal moves not scored by SF
            min_sf_score = min(e["score_cp"] for e in sf_targets)
            baseline = (min_sf_score - 300) / max(self.sf_temperature, 1e-6)
            for lid in legal_ids:
                target_logits[lid] = baseline

            # Override with actual SF scores
            for entry in sf_targets:
                target_logits[entry["token_id"]] = entry["score_cp"] / max(self.sf_temperature, 1e-6)

            # Legal mask
            legal_mask = torch.zeros(self.vocab_size, dtype=torch.bool)
            for lid in legal_ids:
                legal_mask[lid] = True

            # Softmax over legal moves only (illegal stays -inf → 0 probability)
            target_probs = F.softmax(target_logits, dim=0)

            # Value target: SF's best move score normalised to [-1, +1] (white's perspective)
            # sf_targets[0] is always SF's top-1 move (best for side-to-move)
            best_cp = sf_targets[0]["score_cp"]
            value_targets_list.append(math.tanh(best_cp / 400.0))

            position_indices.append(pos_idx)
            target_probs_list.append(target_probs)
            legal_masks_list.append(legal_mask)

        if not position_indices:
            # All annotations cropped — return dummy that collate will skip
            return {
                "input_ids": torch.tensor(input_ids, dtype=torch.long),
                "num_positions": 0,
                "position_indices": torch.zeros(1, dtype=torch.long),
                "target_probs": torch.zeros(1, self.vocab_size),
                "legal_masks": torch.zeros(1, self.vocab_size, dtype=torch.bool),
                "value_targets": torch.zeros(1),
            }

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "num_positions": len(position_indices),
            "position_indices": torch.tensor(position_indices, dtype=torch.long),
            "target_probs": torch.stack(target_probs_list),   # (P, V)
            "legal_masks": torch.stack(legal_masks_list),      # (P, V)
            "value_targets": torch.tensor(value_targets_list, dtype=torch.float32),  # (P,)
        }


def distill_collate_fn(batch: list[dict]) -> dict:
    """Collate variable-length multi-position samples.

    Flattens all annotated positions across the batch for efficient gather.
    """
    input_ids = torch.stack([b["input_ids"] for b in batch])  # (B, S)

    all_batch_indices = []
    all_position_indices = []
    all_target_probs = []
    all_legal_masks = []
    all_value_targets = []

    for b_idx, b in enumerate(batch):
        n = b["num_positions"]
        if n == 0:
            continue
        all_batch_indices.extend([b_idx] * n)
        all_position_indices.append(b["position_indices"])
        all_target_probs.append(b["target_probs"])
        all_legal_masks.append(b["legal_masks"])
        all_value_targets.append(b["value_targets"])

    if not all_position_indices:
        V = batch[0]["target_probs"].shape[-1]
        return {
            "input_ids": input_ids,
            "batch_indices": torch.zeros(0, dtype=torch.long),
            "position_indices": torch.zeros(0, dtype=torch.long),
            "target_probs": torch.zeros(0, V),
            "legal_masks": torch.zeros(0, V, dtype=torch.bool),
            "value_targets": torch.zeros(0),
            "total_positions": 0,
        }

    return {
        "input_ids": input_ids,
        "batch_indices": torch.tensor(all_batch_indices, dtype=torch.long),
        "position_indices": torch.cat(all_position_indices),
        "target_probs": torch.cat(all_target_probs),
        "legal_masks": torch.cat(all_legal_masks),
        "value_targets": torch.cat(all_value_targets),
        "total_positions": len(all_batch_indices),
    }


def load_distill_data(path: str):
    """Load distillation data from a HF Hub dataset (memory-mapped Arrow)."""
    return hf_load_dataset(path, split="train")


# ---------------------------------------------------------------------------
# Checkpoint loading (from pre-trained ChessGPT)
# ---------------------------------------------------------------------------


def load_pretrained(checkpoint_path: str, device: torch.device,
                    model_cfg_override: ModelConfig | None = None,
                    dropout_override: float | None = None):
    """Load a pre-trained ChessGPT model from checkpoint (.pt), safetensors, or HF Hub repo."""
    tokenizer = UCITokenizer()

    path = checkpoint_path

    # HF Hub repo ID (e.g. "malcouffe/chessgpt") → download locally
    if "/" in path and not os.path.exists(path):
        from huggingface_hub import snapshot_download
        print(f"  Downloading from HF Hub: {path} ...")
        path = snapshot_download(path)

    # Resolve directory → find .safetensors or .pt inside
    if os.path.isdir(path):
        for name in ("model.safetensors", "checkpoint.pt", "latest.pt"):
            candidate = os.path.join(path, name)
            if os.path.exists(candidate):
                path = candidate
                break

    if path.endswith(".safetensors"):
        from safetensors.torch import load_file
        state_dict = load_file(path, device=str(device))
        # Strip "model." prefix if present (HF Hub saves wrap in a parent module)
        if any(k.startswith("model.") for k in state_dict):
            state_dict = {k.removeprefix("model."): v for k, v in state_dict.items()}
        # Remap HF naming conventions to ChessGPT naming
        key_map = {"embed_tokens.weight": "token_emb.weight"}
        state_dict = {key_map.get(k, k): v for k, v in state_dict.items()}
        # head.weight is tied to token_emb.weight
        if "head.weight" not in state_dict and "token_emb.weight" in state_dict:
            state_dict["head.weight"] = state_dict["token_emb.weight"]
        step = 0

        # Use config from yaml (model_cfg_override) since safetensors has no config
        if model_cfg_override is None:
            print("Error: model config must be provided in yaml when loading safetensors")
            sys.exit(1)

        dropout = model_cfg_override.dropout
        if dropout_override is not None:
            dropout = dropout_override

        model_cfg = ChessGPTConfig(
            vocab_size=tokenizer.vocab_size,
            d_model=model_cfg_override.d_model,
            n_layers=model_cfg_override.n_layers,
            n_heads=model_cfg_override.n_heads,
            d_ff=model_cfg_override.d_ff,
            max_seq_len=model_cfg_override.max_seq_len,
            dropout=dropout,
        )

        model = ChessGPT(model_cfg).to(device)
        info = model.load_state_dict(state_dict, strict=False)
        if info.missing_keys:
            print(f"  Missing keys (randomly initialised): {info.missing_keys}")
        if info.unexpected_keys:
            print(f"  Unexpected keys (ignored): {info.unexpected_keys}")

    else:
        ckpt = torch.load(path, map_location=device, weights_only=False)

        if "config" in ckpt and "model" in ckpt.get("config", {}):
            model_d = ckpt["config"]["model"]
        else:
            model_d = ckpt.get("train_config", {})

        dropout = float(model_d.get("dropout", 0.0))
        if dropout_override is not None:
            dropout = dropout_override

        model_cfg = ChessGPTConfig(
            vocab_size=tokenizer.vocab_size,
            d_model=int(model_d.get("d_model", 256)),
            n_layers=int(model_d.get("n_layers", 8)),
            n_heads=int(model_d.get("n_heads", 8)),
            d_ff=int(model_d.get("d_ff", 1024)),
            max_seq_len=int(model_d.get("max_seq_len", 256)),
            dropout=dropout,
        )

        model = ChessGPT(model_cfg).to(device)
        info = model.load_state_dict(ckpt["model_state_dict"], strict=False)
        if info.missing_keys:
            print(f"  Missing keys (randomly initialised): {info.missing_keys}")
        if info.unexpected_keys:
            print(f"  Unexpected keys (ignored): {info.unexpected_keys}")
        step = ckpt.get("step", 0)

    print(f"  Loaded pre-trained model from step {step}, params: {model.count_parameters():,}")
    if dropout_override is not None:
        print(f"  Dropout overridden to {dropout_override}")

    return model, tokenizer, model_cfg


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------


def _make_checkpoint(model, optimizer, scaler, step, cfg, best_val_loss, tokens_seen):
    return {
        "step": step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "best_val_loss": best_val_loss,
        "tokens_seen": tokens_seen,
        "config": asdict(cfg),
    }


def save_checkpoint(model, optimizer, scaler, step, cfg, best_val_loss, tokens_seen,
                    filename: str = "last_distill.pt"):
    os.makedirs(cfg.logging.checkpoint_dir, exist_ok=True)
    path = os.path.join(cfg.logging.checkpoint_dir, filename)

    ckpt = _make_checkpoint(model, optimizer, scaler, step, cfg, best_val_loss, tokens_seen)
    torch.save(ckpt, path)

    print(f"  Checkpoint saved: {path} (step {step})")


def push_to_hub(cfg: DistillConfig, step: int, is_best: bool = False):
    """Push checkpoint(s) to HuggingFace Hub.

    When is_best=True, only uploads best_distill.pt.
    When is_best=False (final save), only uploads last_distill.pt.
    """
    try:
        from huggingface_hub import HfApi, upload_file

        hub = cfg.hub
        if not hub.repo_id:
            return

        api = HfApi()
        api.create_repo(repo_id=hub.repo_id, repo_type="model", exist_ok=True)

        if is_best:
            # Upload best checkpoint only
            best_path = os.path.join(cfg.logging.checkpoint_dir, "best_distill.pt")
            if os.path.isfile(best_path):
                print(f"  [hub] Uploading best_distill.pt to {hub.repo_id} ...")
                upload_file(path_or_fileobj=best_path, path_in_repo="best_distill.pt", repo_id=hub.repo_id)
        else:
            # Upload last checkpoint (final save / resume)
            last_path = os.path.join(cfg.logging.checkpoint_dir, "last_distill.pt")
            if os.path.isfile(last_path):
                print(f"  [hub] Uploading last_distill.pt to {hub.repo_id} ...")
                upload_file(path_or_fileobj=last_path, path_in_repo="last_distill.pt", repo_id=hub.repo_id)

        print(f"  [hub] Done: https://huggingface.co/{hub.repo_id}")

    except Exception as e:
        print(f"  [hub] WARNING: push failed (training continues): {e}")


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


@torch.no_grad()
def evaluate(model, val_loader, device, amp_enabled, amp_dtype):
    model.eval()
    total_policy_loss = 0.0
    total_value_loss = 0.0
    total_positions = 0

    for batch in val_loader:
        n_positions = batch["total_positions"]
        if n_positions == 0:
            continue

        input_ids = batch["input_ids"].to(device)
        batch_indices = batch["batch_indices"].to(device)
        position_indices = batch["position_indices"].to(device)
        target_probs = batch["target_probs"].to(device)
        legal_masks = batch["legal_masks"].to(device)
        value_targets = batch["value_targets"].to(device)

        with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
            logits, values, _ = model(input_ids)

        selected_logits = logits[batch_indices, position_indices]  # (P, V)
        selected_logits = selected_logits.masked_fill(~legal_masks, float("-inf"))
        log_probs = F.log_softmax(selected_logits, dim=-1)
        # Compute KL only over legal moves to avoid 0*(log(0)-(-inf))=nan
        lp = log_probs[legal_masks]
        tp = target_probs[legal_masks]
        policy_loss = F.kl_div(lp, tp, reduction="sum")

        selected_values = values[batch_indices, position_indices]
        value_loss = F.mse_loss(selected_values, value_targets, reduction="sum")

        total_policy_loss += policy_loss.item()
        total_value_loss += value_loss.item()
        total_positions += n_positions

    model.train()
    avg_policy = total_policy_loss / max(total_positions, 1)
    avg_value = total_value_loss / max(total_positions, 1)
    return avg_policy, avg_value


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def train(cfg: DistillConfig):
    # H100 optimizations
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")
        torch.backends.cudnn.benchmark = True

    from chessgpt import resolve_device
    device = resolve_device(cfg.device)
    set_seed(cfg.data.seed)
    print(f"Device: {device}")

    # Load pre-trained model
    if not cfg.pretrained_checkpoint:
        print("Error: pretrained_checkpoint is required")
        sys.exit(1)

    print(f"Loading pre-trained model: {cfg.pretrained_checkpoint}")
    dropout = cfg.model.dropout if cfg.model.dropout > 0.0 else None
    model, tokenizer, model_cfg = load_pretrained(
        cfg.pretrained_checkpoint, device,
        model_cfg_override=cfg.model, dropout_override=dropout,
    )
    model.gradient_checkpointing = cfg.gradient_checkpointing

    if cfg.compile:
        print("Compiling model with torch.compile...")
        model = torch.compile(model, mode=cfg.compile_mode)

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.optim.learning_rate,
        betas=cfg.optim.betas,
        weight_decay=cfg.optim.weight_decay,
    )

    amp_enabled = cfg.use_amp and device.type == "cuda"
    amp_dtype = torch.bfloat16 if cfg.amp_dtype == "bf16" else torch.float16
    use_scaler = amp_enabled and amp_dtype == torch.float16
    scaler = torch.amp.GradScaler(enabled=use_scaler)

    # Load data
    print(f"Loading distillation data: {cfg.data.train_path}")
    all_data = load_distill_data(cfg.data.train_path)
    print(f"  Total game records: {len(all_data):,}")

    # Shuffle before train/val split to ensure representative validation set
    all_data = all_data.shuffle(seed=cfg.data.seed)

    # Train/val split
    val_size = max(1, int(len(all_data) * cfg.data.val_ratio))
    train_data = all_data.select(range(val_size, len(all_data)))
    val_data = all_data.select(range(val_size))
    print(f"  Train: {len(train_data):,}, Val: {len(val_data):,}")

    train_dataset = StockfishDistillDataset(
        train_data, cfg.data.max_seq_len, tokenizer.vocab_size, cfg.data.sf_temperature,
    )
    val_dataset = StockfishDistillDataset(
        val_data, cfg.data.max_seq_len, tokenizer.vocab_size, cfg.data.sf_temperature,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.data.batch_size,
        shuffle=True,
        num_workers=cfg.data.num_workers,
        prefetch_factor=cfg.data.prefetch_factor if cfg.data.num_workers > 0 else None,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
        collate_fn=distill_collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.data.batch_size,
        shuffle=False,
        num_workers=cfg.data.num_workers,
        prefetch_factor=cfg.data.prefetch_factor if cfg.data.num_workers > 0 else None,
        pin_memory=(device.type == "cuda"),
        collate_fn=distill_collate_fn,
    )

    # Resume from distillation checkpoint
    start_step = 0
    best_val_loss = float("inf")
    tokens_seen = 0

    if cfg.resume:
        ckpt = torch.load(cfg.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if "scaler_state_dict" in ckpt:
            scaler.load_state_dict(ckpt["scaler_state_dict"])
        start_step = ckpt.get("step", 0)
        best_val_loss = ckpt.get("best_val_loss", float("inf"))
        tokens_seen = ckpt.get("tokens_seen", 0)
        print(f"  Resumed distillation from step {start_step}")

    model.train()
    step = start_step
    running_loss = 0.0
    running_policy_loss = 0.0
    running_value_loss = 0.0
    evals_without_improvement = 0
    last_hub_push_step = 0
    t0 = time.time()

    print(f"\nDistillation training for {cfg.schedule.max_steps} steps")
    print(f"Batch size: {cfg.data.batch_size} x {cfg.optim.grad_accum_steps} accum = {cfg.data.batch_size * cfg.optim.grad_accum_steps} effective")
    print(f"LR: {cfg.optim.learning_rate}, SF temperature: {cfg.data.sf_temperature}")
    if cfg.data.ce_alpha > 0:
        print(f"Combined loss: {1 - cfg.data.ce_alpha:.0%} KL + {cfg.data.ce_alpha:.0%} CE")
    if cfg.early_stopping_patience > 0:
        print(f"Early stopping: patience {cfg.early_stopping_patience} evals")
    print("-" * 60)

    # Graceful shutdown
    _shutdown_requested = False

    def _handle_signal(signum, frame):
        nonlocal _shutdown_requested
        if _shutdown_requested:
            return
        _shutdown_requested = True
        print(f"\n  [{signal.Signals(signum).name}] Graceful shutdown ...")

    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)

    accum_steps = cfg.optim.grad_accum_steps
    epoch = 0
    micro_step = 0
    done = False

    while not done:
        epoch += 1

        for batch in train_loader:
            if step >= cfg.schedule.max_steps or _shutdown_requested:
                done = True
                break

            total_positions = batch["total_positions"]
            if total_positions == 0:
                continue

            input_ids = batch["input_ids"].to(device)
            batch_indices = batch["batch_indices"].to(device)
            position_indices = batch["position_indices"].to(device)
            target_probs = batch["target_probs"].to(device)
            legal_masks = batch["legal_masks"].to(device)
            value_targets = batch["value_targets"].to(device)

            with torch.amp.autocast(
                device_type=device.type, dtype=amp_dtype, enabled=amp_enabled,
            ):
                logits, values, _ = model(input_ids)

                # Gather logits at all annotated positions
                selected_logits = logits[batch_indices, position_indices]  # (P_total, V)
                selected_logits = selected_logits.masked_fill(~legal_masks, float("-inf"))

                log_probs = F.log_softmax(selected_logits, dim=-1)
                # Compute KL only over legal moves to avoid 0*(log(0)-(-inf))=nan
                lp = log_probs[legal_masks]
                tp = target_probs[legal_masks]
                kl_loss = F.kl_div(lp, tp, reduction="sum") / total_positions

                # Combined policy loss: KL + CE on SF top-1 move
                if cfg.data.ce_alpha > 0:
                    top1_targets = target_probs.argmax(dim=-1)  # (P,)
                    ce_loss = F.cross_entropy(selected_logits, top1_targets)
                    policy_loss = (1 - cfg.data.ce_alpha) * kl_loss + cfg.data.ce_alpha * ce_loss
                else:
                    policy_loss = kl_loss

                # Value loss
                selected_values = values[batch_indices, position_indices]
                value_loss = F.mse_loss(selected_values, value_targets)

                loss = policy_loss + cfg.data.value_alpha * value_loss

                # Scale loss for gradient accumulation
                loss = loss / accum_steps

            scaler.scale(loss).backward()

            running_loss += loss.item() * accum_steps  # undo scaling for logging
            running_policy_loss += policy_loss.item()
            running_value_loss += value_loss.item()
            tokens_seen += (input_ids != 0).sum().item()  # exclude PAD tokens
            micro_step += 1

            # Only step optimizer after accumulating enough micro-batches
            if micro_step % accum_steps != 0:
                continue

            # LR schedule
            lr = get_lr(
                step, cfg.optim.learning_rate,
                cfg.schedule.warmup_steps, cfg.schedule.max_steps,
                cfg.schedule.min_lr_ratio,
            )
            set_lr(optimizer, lr)

            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), cfg.optim.grad_clip,
            )

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

            step += 1

            # Logging
            if step % cfg.logging.log_every == 0:
                avg_loss = running_loss / cfg.logging.log_every
                avg_ploss = running_policy_loss / cfg.logging.log_every
                avg_vloss = running_value_loss / cfg.logging.log_every
                elapsed = time.time() - t0
                sps = cfg.logging.log_every / elapsed

                vram_str = ""
                if device.type == "cuda":
                    vram_gb = torch.cuda.max_memory_allocated(device) / 1e9
                    vram_str = f" | vram {vram_gb:.1f}G"

                print(
                    f"step {step:>7d} | "
                    f"loss {avg_loss:.4f} | "
                    f"policy {avg_ploss:.4f} | "
                    f"value {avg_vloss:.4f} | "
                    f"lr {lr:.2e} | "
                    f"grad {grad_norm:.2f} | "
                    f"tokens {tokens_seen / 1e9:.2f}B | "
                    f"{sps:.1f} steps/s"
                    f"{vram_str}"
                )
                running_loss = 0.0
                running_policy_loss = 0.0
                running_value_loss = 0.0
                t0 = time.time()

            # Eval
            if step % cfg.logging.eval_every == 0:
                val_policy, val_value = evaluate(model, val_loader, device, amp_enabled, amp_dtype)
                val_loss = val_policy
                print(f"step {step:>7d} | val_policy {val_policy:.4f} | val_value {val_value:.4f}")

                is_new_best = cfg.logging.save_best and val_loss < best_val_loss
                if is_new_best:
                    best_val_loss = val_loss
                    evals_without_improvement = 0
                    save_checkpoint(model, optimizer, scaler, step, cfg, best_val_loss, tokens_seen,
                                    filename="best_distill.pt")
                    # Throttle hub uploads: only push best every save_every steps
                    if cfg.hub.push_to_hub and (step - last_hub_push_step) >= cfg.logging.save_every:
                        push_to_hub(cfg, step, is_best=True)
                        last_hub_push_step = step
                else:
                    evals_without_improvement += 1

                # Early stopping
                if cfg.early_stopping_patience > 0 and evals_without_improvement >= cfg.early_stopping_patience:
                    print(f"  Early stopping triggered: no improvement for {evals_without_improvement} evals (best val_kl_loss={best_val_loss:.4f})")
                    done = True

                t0 = time.time()

            # Periodic save (no push — push only on new best or at end)
            if step % cfg.logging.save_every == 0:
                save_checkpoint(model, optimizer, scaler, step, cfg, best_val_loss, tokens_seen)

    # Final save
    save_checkpoint(model, optimizer, scaler, step, cfg, best_val_loss, tokens_seen)
    if cfg.hub.push_to_hub:
        push_to_hub(cfg, step)
    print(f"\nDistillation complete. Final step: {step}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args():
    parser = argparse.ArgumentParser(description="Stockfish distillation for ChessGPT")
    parser.add_argument("--config", type=str, default="configs/distill_h100.yaml")
    parser.add_argument("--set", nargs="*", default=[], dest="overrides")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg = load_config(args.config, DistillConfig, args.overrides)
    train(cfg)
