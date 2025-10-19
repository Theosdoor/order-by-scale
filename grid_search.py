
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# grid_search.py
import os
import time
import csv
import argparse
import hashlib
import random
from dataclasses import dataclass, asdict
from itertools import product
from typing import Dict, List, Tuple
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from tqdm.auto import tqdm

import fcntl  # POSIX file locking

from transformer_lens import HookedTransformer
from model_utils import configure_runtime, make_model

# -----------------------------
# Device, seeds, utils
# -----------------------------


def pick_device(explicit: str = "auto") -> torch.device:
    if explicit != "auto":
        return torch.device(explicit)
    if torch.cuda.is_available():
        return torch.device("cuda")
    # if torch.backends.mps.is_available():
    #     return torch.device("mps")
    return torch.device("cpu")


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def count_params(model: nn.Module) -> Tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def make_batch(
    batch_size: int,
    n_digits: int,
    list_len: int,
    device: torch.device,
    rng: torch.Generator,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Returns:
      inputs  [B, T]
      targets [B, T]
    where T = 2L + 1 and evaluation uses targets[:, L+1:].
    """
    pad = n_digits
    sep = n_digits + 1
    T = 2 * list_len + 1

    digits = torch.randint(
        0, n_digits, (batch_size, list_len), generator=rng, device=device
    )

    inputs = torch.full((batch_size, T), pad, dtype=torch.long, device=device)
    targets = torch.full((batch_size, T), sep, dtype=torch.long, device=device)

    inputs[:, :list_len] = digits
    inputs[:, list_len] = sep

    targets[:, :list_len] = digits
    targets[:, list_len] = sep
    targets[:, list_len + 1 :] = digits
    return inputs, targets


def make_validation_set(
    n_examples: int,
    n_digits: int,
    list_len: int,
    device: torch.device,
    seed: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator(device=device)
    g.manual_seed(seed)
    X, Y = make_batch(n_examples, n_digits, list_len, device, g)
    return X, Y


@dataclass(frozen=True)
class Config:
    LIST_LEN: int
    N_DIGITS: int
    D_MODEL: int
    N_HEAD: int
    N_LAYERS: int
    USE_LN: bool
    USE_BIAS: bool
    FREEZE_WV: bool
    FREEZE_WO: bool
    WEIGHT_DECAY: float
    run_idx: int


def eval_accuracy(
    model: HookedTransformer,
    val_inputs: torch.Tensor,
    val_targets: torch.Tensor,
    list_len: int,
    batch_size: int = 1024,
) -> float:
    model.eval()
    device = next(model.parameters()).device
    hits = 0
    tots = 0
    with torch.no_grad():
        for i in range(0, val_inputs.size(0), batch_size):
            xb = val_inputs[i : i + batch_size].to(device)
            yb = val_targets[i : i + batch_size].to(device)
            logits = model(xb)[:, list_len + 1 :, :]  # [B, L, V]
            preds = logits.argmax(dim=-1)
            gold = yb[:, list_len + 1 :]
            hits += (preds == gold).sum().item()
            tots += preds.numel()
    return hits / max(1, tots)


def build_model_from_config(cfg: Config, device: torch.device) -> HookedTransformer:
    assert cfg.D_MODEL % cfg.N_HEAD == 0, "d_model must be divisible by n_heads"
    vocab = cfg.N_DIGITS + 2
    seq_len = 2 * cfg.LIST_LEN + 1
    # Configure shared runtime and build via model_utils
    configure_runtime(list_len=cfg.LIST_LEN, seq_len=seq_len, vocab=vocab, device=device)
    model = make_model(
        n_layers=cfg.N_LAYERS,
        n_heads=cfg.N_HEAD,
        d_model=cfg.D_MODEL,
        ln=cfg.USE_LN,
        use_bias=cfg.USE_BIAS,
        freeze_wv=cfg.FREEZE_WV,
        freeze_wo=cfg.FREEZE_WO,
        device=device,
    )
    return model


def train_and_return_model(
    cfg: Config,
    device: torch.device,
    train_steps: int,
    batch_size: int,
    eval_every: int,
    early_stop_acc: float,
    val_size: int,
    base_seed: int,
    lr: float,
    eval_batch_size: int,
    grad_clip: float,
) -> Tuple[HookedTransformer, Dict]:
    seed_material = (str(asdict(cfg)) + str(base_seed)).encode("utf-8")
    seed = int.from_bytes(hashlib.sha256(seed_material).digest()[:4], "big")
    set_global_seed(seed + cfg.run_idx)

    model = build_model_from_config(cfg, device)
    total_params, trainable_params = count_params(model)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=cfg.WEIGHT_DECAY)
    ce = nn.CrossEntropyLoss()

    val_inputs, val_targets = make_validation_set(
        n_examples=val_size,
        n_digits=cfg.N_DIGITS,
        list_len=cfg.LIST_LEN,
        device=device,
        seed=seed + 123456,
    )

    model.train()
    g = torch.Generator(device=device).manual_seed(seed + 999)

    last_eval = 0.0
    steps_run = 0
    start = time.time()

    pbar = tqdm(total=train_steps, desc="train", leave=False, ncols=120)
    for step in range(1, train_steps + 1):
        xb, yb = make_batch(batch_size, cfg.N_DIGITS, cfg.LIST_LEN, device, g)
        logits = model(xb)[:, cfg.LIST_LEN + 1 :]
        gold = yb[:, cfg.LIST_LEN + 1 :]

        loss = ce(logits.reshape(-1, logits.size(-1)), gold.reshape(-1))
        loss.backward()
        if grad_clip and grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        opt.step()
        opt.zero_grad()

        if step % eval_every == 0 or step == train_steps:
            last_eval = eval_accuracy(
                model, val_inputs, val_targets, cfg.LIST_LEN, batch_size=eval_batch_size
            )
            with torch.no_grad():
                preds = logits.argmax(dim=-1)
                train_acc = (preds == gold).float().mean().item()

            pbar.set_postfix(
                {"val_acc": f"{last_eval:.4f}", "train_acc": f"{train_acc:.4f}"}
            )
            if last_eval >= early_stop_acc:
                steps_run = step
                pbar.update(train_steps - step + 1)
                break

        pbar.update(1)
        steps_run = step
    pbar.close()

    train_time_sec = time.time() - start
    if last_eval == 0.0:
        last_eval = eval_accuracy(
            model, val_inputs, val_targets, cfg.LIST_LEN, batch_size=eval_batch_size
        )

    metrics = {
        "val_acc": last_eval,
        "steps_trained": steps_run,
        "train_time_sec": train_time_sec,
        "params_total": total_params,
        "params_trainable": trainable_params,
        "val_examples": val_inputs.size(0),
        "batch_size": batch_size,
    }
    return model, metrics


def train_one(
    cfg: Config,
    device: torch.device,
    train_steps: int,
    batch_size: int,
    eval_every: int,
    early_stop_acc: float,
    val_size: int,
    base_seed: int,
    lr: float,
    eval_batch_size: int,
    grad_clip: float,
) -> Dict:
    _model, metrics = train_and_return_model(
        cfg=cfg,
        device=device,
        train_steps=train_steps,
        batch_size=batch_size,
        eval_every=eval_every,
        early_stop_acc=early_stop_acc,
        val_size=val_size,
        base_seed=base_seed,
        lr=lr,
        eval_batch_size=eval_batch_size,
        grad_clip=grad_clip,
    )
    return metrics


# -----------------------------
# Grid (exact values requested)
# -----------------------------


def build_grid() -> List[Config]:
    LIST_LEN = list(range(1, 11))
    N_DIGITS = [100]
    D_MODEL = [64]
    N_HEAD = [1]
    N_LAYERS = list(range(1, 11))
    USE_LN = [False]
    USE_BIAS = [False]
    FREEZE_WV = [True]
    FREEZE_WO = [True]
    WEIGHT_DECAY = [0.01]
    RUNS = [0, 1, 2]

    cfgs: List[Config] = []
    for L, ND, DM, NH, NL, LN, UB, FWV, FWO, WD, R in product(
        LIST_LEN,
        N_DIGITS,
        D_MODEL,
        N_HEAD,
        N_LAYERS,
        USE_LN,
        USE_BIAS,
        FREEZE_WV,
        FREEZE_WO,
        WEIGHT_DECAY,
        RUNS,
    ):
        if DM % NH != 0:
            continue
        cfgs.append(
            Config(
                LIST_LEN=L,
                N_DIGITS=ND,
                D_MODEL=DM,
                N_HEAD=NH,
                N_LAYERS=NL,
                USE_LN=LN,
                USE_BIAS=UB,
                FREEZE_WV=FWV,
                FREEZE_WO=FWO,
                WEIGHT_DECAY=WD,
                run_idx=R,
            )
        )
    return cfgs


# -----------------------------
# Sharding helpers
# -----------------------------


def detect_shard_from_env() -> Tuple[int, int]:
    """Infer (shard_index, shard_count) from SLURM variables if present."""
    env = os.environ
    sid = env.get("SLURM_ARRAY_TASK_ID")
    if sid is None:
        return 0, 1
    # Prefer explicit count if available
    scount = env.get("SLURM_ARRAY_TASK_COUNT")
    if scount is not None:
        return int(sid), int(scount)
    # Fall back to min/max/step
    smin = env.get("SLURM_ARRAY_TASK_MIN")
    smax = env.get("SLURM_ARRAY_TASK_MAX")
    sstep = env.get("SLURM_ARRAY_TASK_STEP", "1")
    if smin is not None and smax is not None:
        amin = int(smin)
        amax = int(smax)
        step = int(sstep)
        count = (amax - amin) // step + 1
        idx = (int(sid) - amin) // step
        return idx, count
    return 0, 1


def shard_items(items: List, shard_index: int, shard_count: int) -> List:
    if shard_count <= 1:
        return items
    return [x for i, x in enumerate(items) if (i % shard_count) == shard_index]


def ensure_csv_header(path: str, fields: List[str]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a+", newline="") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        f.seek(0, os.SEEK_END)
        empty = f.tell() == 0
        if empty:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            f.flush()
            os.fsync(f.fileno())
        fcntl.flock(f, fcntl.LOCK_UN)


def append_row_locked(path: str, fields: List[str], row: Dict) -> None:
    with open(path, "a", newline="") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writerow(row)
        f.flush()
        os.fsync(f.fileno())
        fcntl.flock(f, fcntl.LOCK_UN)


# -----------------------------
# CLI and main
# -----------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Copy-Task Transformer Grid Search")
    p.add_argument("--output", type=str, default="results.csv", help="CSV file path")
    p.add_argument("--device", type=str, default="auto", help="cuda|mps|cpu|auto")
    p.add_argument("--seed", type=int, default=42, help="base seed")
    p.add_argument("--train-steps", type=int, default=50000, help="max steps per run")
    p.add_argument("--eval-every", type=int, default=2500, help="validation frequency")
    p.add_argument(
        "--early-stop-acc", type=float, default=0.99, help="early stop threshold"
    )
    p.add_argument("--val-size", type=int, default=8192, help="validation set size")
    p.add_argument("--batch-size", type=int, default=1024, help="training batch size")
    p.add_argument("--lr", type=float, default=1e-3, help="AdamW learning rate")
    p.add_argument("--eval-batch-size", type=int, default=1024, help="evaluation batch size")
    p.add_argument("--grad-clip", type=float, default=0.0, help="max grad-norm clipping (0 disables)")
    p.add_argument("--skip-existing", action="store_true", help="skip configs already present in output CSV")
    # New: sharding (overridden by SLURM env if not set)
    p.add_argument("--shard-index", type=int, default=None, help="0-based shard index")
    p.add_argument("--shard-count", type=int, default=None, help="total shards")
    return p.parse_args()


def main():
    args = parse_args()
    device = pick_device(args.device)
    set_global_seed(args.seed)

    # Build and shard the config list
    full_cfgs = build_grid()
    if args.shard_index is None or args.shard_count is None:
        env_idx, env_cnt = detect_shard_from_env()
        shard_index = env_idx if args.shard_index is None else args.shard_index
        shard_count = env_cnt if args.shard_count is None else args.shard_count
    else:
        shard_index, shard_count = args.shard_index, args.shard_count

    cfgs = shard_items(full_cfgs, shard_index, shard_count)
    total = len(cfgs)

    # CSV fields unchanged
    fields = [
        "LIST_LEN",
        "N_DIGITS",
        "D_MODEL",
        "N_HEAD",
        "N_LAYERS",
        "USE_LN",
        "USE_BIAS",
        "FREEZE_WV",
        "FREEZE_WO",
        "WEIGHT_DECAY",
        "run_idx",
        "val_acc",
        "steps_trained",
        "train_time_sec",
        "params_total",
        "params_trainable",
        "val_examples",
        "batch_size",
        "device",
    ]
    # Header-once init with lock
    ensure_csv_header(args.output, fields)

    # Optionally load existing rows to skip duplicates
    existing_keys = set()
    if args.skip_existing and os.path.exists(args.output):
        try:
            with open(args.output, "r", newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    key = (
                        int(row["LIST_LEN"]),
                        int(row["N_DIGITS"]),
                        int(row["D_MODEL"]),
                        int(row["N_HEAD"]),
                        int(row["N_LAYERS"]),
                        row["USE_LN"] in ("True", "true", "1"),
                        row["USE_BIAS"] in ("True", "true", "1"),
                        row["FREEZE_WV"] in ("True", "true", "1"),
                        row["FREEZE_WO"] in ("True", "true", "1"),
                        float(row["WEIGHT_DECAY"]),
                        int(row["run_idx"]),
                    )
                    existing_keys.add(key)
        except Exception:
            pass

    outer = tqdm(total=total, desc=f"grid[{shard_index}/{shard_count}]", ncols=100)
    for cfg in cfgs:
        if args.skip_existing:
            key = (
                cfg.LIST_LEN,
                cfg.N_DIGITS,
                cfg.D_MODEL,
                cfg.N_HEAD,
                cfg.N_LAYERS,
                cfg.USE_LN,
                cfg.USE_BIAS,
                cfg.FREEZE_WV,
                cfg.FREEZE_WO,
                cfg.WEIGHT_DECAY,
                cfg.run_idx,
            )
            if key in existing_keys:
                outer.update(1)
                continue
        try:
            metrics = train_one(
                cfg=cfg,
                device=device,
                train_steps=args.train_steps,
                batch_size=args.batch_size,
                eval_every=args.eval_every,
                early_stop_acc=args.early_stop_acc,
                val_size=args.val_size,
                base_seed=args.seed,
                lr=args.lr,
                eval_batch_size=args.eval_batch_size,
                grad_clip=args.grad_clip,
            )
        except RuntimeError:
            metrics = {
                "val_acc": float("nan"),
                "steps_trained": 0,
                "train_time_sec": 0.0,
                "params_total": 0,
                "params_trainable": 0,
                "val_examples": args.val_size,
                "batch_size": args.batch_size,
            }

        row = {**asdict(cfg), **metrics, "device": str(device)}
        append_row_locked(args.output, fields, row)
        outer.update(1)
    outer.close()

    print(f"Wrote {total} rows for shard {shard_index}/{shard_count} to {args.output}")


if __name__ == "__main__":
    main()
