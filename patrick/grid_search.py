#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Copy-Task Transformer Grid Search (fixed grid as requested)

Task:
  Inputs:  [d1..dL, SEP, PAD..PAD]
  Targets: [d1..dL, SEP, d1..dL]
  Outputs are evaluated on the last L positions only.

Model:
  Attention-only transformer with optional LN and biases.
  Supports freezing W_V and W_O to identity slices.

Outputs:
  CSV with one row per (config, run). Includes validation accuracy and metadata.

Example:
  python run_grid.py --output results.csv --train-steps 5000 --early-stop-acc 0.999
"""

import os
import math
import time
import csv
import argparse
import hashlib
import random
from dataclasses import dataclass, asdict
from itertools import product
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from tqdm.auto import tqdm


# -----------------------------
# Device, seeds, utils
# -----------------------------

def pick_device(explicit: str = "auto") -> torch.device:
    if explicit != "auto":
        return torch.device(explicit)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
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


# -----------------------------
# Data generation
# -----------------------------

def build_mask(list_len: int, device: torch.device) -> torch.Tensor:
    """
    Custom mask:
      - Strictly causal (no attending to self or future).
      - Outputs cannot attend to input tokens.
    Shape: [1, 1, T, T]
    """
    T = 2 * list_len + 1
    mask = torch.triu(torch.ones(T, T, device=device) * float("-inf"), diagonal=0)
    mask[0, 0] = 0.0
    if list_len > 0:
        mask[list_len + 1 :, :list_len] = float("-inf")
    return mask.view(1, 1, T, T)


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

    digits = torch.randint(0, n_digits, (batch_size, list_len), generator=rng, device=device)

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


# -----------------------------
# Model
# -----------------------------

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, use_bias: bool):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        k = 1.0 / math.sqrt(d_model)
        self.W_Q = nn.Parameter(torch.empty(n_heads, d_model, self.d_head).uniform_(-k, k))
        self.W_K = nn.Parameter(torch.empty(n_heads, d_model, self.d_head).uniform_(-k, k))
        self.W_V = nn.Parameter(torch.empty(n_heads, d_model, self.d_head).uniform_(-k, k))
        self.W_O = nn.Parameter(torch.empty(n_heads, self.d_head, d_model).uniform_(-k, k))

        if use_bias:
            self.b_Q = nn.Parameter(torch.zeros(n_heads, self.d_head))
            self.b_K = nn.Parameter(torch.zeros(n_heads, self.d_head))
            self.b_V = nn.Parameter(torch.zeros(n_heads, self.d_head))
            self.b_O = nn.Parameter(torch.zeros(n_heads, d_model))
        else:
            self.register_buffer("b_Q", torch.zeros(n_heads, self.d_head), persistent=False)
            self.register_buffer("b_K", torch.zeros(n_heads, self.d_head), persistent=False)
            self.register_buffer("b_V", torch.zeros(n_heads, self.d_head), persistent=False)
            self.register_buffer("b_O", torch.zeros(n_heads, d_model), persistent=False)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        """
        x: [B, T, D]
        returns: [B, T, D]
        """
        B, T, D = x.shape
        H, F = self.n_heads, self.d_head

        q = torch.einsum("btd,hdf->bhtf", x, self.W_Q) + self.b_Q[None, :, None, :]
        k = torch.einsum("btd,hdf->bhtf", x, self.W_K) + self.b_K[None, :, None, :]
        v = torch.einsum("btd,hdf->bhtf", x, self.W_V) + self.b_V[None, :, None, :]

        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(F)
        scores = scores + attn_mask  # [B, H, T, T]
        attn = scores.softmax(dim=-1)
        context = torch.matmul(attn, v)  # [B, H, T, F]

        out = torch.einsum("bhtf,hfd->btd", context, self.W_O) + self.b_O.sum(dim=0)[None, None, :]
        return out


class AttnOnlyBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, use_ln: bool, use_bias: bool):
        super().__init__()
        self.ln = nn.LayerNorm(d_model) if use_ln else nn.Identity()
        self.attn = MultiHeadSelfAttention(d_model, n_heads, use_bias)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        y = self.ln(x)
        y = self.attn(y, attn_mask)
        return x + y


class CopyTransformer(nn.Module):
    def __init__(
        self,
        n_layers: int,
        n_heads: int,
        d_model: int,
        vocab_size: int,
        seq_len: int,
        use_ln: bool,
        use_bias: bool,
    ):
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(seq_len, d_model)
        self.blocks = nn.ModuleList(
            [AttnOnlyBlock(d_model, n_heads, use_ln, use_bias) for _ in range(n_layers)]
        )
        self.unembed = nn.Linear(d_model, vocab_size, bias=use_bias)

        k = 1.0 / math.sqrt(d_model)
        nn.init.uniform_(self.token_embed.weight, -k, k)
        nn.init.uniform_(self.pos_embed.weight, -k, k)
        nn.init.uniform_(self.unembed.weight, -k, k)
        if use_bias and self.unembed.bias is not None:
            nn.init.zeros_(self.unembed.bias)

    def forward(self, x_tokens: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        B, T = x_tokens.shape
        pos = torch.arange(T, device=x_tokens.device).unsqueeze(0).expand(B, T)
        x = self.token_embed(x_tokens) + self.pos_embed(pos)
        for blk in self.blocks:
            x = blk(x, attn_mask)
        return self.unembed(x)


# -----------------------------
# Freezing and bias stripping
# -----------------------------

def strip_biases(model: CopyTransformer) -> None:
    for mod in model.modules():
        if hasattr(mod, "bias") and isinstance(mod.bias, torch.Tensor) and mod.bias is not None:
            with torch.no_grad():
                mod.bias.zero_()
            mod.bias.requires_grad = False


def set_WV_identity_and_freeze(model: CopyTransformer) -> None:
    with torch.no_grad():
        for blk in model.blocks:
            H = blk.attn.n_heads
            D = blk.attn.d_model
            F = blk.attn.d_head
            eye = torch.eye(D, device=blk.attn.W_V.device)[:, :F]  # [D, F]
            blk.attn.W_V.copy_(eye.unsqueeze(0).repeat(H, 1, 1))
            blk.attn.W_V.requires_grad_(False)


def set_WO_identity_and_freeze(model: CopyTransformer) -> None:
    with torch.no_grad():
        for blk in model.blocks:
            H = blk.attn.n_heads
            D = blk.attn.d_model
            F = blk.attn.d_head
            eye = torch.eye(F, D, device=blk.attn.W_O.device)  # [F, D]
            blk.attn.W_O.copy_(eye.unsqueeze(0).repeat(H, 1, 1))
            blk.attn.W_O.requires_grad_(False)


# -----------------------------
# Train and eval
# -----------------------------

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
    model: CopyTransformer,
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
        mask = build_mask(list_len, device)
        for i in range(0, val_inputs.size(0), batch_size):
            xb = val_inputs[i : i + batch_size].to(device)
            yb = val_targets[i : i + batch_size].to(device)
            logits = model(xb, mask)[:, list_len + 1 :, :]  # [B, L, V]
            preds = logits.argmax(dim=-1)
            gold = yb[:, list_len + 1 :]
            hits += (preds == gold).sum().item()
            tots += preds.numel()
    return hits / max(1, tots)


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
    grad_clip: float,
) -> Dict:
    # Deterministic per-config seed
    seed_material = (str(asdict(cfg)) + str(base_seed)).encode("utf-8")
    seed = int.from_bytes(hashlib.sha256(seed_material).digest()[:4], "big")
    set_global_seed(seed + cfg.run_idx)

    vocab = cfg.N_DIGITS + 2
    seq_len = 2 * cfg.LIST_LEN + 1

    model = CopyTransformer(
        n_layers=cfg.N_LAYERS,
        n_heads=cfg.N_HEAD,
        d_model=cfg.D_MODEL,
        vocab_size=vocab,
        seq_len=seq_len,
        use_ln=cfg.USE_LN,
        use_bias=cfg.USE_BIAS,
    ).to(device)

    if not cfg.USE_BIAS:
        strip_biases(model)
    if cfg.FREEZE_WV:
        set_WV_identity_and_freeze(model)
    if cfg.FREEZE_WO:
        set_WO_identity_and_freeze(model)

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
    mask = build_mask(cfg.LIST_LEN, device)
    g = torch.Generator(device=device).manual_seed(seed + 999)

    last_eval = 0.0
    steps_run = 0
    start = time.time()

    pbar = tqdm(total=train_steps, desc="train", leave=False, ncols=80)
    for step in range(1, train_steps + 1):
        xb, yb = make_batch(batch_size, cfg.N_DIGITS, cfg.LIST_LEN, device, g)
        logits = model(xb, mask)[:, cfg.LIST_LEN + 1 :, :]
        gold = yb[:, cfg.LIST_LEN + 1 :]

        loss = ce(logits.reshape(-1, logits.size(-1)), gold.reshape(-1))
        loss.backward()
        if grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        opt.step()
        opt.zero_grad(set_to_none=True)

        if step % eval_every == 0 or step == train_steps:
            last_eval = eval_accuracy(model, val_inputs, val_targets, cfg.LIST_LEN)
            if last_eval >= early_stop_acc:
                steps_run = step
                pbar.update(train_steps - step + 1)
                break

        pbar.update(1)
        steps_run = step
    pbar.close()

    train_time_sec = time.time() - start
    if last_eval == 0.0:
        last_eval = eval_accuracy(model, val_inputs, val_targets, cfg.LIST_LEN)

    return {
        "val_acc": last_eval,
        "steps_trained": steps_run,
        "train_time_sec": train_time_sec,
        "params_total": total_params,
        "params_trainable": trainable_params,
        "val_examples": val_inputs.size(0),
        "batch_size": batch_size,
    }


# -----------------------------
# Grid (exact values requested)
# -----------------------------

def build_grid() -> List[Config]:
    LIST_LEN = list(range(2, 6))
    N_DIGITS = [10, 100, 1000]
    D_MODEL = [8, 32, 128, 256]
    N_HEAD = [1, 2, 4]
    N_LAYERS = list(range(2, 7))
    USE_LN = [False]
    USE_BIAS = [False]
    FREEZE_WV = [True]
    FREEZE_WO = [True]
    WEIGHT_DECAY = [0.01]
    RUNS = [0, 1, 2]

    cfgs: List[Config] = []
    for (L, ND, DM, NH, NL, LN, UB, FWV, FWO, WD, R) in product(
        LIST_LEN, N_DIGITS, D_MODEL, N_HEAD, N_LAYERS,
        USE_LN, USE_BIAS, FREEZE_WV, FREEZE_WO, WEIGHT_DECAY, RUNS
    ):
        if DM % NH != 0:
            continue
        cfgs.append(Config(
            LIST_LEN=L, N_DIGITS=ND, D_MODEL=DM, N_HEAD=NH, N_LAYERS=NL,
            USE_LN=LN, USE_BIAS=UB, FREEZE_WV=FWV, FREEZE_WO=FWO,
            WEIGHT_DECAY=WD, run_idx=R
        ))
    return cfgs


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
    p.add_argument("--early-stop-acc", type=float, default=0.99, help="early stop threshold")
    p.add_argument("--val-size", type=int, default=8192, help="validation set size")
    p.add_argument("--batch-size", type=int, default=1024, help="training batch size")
    p.add_argument("--lr", type=float, default=1e-3, help="AdamW learning rate")
    p.add_argument("--grad-clip", type=float, default=1.0, help="global grad norm clip; 0=off")
    return p.parse_args()


def main():
    args = parse_args()
    device = pick_device(args.device)
    set_global_seed(args.seed)

    cfgs = build_grid()
    total = len(cfgs)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    fields = [
        "LIST_LEN", "N_DIGITS", "D_MODEL", "N_HEAD", "N_LAYERS",
        "USE_LN", "USE_BIAS", "FREEZE_WV", "FREEZE_WO", "WEIGHT_DECAY",
        "run_idx", "val_acc", "steps_trained", "train_time_sec",
        "params_total", "params_trainable", "val_examples", "batch_size", "device"
    ]
    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()

        outer = tqdm(total=total, desc="grid", ncols=100)
        for cfg in cfgs:
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
            writer.writerow(row)
            f.flush()
            outer.update(1)
        outer.close()

    print(f"Wrote {total} rows to {args.output}")


if __name__ == "__main__":
    main()
