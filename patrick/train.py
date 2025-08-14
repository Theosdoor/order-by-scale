#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# train.py

"""
Single-config training, saving, and loading utilities for the Copy-Task models.

This script reuses grid_search.py for the model, data, and training loop.
It adds:
  - train_with_config(cfg, ...)
  - save_model(path, model, cfg, metrics=None)
  - load_model(path, map_location='auto') -> (model, cfg, metrics)

Run directly to train and save two models with:
  LIST_LEN=2, N_DIGITS=100, D_MODEL=128, N_HEAD=1,
  N_LAYERS in {2,3}, USE_LN=False, USE_BIAS=False,
  FREEZE_WV=True, FREEZE_WO=True, WEIGHT_DECAY=0.01

Outputs:
  ./models/<deterministic_name>.pt
"""

import os
from dataclasses import asdict
from typing import Dict, Tuple

import torch
import torch.nn as nn  # for typing

from grid_search import (
    Config,
    pick_device,
    set_global_seed,
    build_model_from_config,
    train_and_return_model,
)


# -----------------------------
# Checkpoint I/O
# -----------------------------

def _config_stem(cfg: Config) -> str:
    wd = str(cfg.WEIGHT_DECAY).replace(".", "p")
    stem = (
        f"copytask_L{cfg.LIST_LEN}_ND{cfg.N_DIGITS}_DM{cfg.D_MODEL}"
        f"_H{cfg.N_HEAD}_NL{cfg.N_LAYERS}_LN{int(cfg.USE_LN)}_B{int(cfg.USE_BIAS)}"
        f"_FWV{int(cfg.FREEZE_WV)}_FWO{int(cfg.FREEZE_WO)}_WD{wd}_RUN{cfg.run_idx}"
    )
    return stem


def save_model(path: str, model: nn.Module, cfg: Config, metrics: Dict = None) -> None:
    """Save model weights + config + optional metrics to a single file."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    ckpt = {
        "format": "copy_task_checkpoint/v1",
        "model_class": "HookedTransformer",
        "cfg": asdict(cfg),
        "model_state_dict": model.state_dict(),
        "metrics": metrics or {},
        "torch_version": torch.__version__,
    }
    torch.save(ckpt, path)


def load_model(path: str, map_location: str = "auto") -> Tuple[nn.Module, Config, Dict]:
    """Load a model from disk. Reconstructs the model from the saved config."""
    device = pick_device(map_location if map_location != "auto" else "auto")
    # Map to CPU or the selected device to be safe if source device isn't present.
    ckpt = torch.load(path, map_location=device, weights_only=False)

    if "cfg" not in ckpt or "model_state_dict" not in ckpt:
        raise ValueError(f"Checkpoint missing required keys: {path}")

    cfg_dict = ckpt["cfg"]
    cfg = Config(**cfg_dict)

    model = build_model_from_config(cfg, device)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()
    return model, cfg, ckpt.get("metrics", {})


# -----------------------------
# One-shot training helper
# -----------------------------

def train_with_config(
    cfg: Config,
    device: torch.device,
    *,
    base_seed: int = 42,
    train_steps: int = 50_000,
    batch_size: int = 1024,
    eval_every: int = 2_500,
    early_stop_acc: float = 0.999,
    val_size: int = 8_192,
    lr: float = 1e-4,
) -> Tuple[nn.Module, Dict]:
    """Train one model for a given cfg and return (model, metrics)."""
    model, metrics = train_and_return_model(
        cfg=cfg,
        device=device,
        train_steps=train_steps,
        batch_size=batch_size,
        eval_every=eval_every,
        early_stop_acc=early_stop_acc,
        val_size=val_size,
        base_seed=base_seed,
        lr=lr,
    )
    return model, metrics


# -----------------------------
# Script entry
# -----------------------------

if __name__ == "__main__":
    device = pick_device("auto")
    set_global_seed(0)
    os.makedirs("./models", exist_ok=True)

    common = dict(
        LIST_LEN=2,
        N_DIGITS=100,
        D_MODEL=128,
        N_HEAD=1,
        USE_LN=False,
        USE_BIAS=False,
        FREEZE_WV=True,
        FREEZE_WO=True,
        WEIGHT_DECAY=0.01,
        run_idx=0,
    )

    for n_layers in [2, 3]:
        cfg = Config(N_LAYERS=n_layers, **common)
        model, metrics = train_with_config(cfg, device, batch_size=128, train_steps=100_000)
        path = os.path.join("models", f"{_config_stem(cfg)}.pt")
        save_model(path, model, cfg, metrics)
        print(f"Saved: {path} | val_acc={metrics['val_acc']:.4f} | steps={metrics['steps_trained']}")
