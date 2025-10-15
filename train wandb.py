"""
Training with Weights & Biases sweeps.

- Reads WANDB_API_KEY, WANDB_PROJECT, WANDB_ENTITY from environment or .env
- Supports parameter sweeps over:
    list_len, n_dig, d_model, n_head, n_layer, use_ln, use_bias,
    freeze_wv, freeze_wo, lr, weight_decay, max_train_steps

Project: theo-farrell99-durham-university/Copying Lists in Superposition/
"""

# %% [markdown]
# ## Setup

# %%
import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F

import os
import copy
from datetime import datetime # for unique model naming

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse

import einops
import pandas as pd, itertools
from tqdm.auto import tqdm

from transformer_lens import HookedTransformer, HookedTransformerConfig, utils

from model_utils import (
    configure_runtime,
    build_attention_mask,
    save_model,
    make_model,
    accuracy
)
from data import get_dataset

# --- wandb & dotenv ---
try:
    from dotenv import load_dotenv
except Exception:  # pragma: no cover - optional dependency
    def load_dotenv(*args, **kwargs):
        return None
import wandb

# Define globals with safe defaults; they'll be overwritten by setup_runtime()
LIST_LEN = 2
N_DIGITS = 100
VOCAB = N_DIGITS + 2

float_formatter = "{:.5f}".format
np.set_printoptions(formatter={'float_kind':float_formatter})


# %% [markdown]
# ## Model

# %%
def get_default_config():
    return dict(
        list_len=2,
        n_dig=100,
        d_model=64,
        n_head=1,
        n_layer=2,
        use_ln=False,
        use_bias=False,
        freeze_wv=True,
        freeze_wo=True,
        lr=1e-3,
        weight_decay=0.01,
        max_train_steps=50_000,
        use_checkpointing=False,
        train_split=0.8,
        early_stop_acc=0.999,
        batch_cap_train=128,
        batch_cap_val=256,
        seed=0,
    )

# --- dataset --- (not necessary as we fix seed?)
# DATASET_NAME = None # None ==> generate new one
# listlen2_digits10_dupes
# listlen2_digits10_nodupes
# listlen2_digits100_dupes_traindupesonly
# listlen2_digits100_dupes
# listlen2_digits100_nodupes

def setup_runtime(cfg):
    DEV = ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg.get("seed", 0))

    LIST_LEN = int(cfg["list_len"])  # [d1, d2, ...]
    N_DIGITS = int(cfg["n_dig"])    # vocabulary digits 0..n_dig-1
    SEQ_LEN = LIST_LEN * 2 + 1       # [d..., SEP, o...]
    VOCAB = N_DIGITS + 2             # digits + MASK + SEP

    # Expose for rest of module
    globals().update(dict(LIST_LEN=LIST_LEN, SEQ_LEN=SEQ_LEN, N_DIGITS=N_DIGITS, VOCAB=VOCAB))

    configure_runtime(list_len=LIST_LEN, seq_len=SEQ_LEN, vocab=VOCAB, device=DEV)
    return DEV

# %%
# ---------- mask ----------
# attention mask for [d1, d2, SEP, o1, o2] looks like this:
# -    d1    d2    SEP    o1    o2   (keys)
# d1  -inf  -inf   -inf  -inf  -inf
# d2   0    -inf   -inf  -inf  -inf
# SEP  0      0    -inf  -inf  -inf
# o1  -inf  -inf    0    -inf   -inf
# o2  -inf  -inf    0      0    -inf
# (queries)

def show_mask_once():
    mask_bias, _ = build_attention_mask()
    print(mask_bias.cpu()[0][0])

# %%
def make_dataloaders(cfg):
    MASK = N_DIGITS
    SEP = N_DIGITS + 1
    train_ds, val_ds = get_dataset(
        list_len=LIST_LEN,
        n_digits=N_DIGITS,
        train_split=cfg.get("train_split", 0.8),
        mask_tok=MASK,
        sep_tok=SEP,
    )
    train_bs = min(int(cfg.get("batch_cap_train", 128)), len(train_ds))
    val_bs = min(int(cfg.get("batch_cap_val", 256)), len(val_ds))
    train_dl = DataLoader(train_ds, train_bs, shuffle=True, drop_last=True)
    val_dl = DataLoader(val_ds, val_bs, drop_last=False)
    return train_dl, val_dl


# %%
def train_loop(m, train_dl, val_dl, *, max_steps, early_stop_acc, lr, weight_decay, checkpoints, device):
    opt = torch.optim.AdamW(m.parameters(), lr, weight_decay=weight_decay)
    ce = torch.nn.CrossEntropyLoss()
    dl = itertools.cycle(train_dl)  # infinite iterator
    best_acc = 0.0
    for step in tqdm(range(max_steps), desc="Training"):
        inputs, targets = next(dl)
        logits = m(inputs.to(device))[:, LIST_LEN + 1 :].reshape(-1, VOCAB)
        loss = ce(logits, targets[:, LIST_LEN + 1 :].reshape(-1).to(device))
        loss.backward()
        opt.step()
        opt.zero_grad()

        if (step + 1) % 100 == 0:
            acc = accuracy(m, val_dl, list_len=LIST_LEN, device=device)
            best_acc = max(best_acc, acc)
            wandb.log({"step": step + 1, "loss": loss.item(), "val_acc": acc})
            if acc >= early_stop_acc:
                print(f"Early stopping at step {step + 1} with accuracy {acc:.2%} >= {early_stop_acc:.2%}")
                break
    final_acc = accuracy(m, val_dl, list_len=LIST_LEN, device=device)
    wandb.log({"final_val_acc": final_acc, "best_val_acc": best_acc})
    print(f"Final accuracy: {final_acc:.2%}")
    return final_acc


# %%
def make_model_for_cfg(cfg, device):
    return make_model(
        n_layers=int(cfg["n_layer"]),
        n_heads=int(cfg["n_head"]),
        d_model=int(cfg["d_model"]),
        ln=bool(cfg["use_ln"]),
        use_bias=bool(cfg["use_bias"]),
        freeze_wv=bool(cfg["freeze_wv"]),
        freeze_wo=bool(cfg["freeze_wo"]),
        device=device,
    )

# %%
def build_sweep_config():
    return {
        "method": "bayes",  # or "grid"/"random"
        "metric": {"name": "final_val_acc", "goal": "maximize"},
        "parameters": {
            "list_len": {"values": [2]},
            "n_dig": {"values": [10, 100]},
            "d_model": {"values": [32, 64, 128]},
            "n_head": {"values": [1]},
            "n_layer": {"values": [1, 2, 3]},
            "use_ln": {"values": [False, True]},
            "use_bias": {"values": [False, True]},
            "freeze_wv": {"values": [True, False]},
            "freeze_wo": {"values": [True, False]},
            "lr": {"values": [1e-3, 5e-4, 1e-4]},
            "weight_decay": {"values": [0.0, 0.01]},
            "max_train_steps": {"values": [10_000, 25_000, 50_000]},
        },
        "early_terminate": {"type": "hyperband", "min_iter": 10},
    }


def run_train():
    with wandb.init(project=os.getenv("WANDB_PROJECT", "Copying Lists in Superposition"),
                    entity=os.getenv("WANDB_ENTITY", "theo-farrell99-durham-university")):
        cfg = get_default_config()
        cfg.update(dict(wandb.config))

        device = setup_runtime(cfg)
        show_mask_once()
        train_dl, val_dl = make_dataloaders(cfg)
        model = make_model_for_cfg(cfg, device)

        final_acc = train_loop(
            model,
            train_dl,
            val_dl,
            max_steps=int(cfg["max_train_steps"]),
            early_stop_acc=float(cfg.get("early_stop_acc", 0.999)),
            lr=float(cfg["lr"]),
            weight_decay=float(cfg["weight_decay"]),
            checkpoints=bool(cfg.get("use_checkpointing", False)),
            device=device,
        )

        # Save model checkpoint if it does well
        run_ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        model_name = f"{int(cfg['n_layer'])}layer_{int(cfg['n_dig'])}dig_{int(cfg['d_model'])}d_{run_ts}"
        model_path = os.path.join("models", model_name + ".pt")
        if final_acc >= 0.95:
            save_model(model, model_path)
            wandb.save(model_path)


def main():
    # Load environment variables from .env if present
    load_dotenv()

    # Provide a clear place to paste WANDB_API_KEY securely
    # 1) Create a file named '.env' next to this script
    # 2) Add: WANDB_API_KEY=your_key_here
    # This file is ignored by git (see .gitignore)
    if not os.getenv("WANDB_API_KEY"):
        print("Warning: WANDB_API_KEY is not set. Set it in your environment or in a local .env file.")

    # Optional: set run mode, e.g., offline for no network
    wandb_mode = os.getenv("WANDB_MODE")
    if wandb_mode:
        os.environ["WANDB_MODE"] = wandb_mode

    # Define sweep and launch agent
    sweep_config = build_sweep_config()
    project = os.getenv("WANDB_PROJECT", "Copying Lists in Superposition")
    entity = os.getenv("WANDB_ENTITY", "theo-farrell99-durham-university")
    sweep_id = wandb.sweep(sweep_config, project=project, entity=entity)
    # You can control count via env SWEEP_COUNT or default to 10
    count = int(os.getenv("SWEEP_COUNT", "10"))
    wandb.agent(sweep_id, function=run_train, count=count)


if __name__ == "__main__":
    main()



