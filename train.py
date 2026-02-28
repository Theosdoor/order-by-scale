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
from dotenv import load_dotenv

from transformer_lens import HookedTransformer, HookedTransformerConfig, utils

from src.runtime import configure_runtime
from src.model_utils import make_model, build_attention_mask, save_model, accuracy
from src.datasets import get_dataset

# Wandb project name (hardcoded for this repo)
WANDB_PROJECT = "list-comp"
wandb = None  # Will be imported if --wandb flag is set

float_formatter = "{:.5f}".format
np.set_printoptions(formatter={'float_kind':float_formatter})


# %%
# ---------- parameters ----------
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Train a list comparison transformer model")
    
    # Model architecture
    parser.add_argument("--n-layers", type=int, default=2, help="Number of transformer layers")
    parser.add_argument("--n-heads", type=int, default=1, help="Number of attention heads")
    parser.add_argument("--d-model", type=int, default=64, help="Model dimension")
    parser.add_argument("--n-digits", type=int, default=100, help="Number of digits (vocabulary size - 2)")
    parser.add_argument("--list-len", type=int, default=2, help="Length of input list (e.g., 2 for [d1,d2], 3 for [d1,d2,d3])")
    
    # Model features (flags)
    parser.add_argument("--ln", action="store_true", default=False, help="Use layer normalization")
    parser.add_argument("--bias", action="store_true", default=False, help="Use bias terms")
    parser.add_argument("--wv", action="store_true", default=False, help="Learn W_V (else freeze to identity)")
    parser.add_argument("--wo", action="store_true", default=False, help="Learn W_O (else freeze to identity)")
    parser.add_argument("--mlp", action="store_true", default=False, help="Include MLP layers")
    
    # Training
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--max-steps", type=int, default=100000, help="Max training steps")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--checkpoint", action="store_true", help="Save checkpoints during training")
    parser.add_argument("--min-acc", type=float, default=0.9, help="Minimum accuracy to stop training")
    parser.add_argument("--max-retries", type=int, default=3, help="Max training retries if min-acc not reached")
    parser.add_argument("--early-stop-acc", type=float, default=0.999, help="Stop training when this accuracy is reached")
    parser.add_argument("--early-stopping-patience", type=int, default=None, help="Stop after N eval steps with no improvement (eval every 100 steps). Only triggers if acc > early-stopping-threshold. Set to None to disable.")
    parser.add_argument("--early-stopping-threshold", type=float, default=0.9, help="Only allow early stopping if accuracy is above this threshold")
    parser.add_argument("--train-batch-size", type=int, default=2048, help="Training batch size")
    parser.add_argument("--val-batch-size", type=int, default=4096, help="Validation batch size")
    parser.add_argument("--use-lr-scheduler", action="store_true", help="Use LR scheduler (warmup + cosine decay)")
    parser.add_argument("--warmup-steps", type=int, default=1000, help="Number of warmup steps for LR scheduler")
    parser.add_argument("--max-grad-norm", type=float, default=None, help="Max gradient norm for clipping. Set to None to disable.")
    
    # Logging
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging")
    
    # Output
    parser.add_argument("--name", type=str, default=None, help="Custom model name (overrides auto-generated)")
    
    return parser.parse_known_args()[0]

# Parse arguments (use defaults if running as notebook/interactive kernel)
args = parse_args()

# Extract to module-level variables for compatibility
LIST_LEN = args.list_len
SEQ_LEN = LIST_LEN * 2 + 1  # [d1, ..., dn, SEP, o1, ..., on]

N_DIGITS = args.n_digits
DIGITS = list(range(N_DIGITS))
MASK = N_DIGITS
SEP = N_DIGITS + 1
VOCAB = len(DIGITS) + 2

D_MODEL = args.d_model
N_HEAD = args.n_heads
N_LAYER = args.n_layers
USE_LN = args.ln
USE_BIAS = args.bias
USE_WV = args.wv
USE_WO = args.wo
ATTN_ONLY = not args.mlp

LEARNING_RATE = args.lr
WEIGHT_DECAY = args.weight_decay
MAX_TRAIN_STEPS = args.max_steps
USE_CHECKPOINTING = args.checkpoint
SEED = args.seed

# Generate model name: L{layers}_H{heads}_D{d_model}_V{digits}[_flags]_timestamp
# Flags are only included if non-default (ln, bias, wv, wo, mlp)
RUN_TS = datetime.now().strftime("%y%m%d-%H%M%S")
base_name = f"L{N_LAYER}_H{N_HEAD}_D{D_MODEL}_V{N_DIGITS}"

# Add flags suffix only for non-default settings
flags = []
if LIST_LEN != 2: flags.append(f"len{LIST_LEN}")
if USE_LN: flags.append("ln")
if USE_BIAS: flags.append("bias")
if USE_WV: flags.append("wv")
if USE_WO: flags.append("wo")
if not ATTN_ONLY: flags.append("mlp")
flags_suffix = "_" + "-".join(flags) if flags else ""

MODEL_NAME = args.name if args.name else f"{base_name}{flags_suffix}_{RUN_TS}"

# Construct path relative to project root (parent of model_scripts/)
_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(_PROJECT_ROOT, "models", MODEL_NAME + ".pt")

DEV = "cuda" if torch.cuda.is_available() else "cpu"

# Provide runtime config so we don't need to thread constants everywhere
configure_runtime(list_len=LIST_LEN, seq_len=SEQ_LEN, vocab=VOCAB, device=DEV, seed=SEED)

# %%
# ---------- mask ----------
# attention mask for [d1, d2, SEP, o1, o2] looks like this:
# -    d1    d2    SEP    o1    o2   (keys)
# d1   0    -inf   -inf  -inf  -inf
# d2   0    -inf   -inf  -inf  -inf
# SEP  0      0    -inf  -inf  -inf
# o1  -inf  -inf    0    -inf   -inf
# o2  -inf  -inf    0      0    -inf
# (queries)

# view mask
mask_bias, _ = build_attention_mask()
mask_bias.cpu()[0][0]

# %%
# ---------- dataset ----------
train_ds, val_ds = get_dataset(
    list_len=LIST_LEN, 
    n_digits=N_DIGITS, 
    train_split=0.8,
    mask_tok=MASK, # use MASK as mask token
    sep_tok=SEP, # use SEP as separator token
    )

train_batch_size = min(args.train_batch_size, len(train_ds))
val_batch_size = min(args.val_batch_size, len(val_ds))
train_dl = DataLoader(train_ds, train_batch_size, shuffle=True, drop_last=True)
val_dl = DataLoader(val_ds, val_batch_size, drop_last=False)

print("Input:", train_ds[0][0])
print("Target:", train_ds[0][1])
print(f"Train dataset size: {len(train_ds)}, Validation dataset size: {len(val_ds)}")


# %%
def train(m, max_steps=10000, early_stop_acc=1.0, checkpoints=False, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY, 
          patience=5, patience_threshold=0.9, use_wandb=False, use_lr_scheduler=False, warmup_steps=1000, max_grad_norm=None):
    """Train the model with optional early stopping and wandb logging.
    
    Args:
        patience: Stop after this many eval steps (every 100 train steps) with no improvement.
                  Set to None or 0 to disable early stopping by patience.
        patience_threshold: Only allow patience-based early stopping if accuracy is above this threshold.
        use_lr_scheduler: If True, use warmup + cosine decay LR scheduler.
        warmup_steps: Number of warmup steps for LR scheduler.
        max_grad_norm: Max gradient norm for clipping. Set to None to disable.
    """
    opt = torch.optim.AdamW(m.parameters(), lr, weight_decay=weight_decay)
    
    # Setup LR scheduler (warmup + cosine decay)
    scheduler = None
    if use_lr_scheduler:
        from torch.optim.lr_scheduler import LambdaLR
        import math
        
        def lr_lambda(current_step):
            # Warmup phase
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            # Cosine decay phase
            progress = float(current_step - warmup_steps) / float(max(1, max_steps - warmup_steps))
            return 0.5 * (1.0 + math.cos(math.pi * progress))
        
        scheduler = LambdaLR(opt, lr_lambda)
    
    ce = torch.nn.CrossEntropyLoss()
    dl = itertools.cycle(train_dl)  # infinite iterator
    pbar = tqdm(range(max_steps), desc="Training")
    
    # Early stopping state
    best_acc = 0.0
    steps_without_improvement = 0
    
    for step in pbar:
        inputs, targets = next(dl)
        # get logits/loss for output tokens only
        logits = m(inputs.to(DEV))[:, LIST_LEN+1:].reshape(-1, VOCAB) 
        loss = ce(logits, targets[:, LIST_LEN+1:].reshape(-1).to(DEV))
        loss.backward()
        
        # Gradient clipping
        if max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(m.parameters(), max_grad_norm)
        
        opt.step()
        opt.zero_grad()
        
        # Step LR scheduler
        if scheduler is not None:
            scheduler.step()
        
        if (step + 1) % 100 == 0:
            acc = accuracy(m, val_dl)
            
            # Get current LR for logging
            current_lr = opt.param_groups[0]['lr']
            
            # Log to wandb
            if use_wandb:
                wandb.log({"train/loss": loss.item(), "val/accuracy": acc, "train/step": step + 1, "train/lr": current_lr})
            
            # Check for target accuracy
            if acc >= early_stop_acc:
                print(f"Early stopping at step {step + 1} with accuracy {acc:.2%} >= {early_stop_acc:.2%}")
                break
            
            # Check for patience-based early stopping (only if acc > threshold)
            if acc > best_acc:
                best_acc = acc
                steps_without_improvement = 0
            else:
                steps_without_improvement += 1
                # Only trigger patience-based stopping if we're above the threshold
                if patience and steps_without_improvement >= patience and acc >= patience_threshold:
                    print(f"Early stopping at step {step + 1}: no improvement for {patience} eval steps (acc: {acc:.2%}, best: {best_acc:.2%})")
                    break
            
            # Update tqdm bar w/ metrics
            postfix = {
                "loss": f"{loss.item():.4f}",
                "acc": f"{acc:.2%}",
                "best": f"{best_acc:.2%}",
            }
            if scheduler is not None:
                postfix["lr"] = f"{current_lr:.2e}"
            pbar.set_postfix(postfix)
            
            if checkpoints and (step+1) % 50000 == 0:
                save_model(m, MODEL_PATH)
            
    print(f"Final accuracy: {accuracy(m, val_dl):.2%}")


# %%
# train and SAVE new model
MIN_ACC = args.min_acc
MAX_RETRIES = args.max_retries
EARLY_STOP_ACC = args.early_stop_acc
PATIENCE = args.early_stopping_patience
USE_WANDB = args.wandb

# Initialize wandb if enabled
if USE_WANDB:
    load_dotenv()  # Load .env file for WANDB_API_KEY and WANDB_ENTITY
    import wandb as _wandb
    wandb = _wandb
    wandb.init(
        project=WANDB_PROJECT,
        config={
            "n_layers": N_LAYER,
            "n_heads": N_HEAD,
            "d_model": D_MODEL,
            "n_digits": N_DIGITS,
            "list_len": LIST_LEN,
            "use_ln": USE_LN,
            "use_bias": USE_BIAS,
            "use_wv": USE_WV,
            "use_wo": USE_WO,
            "attn_only": ATTN_ONLY,
            "lr": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "max_steps": MAX_TRAIN_STEPS,
            "early_stopping_patience": PATIENCE,
            "early_stopping_threshold": args.early_stopping_threshold,
            "use_lr_scheduler": args.use_lr_scheduler,
            "warmup_steps": args.warmup_steps,
            "max_grad_norm": args.max_grad_norm,
            "train_batch_size": args.train_batch_size,
            "seed": SEED,
        },
        name=MODEL_NAME,
    )

print(f"Training {MODEL_NAME}")
print(f"  Config: {N_LAYER} layers, {N_HEAD} heads, d_model={D_MODEL}, LN={USE_LN}, bias={USE_BIAS}, WV={USE_WV}, WO={USE_WO}, attn_only={ATTN_ONLY}")
print(f"  Target: min_acc={MIN_ACC:.1%}, max_retries={MAX_RETRIES}, patience={PATIENCE}")

best_acc = 0
best_model = None

for attempt in range(MAX_RETRIES):
    model = make_model(
        n_layers=N_LAYER,
        n_heads=N_HEAD,
        d_model=D_MODEL,
        ln=USE_LN,
        use_bias=USE_BIAS,
        use_wv=USE_WV,
        use_wo=USE_WO,
        attn_only=ATTN_ONLY,
    )
    train(model, max_steps=MAX_TRAIN_STEPS, early_stop_acc=EARLY_STOP_ACC,
          checkpoints=USE_CHECKPOINTING, patience=PATIENCE, patience_threshold=args.early_stopping_threshold,
          use_wandb=USE_WANDB, use_lr_scheduler=args.use_lr_scheduler, warmup_steps=args.warmup_steps,
          max_grad_norm=args.max_grad_norm)
    acc = accuracy(model, val_dl)
    
    if acc > best_acc:
        best_acc = acc
        best_model = model
    
    if acc >= MIN_ACC:
        print(f"Achieved {acc:.2%} >= {MIN_ACC:.1%} on attempt {attempt+1}")
        break
    else:
        print(f"Attempt {attempt+1}/{MAX_RETRIES}: acc={acc:.2%} < {MIN_ACC:.1%}, retrying...")
else:
    print(f"Warning: Best accuracy {best_acc:.2%} after {MAX_RETRIES} attempts (target: {MIN_ACC:.1%})")

# Always save the best model
MODEL_NAME_WITH_ACC = f'{MODEL_NAME}_acc{best_acc:.4f}'
MODEL_PATH_WITH_ACC = os.path.join(_PROJECT_ROOT, "models", f"{MODEL_NAME_WITH_ACC}.pt")  # noqa: E501
save_model(best_model, MODEL_PATH_WITH_ACC)

# Finish wandb run
if USE_WANDB:
    wandb.log({"final/accuracy": best_acc})
    wandb.finish()

# %%
# --- Model Parameters Overview ---
m_for_overview = globals().get('model', None)
if m_for_overview is not None:
    print("--- Overview of Model Parameters ---")   
    total_params = 0
    trainable_params = 0

    # Use a formatted string for better alignment
    print(f"{'Parameter Name':<40} | {'Shape':<20} | {'Trainable':<10}")
    print("-" * 80)

    for name, param in m_for_overview.named_parameters():
        shape_str = str(tuple(param.shape))
        is_trainable = "Yes" if param.requires_grad else "No"
        total_params += param.numel()

        if not param.requires_grad:
            continue
        # Print only trainable parameters
        print(f"{name:<40} | {shape_str:<20} | {is_trainable:<10}")
        trainable_params += param.numel()

    print("-" * 80)
    print(f"Total parameters: {total_params}")
    print(f"Trainable parameters: {trainable_params}")
    print("-" * 80)

# %%
