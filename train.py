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

float_formatter = "{:.5f}".format
np.set_printoptions(formatter={'float_kind':float_formatter})


# %% [markdown]
# ## Model

# %%
# ---------- parameters ----------
LIST_LEN = 2 # [d1, d2]
SEQ_LEN = LIST_LEN * 2 + 1 # [d1, d2, SEP, o1, o2]

N_DIGITS = 100
DIGITS = list(range(N_DIGITS)) # 100 digits from 0 to 99
MASK = N_DIGITS # special masking token for o1 and o2
SEP = N_DIGITS+1 # special seperator token for the model to think about the input (+1 to avoid confusion with the last digit)
VOCAB = len(DIGITS) + 2  # + the special tokens

D_MODEL = 64
N_HEAD = 1
N_LAYER = 2
USE_LN = False # use layer norm in model
USE_BIAS = False # use bias in model
FREEZE_WV = True # no value matrix in attn 
FREEZE_WO = True # no output matrix in attn (i.e. attn head can only copy inputs to outputs)

LEARNING_RATE = 1e-3 # default 1e-3
WEIGHT_DECAY = 0.01 # default 0.01
MAX_TRAIN_STEPS = 50_000 # max training steps
USE_CHECKPOINTING = False # whether to use checkpointing for training

RUN_TS = datetime.now().strftime("%Y%m%d-%H%M%S")
MODEL_NAME = f'{N_LAYER}layer_{N_DIGITS}dig_{D_MODEL}d_{RUN_TS}'
# MODEL_NAME = 
MODEL_PATH = "models/" + MODEL_NAME + ".pt"

# --- dataset --- (not necessary as we fix seed?)
# DATASET_NAME = None # None ==> generate new one
# listlen2_digits10_dupes
# listlen2_digits10_nodupes
# listlen2_digits100_dupes_traindupesonly
# listlen2_digits100_dupes
# listlen2_digits100_nodupes

DEV = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)
torch.manual_seed(0)

# Provide runtime config so we don't need to thread constants everywhere
configure_runtime(list_len=LIST_LEN, seq_len=SEQ_LEN, vocab=VOCAB, device=DEV)

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
print(mask_bias.cpu()[0][0])

# %%
# ---------- dataset ----------
train_ds, val_ds = get_dataset(
    list_len=LIST_LEN, 
    n_digits=N_DIGITS, 
    train_split=0.8,
    mask_tok=MASK, # use MASK as mask token
    sep_tok=SEP, # use SEP as separator token
    )

train_batch_size = min(128, len(train_ds))
val_batch_size = min(256, len(val_ds))
train_dl = DataLoader(train_ds, train_batch_size, shuffle=True, drop_last=True)
val_dl = DataLoader(val_ds, val_batch_size, drop_last=False)

print("Input:", train_ds[0][0])
print("Target:", train_ds[0][1])
print(f"Train dataset size: {len(train_ds)}, Validation dataset size: {len(val_ds)}")


# %%
def train(m, max_steps=10_000, early_stop_acc=0.999, checkpoints=False, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY):
    opt = torch.optim.AdamW(m.parameters(), lr, weight_decay=weight_decay)
    ce = torch.nn.CrossEntropyLoss()
    dl = itertools.cycle(train_dl)  # infinite iterator
    pbar = tqdm(range(max_steps), desc="Training")
    for step in pbar:
        inputs, targets = next(dl)
        # get logits/loss for output tokens only
        logits = m(inputs.to(DEV))[:, LIST_LEN+1:].reshape(-1, VOCAB) 
        loss = ce(logits, targets[:, LIST_LEN+1:].reshape(-1).to(DEV))
        loss.backward()
        opt.step()
        opt.zero_grad()
        if (step + 1) % 100 == 0:
            acc = accuracy(m, val_dl)
            if acc > early_stop_acc:
                print(f"Early stopping at step {step + 1} with accuracy {acc:.2%} >= {early_stop_acc:.2%}")
                break
            # Update tqdm bar w/ metrics
            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "acc": f"{acc:.2%}",
            })
            if checkpoints and (step+1) % 50_000 == 0:
                save_model(m, MODEL_PATH)
            
    print(f"Final accuracy: {accuracy(m, val_dl):.2%}")


# %%
# Check train set
train_ds[:5]

# %%
# ---------- experiment grid ----------
def make_name(d_model, n_layers, ln, use_bias, freeze_wv, freeze_wo):
    parts = [
        f"d{d_model}",
        f"{n_layers}L",
        ("LN" if ln else "noLN"),
        ("Bias" if use_bias else "noBias"),
        ("fWV" if freeze_wv else "uWV"), # freeze / unfreeze
        ("fWO" if freeze_wo else "uWO"),
    ]
    return "_".join(parts)

specs = [
    # {'name': 'd256', 'd_model': 256},
    # {'name': 'd128', 'd_model': 128},
    # {'name': 'd64', 'd_model': 64},
    
    # {'name': 'd32', 'd_model': 32},
    # {'name': 'd32_ln_bias', 'd_model': 32, 'ln': True, 'use_bias': True},
    # {'name': 'd32_noLN', 'd_model': 32, 'ln': False, 'use_bias': True},
    # {'name': 'd32_noBias', 'd_model': 32, 'ln': True, 'use_bias': False},
    # {'name': 'd32_noLNnoBias', 'd_model': 32, 'ln': False, 'use_bias': False},
    # {'name': 'd32_fwo', 'd_model': 32, 'freeze_wo': True},
    # {'name': 'd32_unfwo', 'd_model': 32, 'freeze_wo': False},

    # {'name': 'd16', 'd_model': 16},
    # {'name': 'd16_ln_bias', 'd_model': 16, 'ln': True, 'use_bias': True},
    # {'name': 'd16_noLN', 'd_model': 16, 'ln': False, 'use_bias': True},
    # {'name': 'd16_noBias', 'd_model': 16, 'ln': True, 'use_bias': False},
    # {'name': 'd16_noLNnoBias', 'd_model': 16, 'ln': False, 'use_bias': False},
    # {'name': 'd16_fwo', 'd_model': 16, 'freeze_wo': True},
    # {'name': 'd16_unfwo', 'd_model': 16, 'freeze_wo': False},

    # {'name': 'd8', 'd_model': 8},
    # {'name': 'd8_ln_bias', 'd_model': 8, 'ln': True, 'use_bias': True},
    # {'name': 'd8_noLN', 'd_model': 8, 'ln': False, 'use_bias': True},
    # {'name': 'd8_noBias', 'd_model': 8, 'ln': True, 'use_bias': False},
    # {'name': 'd8_noLNnoBias', 'd_model': 8, 'ln': False, 'use_bias': False},
    # {'name': 'd8_fwo', 'd_model': 8, 'freeze_wo': True},
    # {'name': 'd8_unfwo', 'd_model': 8, 'freeze_wo': False},

    # {'name': 'd4', 'd_model': 4},
]

# -----------------------
rows = []
for spec in specs:
    # Create a full spec by starting with defaults and updating with the current spec
    full_spec = {
        'n_layers': N_LAYER,
        'n_heads': N_HEAD,
        'd_model': D_MODEL,
        'ln': USE_LN,
        'bias': USE_BIAS,
        'freeze_wv': FREEZE_WV,
        'freeze_wo': FREEZE_WO,
        'lr': LEARNING_RATE,
        'weight_decay': WEIGHT_DECAY,
    }
    full_spec.update(spec) # Overwrite defaults with provided spec values

    print(f"--- Training model: {full_spec['name']} ---")
    model = make_model(
        n_layers=full_spec['n_layers'],
        n_heads=full_spec['n_heads'],
        d_model=full_spec['d_model'],
        ln=full_spec['ln'],
        use_bias=full_spec['bias'],
        freeze_wv=full_spec['freeze_wv'],
        freeze_wo=full_spec['freeze_wo'],
    )

    train(model, max_steps=50_000, lr=full_spec['lr'], weight_decay=full_spec['weight_decay'])
    
    # Add all spec parameters to the results
    result = full_spec.copy()
    result['val_acc'] = round(accuracy(model, val_dl), 4)
    rows.append(result)

df = pd.DataFrame(rows)

# Move 'name' column to the front for better readability
if 'name' in df.columns:
    cols = ['name'] + [col for col in df.columns if col != 'name']
    df = df[cols]

print(df.to_markdown(index=False))

# %% [markdown]
# **RESULTS**
# 

# %%
# train and SAVE new model
acc = 0
while acc < 0.9:
    print(f"Training {MODEL_NAME}")
    model = make_model(
        n_layers=N_LAYER,
        n_heads=N_HEAD,
        d_model=D_MODEL,
        ln=USE_LN,
        use_bias=USE_BIAS,
        freeze_wv=FREEZE_WV,
        freeze_wo=FREEZE_WO,
    )
    train(model, max_steps=MAX_TRAIN_STEPS, checkpoints=USE_CHECKPOINTING)
    acc = accuracy(model, val_dl)
    if acc > 0.8:
        save_model(model, MODEL_PATH)

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

# %%
# Embedding size ablation: run 3 trials per d and save average final accuracy to markdown
# ds = [128, 64, 32, 8]
ds = [64]
n_runs = 3

abl_rows = []
for d in ds:
    print(f"\n=== Embedding ablation: d_model={d} (n_runs={n_runs}) ===")
    run_accs = []
    for run in range(n_runs):
        # Vary seeds per run so results aren't identical
        torch.manual_seed(run)
        np.random.seed(run)

        model = make_model(
            n_layers=N_LAYER,
            n_heads=N_HEAD,
            d_model=d,
            ln=USE_LN,
            use_bias=USE_BIAS,
            freeze_wv=FREEZE_WV,
            freeze_wo=FREEZE_WO,
        )

        # Train and record final validation accuracy for this run
        train(
            model,
            max_steps=MAX_TRAIN_STEPS,
            lr=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY,
        )
        final_acc = accuracy(model, val_dl)
        run_accs.append(final_acc)
        print(f"Run {run+1}/{n_runs} final acc: {final_acc:.4f}")

    avg_acc = float(np.mean(run_accs)) if len(run_accs) > 0 else 0.0
    abl_rows.append({
        'd_model': d,
        'avg_final_acc': round(avg_acc, 4),
    })

abl_df = pd.DataFrame(abl_rows).sort_values('d_model', ascending=False)
md_table = abl_df.to_markdown(index=False)
print("\nAverage final accuracy per d_model:\n")
print(md_table)

# Save to markdown file
with open("embed_abl.md", "w") as f:
    f.write("# Embedding Size Ablation (3 runs each)\n\n")
    f.write(md_table)
    f.write("\n")





# %%
# %% [markdown]
# ## Grid search (paper-ready): LN, Bias, freeze_wv, freeze_wo at d_model=64, n_heads=1, n_layers=2

# %%
# import itertools as _it
# import matplotlib.cm as cm
# import matplotlib.colors as mcolors

# # Fixed architecture for this grid
# GRID_D_MODEL = 64
# GRID_N_HEADS = 1
# GRID_N_LAYERS = 2

# # Search axes
# grid_lns = [False, True]
# grid_biases = [False, True]
# grid_freeze_wv = [False, True]
# grid_freeze_wo = [False, True]

# n_runs = 30

# grid_rows = []

# print("\n=== Grid search: d_model=64, n_heads=1, n_layers=2 over LN/Bias/fWV/fWO ===")
# total_configs = len(grid_lns) * len(grid_biases) * len(grid_freeze_wv) * len(grid_freeze_wo)
# for cfg_idx, (ln, bias, fwv, fwo) in enumerate(_it.product(grid_lns, grid_biases, grid_freeze_wv, grid_freeze_wo), start=1):
#     # Base deterministic seed per config for reproducibility
#     base_seed = (int(ln) << 3) | (int(bias) << 2) | (int(fwv) << 1) | int(fwo)

#     run_accs = []
#     for run_idx in range(1, n_runs + 1):
#         # Vary seed per run to avoid identical learning dynamics
#         seed = base_seed * 100 + run_idx
#         torch.manual_seed(seed)
#         np.random.seed(seed)

#         model = make_model(
#             n_layers=GRID_N_LAYERS,
#             n_heads=GRID_N_HEADS,
#             d_model=GRID_D_MODEL,
#             ln=ln,
#             use_bias=bias,
#             freeze_wv=fwv,
#             freeze_wo=fwo,
#         )

#         # Train for 50k steps per run w/ early stopping at 99.9% acc
#         train(
#             model,
#             max_steps=50_000,
#             lr=LEARNING_RATE,
#             weight_decay=WEIGHT_DECAY,
#             early_stop_acc=0.999,
#         )
#         final_acc = float(accuracy(model, val_dl))
#         run_accs.append(final_acc)

#         # Progress print after every run in requested format
#         print(f"{cfg_idx}:{run_idx} / {total_configs}")

#     avg_acc = float(np.mean(run_accs)) if len(run_accs) else float('nan')
#     print(f"Config LN={ln}, Bias={bias}, fWV={fwv}, fWO={fwo} ==> Mean val acc over {n_runs} runs: {avg_acc:.4f}")

#     grid_rows.append({
#         'd_model': GRID_D_MODEL,
#         'n_heads': GRID_N_HEADS,
#         'n_layers': GRID_N_LAYERS,
#         'ln': ln,
#         'bias': bias,
#         'freeze_wv': fwv,
#         'freeze_wo': fwo,
#         'val_acc': round(avg_acc, 4),
#     })

# grid_df = pd.DataFrame(grid_rows).sort_values(['freeze_wv', 'freeze_wo', 'ln', 'bias']).reset_index(drop=True)

# # Print markdown table (copy-paste ready for paper)
# md_table = grid_df.to_markdown(index=False)
# print("\nGrid search results (markdown):\n")
# print(md_table)

# # Save artifacts for the paper
# os.makedirs('figs', exist_ok=True)
# os.makedirs('results', exist_ok=True)
# timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
# table_md_path = f"results/gridsearch_d{GRID_D_MODEL}_L{GRID_N_LAYERS}_H{GRID_N_HEADS}_{timestamp}.md"
# csv_path = f"results/gridsearch_d{GRID_D_MODEL}_L{GRID_N_LAYERS}_H{GRID_N_HEADS}_{timestamp}.csv"
# with open(table_md_path, 'w') as f:
#     f.write(f"# Grid search results (d={GRID_D_MODEL}, L={GRID_N_LAYERS}, H={GRID_N_HEADS})\n\n")
#     f.write(md_table)
# grid_df.to_csv(csv_path, index=False)
# print(f"Saved markdown table to {table_md_path} and CSV to {csv_path}")

# # Create a small figure panel: 2x2 subplots for (freeze_wv, freeze_wo), each a 2x2 heatmap over (ln, bias)
# fig, axes = plt.subplots(2, 2, figsize=(8, 7), constrained_layout=True)
# vmin, vmax = 0.0, 1.0
# _last_im = None

# def _get_acc(ln, bias, fwv, fwo):
#     row = grid_df[(grid_df['ln']==ln) & (grid_df['bias']==bias) & (grid_df['freeze_wv']==fwv) & (grid_df['freeze_wo']==fwo)]
#     return float(row['val_acc'].values[0]) if len(row) else float('nan')

# fwv_order = [True, False]  # row order
# fwo_order = [True, False]  # col order

# for i, fwv in enumerate(fwv_order):
#     for j, fwo in enumerate(fwo_order):
#         ax = axes[i, j]
#         # Build 2x2 matrix: rows = [no LN, LN], cols = [no Bias, Bias]
#         mat = np.array([
#             [_get_acc(False, False, fwv, fwo), _get_acc(False, True, fwv, fwo)],
#             [_get_acc(True,  False, fwv, fwo), _get_acc(True,  True, fwv, fwo)],
#         ])
#         im = ax.imshow(mat, cmap='viridis', vmin=vmin, vmax=vmax)
#         _last_im = im
#         # Annotate cells
#         for r in range(2):
#             for c in range(2):
#                 val = mat[r, c]
#                 ax.text(c, r, f"{val:.2f}", va='center', ha='center', color='white' if val < 0.6 else 'black', fontsize=10)
#         ax.set_xticks([0,1], labels=['No Bias', 'Bias'])
#         ax.set_yticks([0,1], labels=['No LN', 'LN'])
#         ax.set_title(f"freeze_wv={fwv}, freeze_wo={fwo}")

# # Single colorbar using ScalarMappable to avoid unbound issues
# sm = cm.ScalarMappable(cmap='viridis', norm=mcolors.Normalize(vmin=vmin, vmax=vmax))
# sm.set_array([])
# cbar = fig.colorbar(sm, ax=axes.ravel().tolist(), shrink=0.85)
# cbar.set_label('Validation accuracy')
# fig.suptitle(f"Grid: d={GRID_D_MODEL}, L={GRID_N_LAYERS}, H={GRID_N_HEADS}")

# fig_path_png = f"figs/gridsearch_d{GRID_D_MODEL}_L{GRID_N_LAYERS}_H{GRID_N_HEADS}_{timestamp}.png"
# fig_path_pdf = f"figs/gridsearch_d{GRID_D_MODEL}_L{GRID_N_LAYERS}_H{GRID_N_HEADS}_{timestamp}.pdf"
# fig.savefig(fig_path_png, dpi=300)
# fig.savefig(fig_path_pdf)
# plt.close(fig)
# print(f"Saved figure to {fig_path_png} and {fig_path_pdf}")

# # Quick summary: best config
# best_row = grid_df.iloc[grid_df['val_acc'].argmax()]
# print("\nBest config:")
# print(best_row.to_dict())


# %%
