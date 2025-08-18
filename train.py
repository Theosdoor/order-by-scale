# %% [markdown]
# ## Setup

# %%
import numpy as np
import torch
from torch.utils.data import DataLoader

import os
import copy

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import pandas as pd, itertools
from tqdm.auto import tqdm

from transformer_lens import HookedTransformer, HookedTransformerConfig, utils



# Configure plotly to use static rendering if widgets fail
import plotly.io as pio
pio.renderers.default = "notebook"

float_formatter = "{:.5f}".format
np.set_printoptions(formatter={'float_kind':float_formatter})


# %% [markdown]
# ## Model

# %%
# ---------- constants ----------
MODEL_NAME = 'v3_2layer_100dig_128d'
MODEL_PATH = "models/" + MODEL_NAME + ".pt"

DATASET_NAME = "listlen2_digits100_dupes_traindupesonly"
# listlen2_digits10_dupes
# listlen2_digits10_nodupes
# listlen2_digits100_dupes
# listlen2_digits100_nodupes

LIST_LEN = 2 # [d1, d2]
SEQ_LEN = LIST_LEN * 2 + 1 # [d1, d2, SEP, o1, o2]

N_DIGITS = 100
DIGITS = list(range(N_DIGITS)) # 100 digits from 0 to 99
PAD = N_DIGITS # special padding token
SEP = N_DIGITS + 1 # special seperator token for the model to think about the input (+1 to avoid confusion with the last digit)
VOCAB = len(DIGITS) + 2  # + the special tokens

D_MODEL = 128
N_HEAD = 1 # 1
N_LAYER = 2 # 2
USE_LN = False # use layer norm in model
USE_BIAS = False # use bias in model
FREEZE_WV = True # no value matrix in attn 
FREEZE_WO = True # no output matrix in attn (i.e. attn head can only copy inputs to outputs)

LEARNING_RATE = 1e-3 # default 1e-3
WEIGHT_DECAY = 0.01 # default 0.01
MAX_TRAIN_STEPS = 500_000 # max training steps
USE_CHECKPOINTING = True # whether to use checkpointing for training
TRACK_ATTN_FLOW = False # whether to track attention flow during training

DEV = (
    "cuda"
    if torch.cuda.is_available()
    else ("mps" if torch.backends.mps.is_available() else "cpu")
)
device = DEV
torch.manual_seed(0)

# ---------- mask ----------
# attention mask for [d1, d2, SEP, o1, o2] looks like this (query rows are horizontal, key columns are vertical):
# -    d1    d2    SEP    o1    o2   (keys)
# d1  -inf  -inf   -inf  -inf  -inf
# d2   0    -inf   -inf  -inf  -inf
# SEP  0      0    -inf  -inf  -inf
# o1  -inf  -inf    0    -inf   -inf
# o2  -inf  -inf    0      0    -inf
# (queries)

mask_bias = torch.triu(torch.ones(SEQ_LEN, SEQ_LEN) * float("-inf")) # upper triangular bias mask (lead_diag & above = -inf, rest = 0)
mask_bias[0, 0] = 0. # don't want a full row of -inf! otherwise we get nan erros & training breaks
mask_bias[LIST_LEN+1:, :LIST_LEN] = float("-inf") # stop output tokens from attending to input tokens
mask_bias = mask_bias.unsqueeze(0).unsqueeze(0) # (1,1,T,T) broadcastable across batch and heads

print(mask_bias.cpu()[0][0])


# %%
# ---------- data ----------
DATASET_PATH = f"data/{DATASET_NAME}.pt"

if not os.path.exists(DATASET_PATH):
    raise FileNotFoundError(f"Dataset not found at {DATASET_PATH}. Please run data.py to generate it.")

saved_data = torch.load(DATASET_PATH, weights_only=False)
train_ds = saved_data['train']
val_ds = saved_data['val']

train_batch_size = min(128, len(train_ds))
val_batch_size = min(256, len(val_ds))
train_dl = DataLoader(train_ds, train_batch_size, shuffle=True, drop_last=True)
val_dl = DataLoader(val_ds, val_batch_size, drop_last=False)

print("Input:", train_ds[0][0])
print("Target:", train_ds[0][1])
print(f"Train dataset size: {len(train_ds)}, Validation dataset size: {len(val_ds)}")

# Sanity check: dataset token range must fit VOCAB
with torch.no_grad():
    in_max = train_ds.tensors[0].max().item()
    tgt_max = train_ds.tensors[1].max().item()
    ds_max = max(in_max, tgt_max)
if ds_max >= VOCAB:
    raise ValueError(
        f"Dataset contains token id {ds_max} but model VOCAB={VOCAB}. "
        f"Mismatch: did you set N_DIGITS to {ds_max-1} (PAD={ds_max-1}, SEP={ds_max})?"
    )

# %%


# %%
# ---------- config helper ----------
def attach_custom_mask(model):
    def _mask(scores, hook=None):
        # scores: (batch, heads, Q, K)
        return scores + mask_bias.to(scores.device)
    
    # register the same mask hook on every layer
    for block in model.blocks:
        block.attn.hook_attn_scores.add_perma_hook(_mask, dir="fwd")


def strip_bias(m):
    for mod in m.modules():
        if hasattr(mod, "bias") and mod.bias is not None:
            mod.bias.requires_grad_(False)
            torch.nn.init.zeros_(mod.bias)
            print(mod)

    # remove biases from attention layers
    attn_biases = ['b_Q', 'b_K', 'b_V', 'b_O']
    for block in m.blocks:
        for b in attn_biases:
            mod = getattr(block.attn, b, None)
            if mod is not None:
                mod.requires_grad_(False)
                torch.nn.init.zeros_(mod)

    # remove unembed bias
    if hasattr(m, "unembed") and m.b_U is not None:
        m.unembed.b_U.requires_grad_(False)
        torch.nn.init.zeros_(m.unembed.b_U)

def set_WV_identity_and_freeze(model, d_model):
    with torch.no_grad():
        # Create a stack of identity-like matrices for W_V
        # Each matrix is of shape (d_model, d_head)
        # We take the first d_head columns of the d_model x d_model identity matrix
        identity_slice = torch.eye(d_model, model.cfg.d_head)
        # Repeat for each head
        W_V_identity = identity_slice.unsqueeze(0).repeat(model.cfg.n_heads, 1, 1)
        
        for block in model.blocks:
            block.attn.W_V.copy_(W_V_identity)
            block.attn.W_V.requires_grad = False

def set_WO_identity_and_freeze(model, d_model):
    with torch.no_grad():
        # Create a stack of identity-like matrices for W_O
        # Each matrix is of shape (d_head, d_model)
        # We take the first d_head rows of the d_model x d_model identity matrix
        identity_slice = torch.eye(model.cfg.d_head, d_model)
        # Repeat for each head
        W_O_identity = identity_slice.unsqueeze(0).repeat(model.cfg.n_heads, 1, 1)

        for block in model.blocks:
            block.attn.W_O.copy_(W_O_identity)
            block.attn.W_O.requires_grad = False


def make_model(n_layers=N_LAYER, n_heads=N_HEAD, d_model=D_MODEL, ln=USE_LN, use_bias=USE_BIAS, freeze_wv=FREEZE_WV, freeze_wo=FREEZE_WO):
    cfg = HookedTransformerConfig(
        n_layers = n_layers,
        n_heads = n_heads,
        d_model = d_model,
        d_head = d_model//n_heads,
        n_ctx=SEQ_LEN,
        d_vocab=VOCAB,
        attn_only=True, # no MLP!
        normalization_type=("LN" if ln else None),
    )
    model = HookedTransformer(cfg).to(DEV)
    if freeze_wv:
        set_WV_identity_and_freeze(model, d_model)
    if freeze_wo:
        set_WO_identity_and_freeze(model, d_model)
    if not use_bias:
        strip_bias(model)
    
    attach_custom_mask(model)
    return model

# %%
# ----- Model saving / loading helpers ------
def save_model(model, path = MODEL_PATH):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")

def load_model(path = MODEL_PATH, device = DEV):
    print("Loading model from", path)
    model = make_model()
    model.load_state_dict(
        torch.load(path, map_location=device)
    )  # map weights to target device
    model.eval()
    return model

# %%

# NEW: save an attention flow figure for a single example
def save_attention_flow_figure(m, example_input, out_path, title="Attention Flow", threshold=0.05):
    m.eval()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    with torch.no_grad():
        _, cache = m.run_with_cache(example_input.to(DEV), return_type="logits")

    # Collect attention patterns per layer -> [L, Q, K]
    att = (
        torch.stack(
            [cache[f"blocks.{layer}.attn.hook_pattern"] for layer in range(m.cfg.n_layers)],
            dim=0,
        )
        .cpu()
        .numpy()
        .squeeze()
    )

    # Residual stream (embed + post-resid after each layer)
    resid_keys = ["hook_embed"] + [f"blocks.{l}.hook_resid_post" for l in range(m.cfg.n_layers)]
    resid_values = torch.stack([cache[k] for k in resid_keys], dim=0)  # [L+1, 1, seq, d_model]

    # Get W_U (compatibly)
    W_U = getattr(m, "W_U", m.unembed.W_U)

    # Logit lens: decode most likely token at each position after each layer
    position_tokens = (resid_values @ W_U).squeeze(1).argmax(-1)  # [L+1, seq]

    L, N, _ = att.shape
    x_positions = np.arange(L + 1)  # input + after each layer
    y_positions = np.arange(N)[::-1]  # top token = index 0

    fig, ax = plt.subplots(figsize=(8, 6))

    # Draw nodes and token labels at each layer column
    for lx in range(L + 1):
        xs = np.full(N, lx)
        ax.scatter(xs, y_positions, s=50)
        for i, y in enumerate(y_positions):
            ax.text(
                lx + 0.03,
                y + 0.03,
                str(position_tokens[lx][i].item()),
                fontsize=10,
                va="bottom",
                ha="left",
            )

    # Horizontal dashed arrows between all dots in the same row (residual stream)
    for lx in range(L):
        for y in y_positions:
            ax.annotate(
                "",
                xy=(lx + 1, y),
                xytext=(lx, y),
                arrowprops=dict(
                    arrowstyle="->", lw=1.2, linestyle="--", color="gray", alpha=0.6
                ),
                zorder=0,
            )

    # Attention edges from layer l (keys) to layer l+1 (queries)
    for l in range(L):
        for q in range(N):
            for k in range(N):
                w = att[l, q, k]
                if w <= threshold:
                    continue
                x0, y0 = l, y_positions[k]
                x1, y1 = l + 1, y_positions[q]
                ax.annotate(
                    "",
                    xy=(x1, y1),
                    xytext=(x0, y0),
                    arrowprops=dict(arrowstyle="->", lw=1 + 4 * w, alpha=w),
                )

    # Labels and layout
    ax.set_xticks(x_positions)
    ax.set_xticklabels(["Input"] + [f"After L{l+1}" for l in range(m.cfg.n_layers)])
    ax.set_yticks(y_positions)
    position_names = ["d1", "d2", "SEP", "o1", "o2"]
    ax.set_yticklabels(position_names)
    ax.set_xlim(-0.5, L + 0.5)
    ax.set_ylim(-0.5, N - 0.5)
    ax.set_title(title)
    ax.grid(False)
    ax.set_aspect("auto")
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Legend
    legend_elements = [
        Line2D([0], [0], linestyle="--", color="gray", lw=1.5, label="Residual stream (dotted)"),
        Line2D([0], [0], linestyle="-", color="black", lw=1.5, label="Attention (solid)"),
    ]
    ax.legend(
        handles=legend_elements,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.18),
        frameon=False,
        ncol=2,
    )

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

# %%
# ---------- utilities ----------
def accuracy(m):
    m.eval()
    hits = tots = 0
    with torch.no_grad():
        for inputs, targets in val_dl:
            logits = m(inputs.to(DEV))[:, LIST_LEN+1:]  # (batch, 2, vocab)
            preds = logits.argmax(-1)
            hits += (preds == targets[:, LIST_LEN+1:].to(DEV)).sum().item()
            tots += preds.numel()
    return hits / tots


# def train(m, max_steps=10_000, early_stop_acc=0.999, checkpoints=False, weight_decay=WEIGHT_DECAY, verbose=True):
#     opt = torch.optim.AdamW(m.parameters(), 1e-3, weight_decay=weight_decay)
#     ce = torch.nn.CrossEntropyLoss()
#     dl = itertools.cycle(train_dl)  # infinite iterator
#     for step in tqdm(range(max_steps), desc="Training"):
#         inputs, targets = next(dl)
#         # get logits/loss for output tokens only
#         logits = m(inputs.to(DEV))[:, LIST_LEN+1:].reshape(-1, VOCAB) 
#         loss = ce(logits, targets[:, LIST_LEN+1:].reshape(-1).to(DEV))
#         loss.backward()
#         opt.step()
#         opt.zero_grad()
#         if (step + 1) % 100 == 0:
#             acc = accuracy(m)
#             if acc >= early_stop_acc:
#                 print(f"Early stopping at step {step + 1} with accuracy {acc:.2%} >= {early_stop_acc:.2%}")
#                 break
#             update_every = max(min(10_000, 0.05*max_steps), 1000)
#             if verbose and (step+1) % update_every == 0:
#                 print(f"Step {step + 1}, Loss: {loss.item():.4f}, Accuracy: {acc:.2%}")
#             if checkpoints and (step+1) % 50_000 == 0:
#                 save_model(m, MODEL_PATH)
            
#     print(f"Final accuracy: {accuracy(m):.2%}")


def train(m, max_steps=10_000, early_stop_acc=0.999, checkpoints=False, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY, verbose=True,
          vis_every=None, vis_dir=None, vis_example_idx=0):
    opt = torch.optim.AdamW(m.parameters(), lr, weight_decay=weight_decay)
    ce = torch.nn.CrossEntropyLoss()
    dl = itertools.cycle(train_dl)  # infinite iterator

    # Setup visualization defaults
    if vis_every is not None and vis_every > 0:
        if vis_dir is None:
            # Save under artifacts/attn_flow/<MODEL_NAME>/
            vis_dir = os.path.join("artifacts", "attn_flow", MODEL_NAME)
        os.makedirs(vis_dir, exist_ok=True)
        # Fixed example from validation set
        example_input = val_ds.tensors[0][vis_example_idx].unsqueeze(0).to(DEV)

        # Save an initial snapshot at step 0
        save_attention_flow_figure(
            m, example_input,
            out_path=os.path.join(vis_dir, f"step_000000.png"),
            title=f"Attention Flow (Step 0)"
        )

    last_saved_step = -1

    for step in tqdm(range(max_steps), desc="Training"):
        inputs, targets = next(dl)
        # get logits/loss for output tokens only
        logits = m(inputs.to(DEV))[:, LIST_LEN+1:].reshape(-1, VOCAB) 
        loss = ce(logits, targets[:, LIST_LEN+1:].reshape(-1).to(DEV))
        loss.backward()
        opt.step()
        opt.zero_grad()

        # Periodic eval/log
        if (step + 1) % 100 == 0:
            acc = accuracy(m)
            if acc >= early_stop_acc:
                print(f"Early stopping at step {step + 1} with accuracy {acc:.2%} >= {early_stop_acc:.2%}")
                # Save a final snapshot before breaking
                if vis_every is not None and vis_every > 0:
                    save_attention_flow_figure(
                        m, example_input,
                        out_path=os.path.join(vis_dir, f"step_{step+1:06d}.png"),
                        title=f"Attention Flow (Step {step+1})"
                    )
                break
            update_every = max(min(10_000, 0.05*max_steps), 1000)
            if verbose and (step+1) % update_every == 0:
                print(f"Step {step + 1}, Loss: {loss.item():.4f}, Accuracy: {acc:.2%}")
            if checkpoints and (step+1) % 50_000 == 0:
                save_model(m, MODEL_PATH)

        # Periodic attention flow snapshots
        if TRACK_ATTN_FLOW and vis_every is not None and vis_every > 0 and ((step + 1) % vis_every == 0):
            save_attention_flow_figure(
                m, example_input,
                out_path=os.path.join(vis_dir, f"step_{step+1:06d}.png"),
                title=f"Attention Flow (Step {step+1})"
            )
            last_saved_step = step + 1
            
    # Final accuracy print and final snapshot if not already saved at this step
    print(f"Final accuracy: {accuracy(m):.2%}")
    if vis_every is not None and vis_every > 0 and last_saved_step != max_steps:
        # Save final snapshot at max_steps (or last step reached)
        save_attention_flow_figure(
            m, example_input,
            out_path=os.path.join(vis_dir, f"step_{min(last_saved_step, max_steps):06d}_final.png"),
            title=f"Attention Flow (Final)"
        )
# ...existing code...


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
    # {'name': 'd128', 'd_model': 128, 'weight_decay': 1.0},
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

from itertools import product
# specs = []
# d_model = 128
# for n_layers, ln, use_bias, freeze_wv, freeze_wo in product(
#     [2, 3],            # layers
#     [False, True],     # ln
#     [False, True],     # use_bias
#     [False, True],     # freeze_wv
#     [False, True],     # freeze_wo
# ):
#     specs.append({
#         "name": make_name(d_model, n_layers, ln, use_bias, freeze_wv, freeze_wo),
#         "d_model": d_model,
#         "n_layers": n_layers,
#         "ln": ln,
#         "use_bias": use_bias,
#         "freeze_wv": freeze_wv,
#         "freeze_wo": freeze_wo,
#     })

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

    train(model, max_steps=50_000, lr=full_spec['lr'], weight_decay=full_spec['weight_decay'], verbose=True)
    
    # Add all spec parameters to the results
    result = full_spec.copy()
    result['val_acc'] = round(accuracy(model), 4)
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
# | name                        |   n_layers |   n_heads |   d_model | ln    | use_bias   | freeze_wv   | freeze_wo   |   weight_decay |   val_acc |
# |:----------------------------|-----------:|----------:|----------:|:------|:-----------|:------------|:------------|---------------:|----------:|
# | d128_2L_noLN_noBias_uWV_uWO |          2 |         1 |       128 | False | False      | False       | False       |           0.01 |    0.4625 |
# | d128_2L_noLN_noBias_uWV_fWO |          2 |         1 |       128 | False | False      | False       | True        |           0.01 |    0.4895 |
# | d128_2L_noLN_noBias_fWV_uWO |          2 |         1 |       128 | False | False      | True        | False       |           0.01 |    0.463  |
# | d128_2L_noLN_noBias_fWV_fWO |          2 |         1 |       128 | False | False      | True        | True        |           0.01 |    0.9173 |
# | d128_2L_noLN_Bias_uWV_uWO   |          2 |         1 |       128 | False | True       | False       | False       |           0.01 |    0.868  |
# | d128_2L_noLN_Bias_uWV_fWO   |          2 |         1 |       128 | False | True       | False       | True        |           0.01 |    0.8945 |
# | d128_2L_noLN_Bias_fWV_uWO   |          2 |         1 |       128 | False | True       | True        | False       |           0.01 |    0.4645 |
# | d128_2L_noLN_Bias_fWV_fWO   |          2 |         1 |       128 | False | True       | True        | True        |           0.01 |    0.9183 |
# | d128_2L_LN_noBias_uWV_uWO   |          2 |         1 |       128 | True  | False      | False       | False       |           0.01 |    0.4743 |
# | d128_2L_LN_noBias_uWV_fWO   |          2 |         1 |       128 | True  | False      | False       | True        |           0.01 |    0.4607 |
# | d128_2L_LN_noBias_fWV_uWO   |          2 |         1 |       128 | True  | False      | True        | False       |           0.01 |    0.4632 |
# | d128_2L_LN_noBias_fWV_fWO   |          2 |         1 |       128 | True  | False      | True        | True        |           0.01 |    0.4485 |
# | d128_2L_LN_Bias_uWV_uWO     |          2 |         1 |       128 | True  | True       | False       | False       |           0.01 |    0.4733 |
# | d128_2L_LN_Bias_uWV_fWO     |          2 |         1 |       128 | True  | True       | False       | True        |           0.01 |    0.4647 |
# | d128_2L_LN_Bias_fWV_uWO     |          2 |         1 |       128 | True  | True       | True        | False       |           0.01 |    0.4755 |
# | d128_2L_LN_Bias_fWV_fWO     |          2 |         1 |       128 | True  | True       | True        | True        |           0.01 |    0.4602 |
# 
# | name   |   n_layers |   n_heads |   d_model | ln    | use_bias   | freeze_wv   | freeze_wo   |   weight_decay |   val_acc |
# |:-------|-----------:|----------:|----------:|:------|:-----------|:------------|:------------|---------------:|----------:|
# | d256   |          2 |         1 |       256 | False | False  | True        | True        |           0.01 |    0.8697 |
# | d128   |          2 |         1 |       128 | False | False      | True        | True        |           0.01 |    0.9038 |
# | d64    |          2 |         1 |        64 | False | False      | True        | True        |           0.01 |    0.6836 |
# | d32    |          2 |         1 |        32 | False | False      | True        | True        |           0.01 |    0.4278 |
# | d16    |          2 |         1 |        16 | False | False      | True        | True        |           0.01 |    0.4497 |

# %%
# LOAD existing or train and SAVE new model
load_existing = True  # Set to False to always train a new model

if os.path.exists(MODEL_PATH) and load_existing:
    model = load_model(MODEL_PATH, device=DEV)
else:
    if os.path.exists(MODEL_PATH):
        MODEL_PATH = MODEL_PATH.replace(".pt", "_new.pt")
        print(f"Model path already exists. Saving new model to {MODEL_PATH}")
    print("Training model")
    model = make_model()
    train(model, max_steps=MAX_TRAIN_STEPS, early_stop_acc=0.999, 
          checkpoints=USE_CHECKPOINTING, vis_every=10000, vis_example_idx=0
          )
    save_model(model, MODEL_PATH)

# from torchinfo import summary
# summary(model) 

# %%
# --- Model Parameters Overview ---
print("--- Overview of Model Parameters ---")   
total_params = 0
trainable_params = 0

# Use a formatted string for better alignment
print(f"{'Parameter Name':<40} | {'Shape':<20} | {'Trainable':<10}")
print("-" * 80)

for name, param in model.named_parameters():
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



