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

import umap

import einops
import pandas as pd, itertools
from tqdm.auto import tqdm

from transformer_lens import HookedTransformer, HookedTransformerConfig, utils

from model_utils import (
    configure_runtime,
    build_attention_mask,
    make_model,
    load_model,
    accuracy,
    parse_model_name,
)
from data import get_dataset

# Configure plotly to use static rendering if widgets fail
import plotly.io as pio
pio.renderers.default = "notebook"

float_formatter = "{:.5f}".format
np.set_printoptions(formatter={'float_kind':float_formatter})


# %% [markdown]
# ## Model

# %%
# ---------- parameters ----------
MODEL_NAME = '2layer_100dig_64d' 
MODEL_PATH = "models/" + MODEL_NAME + ".pt"

# Derive architecture parameters from name
try:
    N_DIGITS, D_MODEL, N_LAYER = parse_model_name(MODEL_NAME)
except ValueError as e:
    print(f"[parse_model_name warning] {e}. Falling back to manual defaults.")
    N_DIGITS, D_MODEL, N_LAYER = 100, 64, 2

LIST_LEN = 2  # [d1, d2]
SEQ_LEN = LIST_LEN * 2 + 1  # [d1, d2, SEP, o1, o2]

DIGITS = list(range(N_DIGITS))  # 0 .. N_DIGITS-1
MASK = N_DIGITS  # special masking token for o1 and o2
SEP = N_DIGITS + 1  # special separator token
VOCAB = len(DIGITS) + 2  # + the special tokens

N_HEAD = 1
USE_LN = False # use layer norm in model
USE_BIAS = False # use bias in model
FREEZE_WV = True # no value matrix in attn 
FREEZE_WO = True # no output matrix in attn (i.e. attn head can only copy inputs to outputs)


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
# attention mask for [d1, d2, SEP, o1, o2] looks like this (query rows are horizontal, key columns are vertical):
# -    d1    d2    SEP    o1    o2   (keys)
# d1  -inf  -inf   -inf  -inf  -inf
# d2   0    -inf   -inf  -inf  -inf
# SEP  0      0    -inf  -inf  -inf
# o1  -inf  -inf    0    -inf   -inf
# o2  -inf  -inf    0      0    -inf
# (queries)

mask_bias, _mask_bias_l0 = build_attention_mask()
# print(mask_bias.cpu()[0][0])

# %%
# ---------- data ----------
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
# Check train set
train_ds[:5]

# %%
# LOAD existing model
if os.path.exists(MODEL_PATH):
    model = load_model(
        MODEL_PATH,
        n_layers=N_LAYER,
        n_heads=N_HEAD,
        d_model=D_MODEL,
        ln=USE_LN,
        use_bias=USE_BIAS,
        freeze_wv=FREEZE_WV,
        freeze_wo=FREEZE_WO,
        device=DEV,
    )
else:
    raise FileNotFoundError(f"Model file {MODEL_PATH} does not exist. Please train the model first.")

accuracy(model, val_dl)







# %%
# --- Model Parameters Overview ---

print("--- Overview of TRAINABLE Model Parameters ---")
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

# %% [markdown]
# ### Model attention

# %% [markdown]
# We confirm below that the model does not leak attention onto the first two tokens, which are the inputs to the task. The model should only attend to the first two tokens when predicting the third token, and not attend to them at all when predicting the fourth and fifth tokens.

# %%
# --- Using Plotly for visualization ---

def check_attention(m, dataloader, eps=1e-3):
    for inputs, _ in dataloader:
        with torch.no_grad():
            _, cache = m.run_with_cache(inputs.to(DEV))
        for l in range(m.cfg.n_layers):
            pat = cache["pattern", l][:, 0]  # (batch, Q, K)
            leak = pat[:, LIST_LEN+1:, :LIST_LEN].sum(dim=-1)  # mass on forbidden keys
            if (leak > eps).any():
                raise ValueError(f"❌ Layer {l}: output tokens attend to x₁/x₂ by >{eps:.0e}")
    print("✅ no attention leakage onto x₁/x₂")


sample = val_ds[1][0] # Example input sequence
print(f"Sample sequence: {sample.cpu().numpy()}")  # Print the sample sequence for reference
_, cache = model.run_with_cache(sample.unsqueeze(0).to(DEV))

# --- Create Plotly visualization ---
token_labels = [f'd{i+1}' for i in range(LIST_LEN)] + ['SEP'] + [f'o{i+1}' for i in range(LIST_LEN)]
subplot_titles = [f"Layer {l} Attention Pattern" for l in range(model.cfg.n_layers)]

fig = make_subplots(
    rows=1, 
    cols=model.cfg.n_layers, 
    subplot_titles=subplot_titles,
    horizontal_spacing=0.08 # Add spacing between plots
)

for l in range(model.cfg.n_layers):
    pat = cache["pattern", l][0, 0].cpu().numpy()
    
    fig.add_trace(
        go.Heatmap(
            z=pat,
            x=token_labels,
            y=token_labels,
            colorscale="Viridis",
            zmin=0,
            zmax=1,
            showscale=(l == model.cfg.n_layers - 1) # Show colorbar only for the last plot
        ),
        row=1, col=l+1
    )

fig.update_layout(
    title_text="Attention Patterns for a Sample Sequence",
    width=850,
    height=450,
    template="plotly_white"
)

# Apply settings to all axes
fig.update_xaxes(title_text="Key Position")
fig.update_yaxes(title_text="Query Position", autorange='reversed')

fig.show()

check_attention(model, val_dl)

# %%
# --- Mean Attention Patterns ---

all_pats = [[] for _ in range(model.cfg.n_layers)]
for inputs, _ in val_dl:
    with torch.no_grad():
        _, cache = model.run_with_cache(inputs.to(DEV))
    for l in range(model.cfg.n_layers):
        pat = cache["pattern", l][:, 0]  # (batch, Q, K)
        all_pats[l].append(pat)
all_pats = [torch.cat(pats, dim=0) for pats in all_pats]

for l, pats in enumerate(all_pats):
    identical = torch.allclose(pats, pats[0].expand_as(pats))
    print(f"Layer {l}: all attention patterns identical? {'✅' if identical else '❌'}")

with torch.no_grad():
    avg_pats = [
        torch.zeros(SEQ_LEN, SEQ_LEN, device=DEV) for _ in range(model.cfg.n_layers)
    ]
    n = 0
    for inputs, _ in val_dl:
        _, cache = model.run_with_cache(inputs.to(DEV))
        for l in range(model.cfg.n_layers):
            avg_pats[l] += cache["pattern", l][:, 0].sum(0)
        n += inputs.shape[0]
    avg_pats = [p / n for p in avg_pats]

# --- Visualize Average Attention Patterns ---
token_labels = [f'd{i+1}' for i in range(LIST_LEN)] + ['SEP'] + [f'o{i+1}' for i in range(LIST_LEN)]
subplot_titles = [f"Layer {l} Average Attention" for l in range(model.cfg.n_layers)]

fig = make_subplots(
    rows=1, 
    cols=model.cfg.n_layers, 
    subplot_titles=subplot_titles,
    horizontal_spacing=0.08
)

for l in range(model.cfg.n_layers):
    avg_pat_np = avg_pats[l].cpu().numpy()
    
    fig.add_trace(
        go.Heatmap(
            z=avg_pat_np,
            x=token_labels,
            y=token_labels,
            colorscale="Viridis",
            zmin=0,
            zmax=1,
            showscale=(l == model.cfg.n_layers - 1) # Show colorbar only for the last plot
        ),
        row=1, col=l+1
    )

fig.update_layout(
    title_text="Average Attention Patterns Across Validation Set",
    width=850,
    height=450,
    template="plotly_white"
)
fig.update_xaxes(title_text="Key Position")
fig.update_yaxes(title_text="Query Position", autorange='reversed')
fig.show()


# Create a deep copy of the model to avoid modifying the original
model_with_avg_attn = copy.deepcopy(model)

def mk_hook(avg):
    logits = (avg + 1e-12).log()  # log-prob so softmax≈avg, ε avoids -∞

    def f(scores, hook):
        return logits.unsqueeze(0).unsqueeze(0).expand_as(scores)

    return f

for l in range(model_with_avg_attn.cfg.n_layers):
    model_with_avg_attn.blocks[l].attn.hook_attn_scores.add_hook(
        mk_hook(avg_pats[l]), dir="fwd"
    )

print("Accuracy with avg-attn:", accuracy(model_with_avg_attn, val_dl))

# %% [markdown]
# ## Interp

# %%
# --- Setup ---
head_index_to_ablate = 0 # fixed

# Define the loss function
loss_fn = torch.nn.CrossEntropyLoss()

# Check loss on validation set
val_inputs = val_ds.tensors[0].to(DEV)
val_targets = val_ds.tensors[1].to(DEV)
sample_idx = 0  # Use the xth sample in the validation set for comparing predictions
sample_list = val_inputs[sample_idx].cpu().numpy()

# --- Calculate Original Loss on last 2 digits ---
with torch.no_grad():
    original_logits, cache = model.run_with_cache(val_inputs, return_type="logits")
    output_logits = original_logits[:, LIST_LEN+1:] # Slice to get logits for the last two positions
    output_targets = val_targets[:, LIST_LEN+1:] # Slice to get the target tokens
    
    original_loss = loss_fn(output_logits.reshape(-1, VOCAB), output_targets.reshape(-1)) # Calculate the loss
    # Calculate accuracy
    original_predictions = original_logits.argmax(dim=-1) 
    original_output_predictions = original_predictions[:, LIST_LEN+1:]
    original_accuracy = (original_output_predictions == output_targets).float().mean()

print(f"Original loss: {original_loss.item()}")
print(f"Original accuracy: {original_accuracy.item()}")
print(f"Sample sequence {sample_idx}: {sample_list}")

# %%
from matplotlib.patches import FancyArrowPatch
import matplotlib.patheffects as pe
from matplotlib.path import Path as MplPath  # NEW: inspect curve type

# Use a single validation example (sample_idx is defined earlier)
example = val_inputs[sample_idx].unsqueeze(0).to(DEV)

# Run and cache activations
_, cache = model.run_with_cache(example, return_type="logits")

# Collect attention patterns per layer -> shape [L, Q, K]
att = (
    torch.stack(
        [cache[f"blocks.{layer}.attn.hook_pattern"] for layer in range(model.cfg.n_layers)],
        dim=0,
    )
    .cpu()
    .numpy()
    .squeeze()
)

# prune arrows (these ones don't have any effect on the output)
att[0][:2] = 0. * att[0][:2]
att[1][:3] = 0. * att[1][:3]
# ablate = {
#     # 0: [(4, 2), (3, 2), (4, 3), (0, 0), (1, 0)],
#     # 1: [(3, 2), (4, 3), (0, 0), (1, 0), (2, 1)],
#     # 2: [(0, 0), (1, 0), (2, 0), (2, 1), (3, 0), (4, 0), (4, 1), (4, 2), (4, 3)],
# }

# # Vectorized assignment using numpy indexing (robust for single/multiple pairs)
# for layer, pairs in ablate.items():
#     if not pairs:
#         continue
#     arr = np.array(pairs, dtype=int)  # shape (n_pairs, 2)
#     qs = arr[:, 0]
#     ks = arr[:, 1]
#     att[layer, qs, ks] = 0.0

# Collect residual stream (embed + post-resid after each layer)
resid_keys = ["hook_embed"] + [f"blocks.{l}.hook_resid_post" for l in range(model.cfg.n_layers)]
resid_values = torch.stack([cache[k] for k in resid_keys], dim=0)  # [L+1, 1, seq, d_model]

# Get W_U (compatibly)
W_U = getattr(model, "W_U", model.unembed.W_U)

# Logit lens: decode most likely token at each position after each layer
position_tokens = (resid_values @ W_U).squeeze(1).argmax(-1)  # [L+1, seq]

L, N, _ = att.shape


# --- Pretty paper figure: Attention flow (with red translucent arrows + Δacc labels) ---

# Layout
x_positions = np.arange(L + 1)  # columns: input + after each layer
y_positions = np.arange(N)[::-1]  # top token index = 0

# Styling for publication (vector-safe fonts for PDF/SVG)
plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

fig, ax = plt.subplots(figsize=(6.8, 4.2), dpi=300)

# Node roles and colors
position_names = ["d1", "d2", "SEP", "o1", "o2"]
roles = ["input", "input", "sep", "output", "output"]
role_colors = {"input": "#4C78A8", "sep": "#F58518", "output": "#54A24B"}
node_colors = [role_colors[r] for r in roles]

# Draw nodes and labels (keep these on top of arrows)
for lx in range(L + 1):
    xs = np.full(N, lx)
    ax.scatter(xs, y_positions, s=180, c=node_colors,
               edgecolor="black", linewidth=0.6, zorder=4)
    # left: position label (only in first col); right: decoded token id (every col)
    for i, y in enumerate(y_positions):
        if lx == 0:
            ax.text(lx - 0.14, y, position_names[i], va="center", ha="right",
                    fontsize=9, color="#334155", zorder=5)
        ax.text(lx + 0.14, y, str(position_tokens[lx, i].item()),
                va="center", ha="left", fontsize=9, fontweight="bold",
                color="black", zorder=5)

# Residual stream (dotted straight arrows)
for lx in range(L):
    for y in y_positions:
        arrow = FancyArrowPatch((lx, y), (lx + 1, y),
                                arrowstyle="-", mutation_scale=8,
                                lw=1.0, linestyle=(0, (2, 2)), color="#94A3B8",
                                alpha=0.7, zorder=1, clip_on=False)
        ax.add_patch(arrow)

# --- Per-edge Δaccuracy placeholders ---
# Fill this dict with your measured changes in percentage points (pp).
# Key: (layer, query_idx, key_idx) where layer is 0-indexed.
# Example: delta_acc_pp[(0, 2, 0)] = -1.7  # ablating L0: q=SEP, k=d1 lowers acc by 1.7 pp
delta_acc_pp = {
    # (l, q, k): value_in_pp,
    (0, 2, 0): -87.4, # sep --> d1
    (0, 2, 1): -74.3, # sep --> d2
    (0, 3, 2): -0.2, # o1 --> sep
    (0, 4, 2): -0.1, # o2 --> sep
    (1, 3, 2): -48.6, # o1 --> sep (L1)
    (1, 4, 2): -42.4, # o2 --> sep (L1)
    (1, 4, 3): -37.8, # o2 --> o1 (L1)
    
    # (0, 2, 0): -1.3,
    # (0, 2, 1): -49.6,
    # (1, 2, 0): -49.6,
    # (1, 4, 2): -49.5,
    # (2, 3, 2): -49.5,
}

def format_delta_pp(val):
    # Less obtrusive: short text, no "Δacc:" prefix
    if val is None:
        return "—"
    sign = "+" if val >= 0 else "−"
    return f"{sign}{abs(val):.1f}%"

# Attention edges (curved; width/alpha ~ weight)
threshold = 0.05  # ignore tiny weights
arrow_color = "#DC2626"   # red
arrow_alpha = 0.35        # translucent to avoid obscuring text

# Label controls
label_threshold = 0.04     # only label edges above this weight
show_placeholder = False   # set True to show "—" for missing entries
label_offset = 0.12        # distance of label from the edge midpoint

CURVE_STRENGTH = 0.0  # try 0.04–0.08; set 0.0 if you want perfectly straight


def edge_style(w):
    lw = 0.6 + 2.0 * np.sqrt(float(w))
    alpha = arrow_alpha
    return lw, alpha

def angle_in_display(ax, x0, y0, x1, y1):
    # Compute angle in screen space so rotation matches visual slope despite axis scales
    X0, Y0 = ax.transData.transform((x0, y0))
    X1, Y1 = ax.transData.transform((x1, y1))
    return np.degrees(np.arctan2(Y1 - Y0, X1 - X0))

for l in range(L):
    for q in range(N):
        for k in range(N):
            w = att[l, q, k]
            if w <= threshold:
                continue

            x0, y0 = l,      y_positions[k]
            x1, y1 = l + 1, y_positions[q]
            dy = y1 - y0

            # Curvature and style
            rad = np.sign(dy) * CURVE_STRENGTH * (min(abs(dy), 2) / 2.0)
            lw, alpha = edge_style(w)

            arrow = FancyArrowPatch(
                (x0, y0), (x1, y1),
                connectionstyle=f"arc3,rad={rad}",
                arrowstyle="->", mutation_scale=8,
                lw=lw, color=arrow_color, alpha=alpha,
                zorder=2, shrinkA=8, shrinkB=8,
                joinstyle="round", capstyle="round",
                clip_on=False,
            )
            ax.add_patch(arrow)

            # Check if a label should be drawn
            delta_val = delta_acc_pp.get((l, q, k))
            if (delta_val is None and not show_placeholder) or (w < label_threshold):
                continue
            
            label_text = format_delta_pp(delta_val)

            # --- NEW ROBUST LABEL PLACEMENT LOGIC ---

            # 1. Calculate the angle of the straight line between nodes in display space
            angle_deg = angle_in_display(ax, x0, y0, x1, y1) -8
            angle_rad = np.deg2rad(angle_deg)

            # 2. Calculate the simple midpoint of the straight line in data space
            mid_data = np.array([(x0 + x1) / 2.0, (y0 + y1) / 2.0])

            # 3. Calculate a perpendicular "nudge" vector in display space
            #    This vector points perpendicularly outwards from the line on the screen
            perp_vec_disp = np.array([-np.sin(angle_rad), np.cos(angle_rad)])
            
            # 4. Define how far to nudge the label in pixels.
            #    This is proportional to the curvature `rad`.
            #    This "magic number" controls the strength of the effect. Tune if needed.
            offset_strength_px = -400.0 
            pixel_offset = rad * offset_strength_px

            # 5. Apply the nudge in display space for visual correctness
            mid_disp = ax.transData.transform(mid_data)
            label_pos_disp = mid_disp + pixel_offset * perp_vec_disp
            
            # 6. Transform the final label position back to data space
            lx, ly = ax.transData.inverted().transform(label_pos_disp)

            # 7. Annotate at the final calculated position
            ann = ax.annotate(
                label_text,
                xy=(lx, ly),
                ha="center",
                va="center",
                fontsize=7,
                color="#111827",
                rotation=angle_deg, # Rotate to match the chord
                rotation_mode="anchor",
                zorder=4.2,
                clip_on=False,
            )
            ann.set_path_effects([pe.withStroke(linewidth=2.0, foreground="white", alpha=0.85)])
         
# Axes cosmetics
ax.set_xlim(-0.5, L + 0.5)
ax.set_ylim(-0.5, N - 0.5)
ax.set_xticks(x_positions)
ax.set_xticklabels(["Input"] + [f"After L{l+1}" for l in range(L)])
ax.set_yticks([])  # we draw our own labels
for spine in ax.spines.values():
    spine.set_visible(False)
# ax.set_title("Attention flow with residual stream")

# Legend (update attention color to red)
legend_elements = [
    Line2D([0], [0], marker="o", color="w", label="Inputs",
           markerfacecolor=role_colors["input"], markeredgecolor="black", markersize=7),
    Line2D([0], [0], marker="o", color="w", label="SEP",
           markerfacecolor=role_colors["sep"], markeredgecolor="black", markersize=7),
    Line2D([0], [0], marker="o", color="w", label="Outputs",
           markerfacecolor=role_colors["output"], markeredgecolor="black", markersize=7),
    Line2D([0], [0], linestyle=(0, (2, 2)), color="#94A3B8", lw=1.2, label="Residual"),
    Line2D([0], [0], color=arrow_color, lw=2, label="Attention"),
]
ax.legend(handles=legend_elements, frameon=False, ncol=5,
          loc="upper center", bbox_to_anchor=(0.5, -0.12))

plt.tight_layout()
# os.makedirs("figs", exist_ok=True)
# plt.savefig("figs/attention_flow.pdf", bbox_inches="tight")
# plt.savefig("figs/attn_flow.png", dpi=300, bbox_inches="tight")
plt.show()


# %% [markdown]
# ### Attn rollout

# %%
# ---- constants for ablation ----
# Get attention patterns for both layers on the validation set

with torch.no_grad():
    logits, cache = model.run_with_cache(val_inputs)
    dig_logits = logits[:,:,:-2]  # exclude SEP and mask token logits
    b_size = dig_logits.shape[0]  # batch size

# get required attention values
alpha = cache["pattern", 0][:, head_index_to_ablate]  # Layer 0
beta = cache["pattern", 1][:, head_index_to_ablate]   # Layer 1
alpha_sep_d1 = alpha[:,2, 0].unsqueeze(-1)  # SEP -> d1
alpha_sep_d2 = alpha[:,2, 1].unsqueeze(-1)  # SEP -> d2
beta_o2_o1 = beta[:,-1, -2].unsqueeze(-1)
beta_o2_sep = beta[:,-1, 2].unsqueeze(-1) # beta_o2_o1 + beta_o2_sep = 1.0

# Weights and embeddings
W_E = model.W_E.detach()  # (vocab, d_model)
W_pos = model.W_pos.detach()  # (seq_len, d_model)
W_U = model.unembed.W_U.detach()  # (d_model, vocab)

# Input tokens for d1, d2
d1_tok = val_inputs[:, 0]
d2_tok = val_inputs[:, 1]

# get embeds
big_d1 = W_E[d1_tok] + W_pos[0,:]  # d1 embedding (d_model)
big_d2 = W_E[d2_tok] + W_pos[1,:] # d2 embedding (d_model)
pos_o1 = W_pos[-2,:]  # o1 position (d_model)
pos_o2 = W_pos[-1,:]  # o2 position (d_model)
mask_embed = W_E[MASK]  # (d_model)
sep_embed = W_E[SEP] + W_pos[2,:]  # SEP token embedding (d_model)

# get shapes right
pos_o1 = pos_o1.expand(b_size, -1)
pos_o2 = pos_o2.expand(b_size, -1)
mask_embed = mask_embed.expand(b_size, -1)
sep_embed = sep_embed.expand(b_size, -1)

print("mask_embed shape:", mask_embed.shape)
print("sep_embed shape:", sep_embed.shape)
print("pos_o1 shape:", pos_o1.shape)
print("pos_o2 shape:", pos_o2.shape)
print("big_d1 shape:", big_d1.shape)
print("big_d2 shape:", big_d2.shape)
print("alpha_sep_d1 shape:", alpha_sep_d1.shape)
print("alpha_sep_d2 shape:", alpha_sep_d2.shape)
print("beta_o2_o1 shape:", beta_o2_o1.shape)
print("beta_o2_sep shape:", beta_o2_sep.shape)

# %%
# verify by reconstructing logits
l_o1 = (mask_embed + sep_embed + pos_o1 + alpha_sep_d1*big_d1+ alpha_sep_d2*big_d2) @ W_U  # logits for o1 (d_model)
l_o1_digits = l_o1[:, :N_DIGITS]
patched_o1_logits = l_o1_digits.argmax(dim=-1)
acc_patched_o1 = (patched_o1_logits == val_targets[:, -2]).float().mean().item()
print(f"Reconstructed accuracy for o1: {acc_patched_o1:.4f}")

l_o2 = ((1+beta_o2_o1)*mask_embed + beta_o2_o1*pos_o1 + pos_o2 + beta_o2_sep*(alpha_sep_d1*big_d1 + alpha_sep_d2*big_d2 + sep_embed)) @ W_U  # logits for o2 (d_model)
l_o2_digits = l_o2[:, :N_DIGITS]
patched_o2_logits = l_o2_digits.argmax(dim=-1)
acc_patched_o2 = (patched_o2_logits == val_targets[:, -1]).float().mean().item()
print(f"Reconstructed accuracy for o2: {acc_patched_o2:.4f}")

# Compare reconstructed logits to model logits for o2
with torch.no_grad():
    model_o2_logits = logits[:, -1, :N_DIGITS]  # [B, N_DIGITS]
    l2_diff = ((l_o2_digits - model_o2_logits).norm(dim=-1).mean().item())
    print(f"Mean L2 diff between reconstructed and model o2 logits: {l2_diff:.4f}")

print(f'Reconstructed accuracy: {(acc_patched_o1 + acc_patched_o2) / 2.0}')

# %%
# --- Logit Difference Calculation ---
# scale constants
scaled_pos_o1 = -beta_o2_sep * pos_o1
scaled_big_d1 = -beta_o2_o1 * alpha_sep_d1 * big_d1
scaled_big_d2 = -beta_o2_o1 * alpha_sep_d2 * big_d2
scaled_sep_embed = -beta_o2_o1 * sep_embed
scaled_mask_embed = beta_o2_o1 * mask_embed

# EQN 3 --> logit_o2 - logit_o1
# logit_diff = pos_o2 - (1-beta_o2_o1)*pos_o1 - beta_o2_o1*(alpha_sep_d1*big_d1 + alpha_sep_d2*big_d2 + sep_embed - mask_embed)  # (d_model) <--- CORRECT!
logit_diff = pos_o2 + scaled_pos_o1 + scaled_big_d1 + scaled_big_d2 + scaled_sep_embed + scaled_mask_embed
logit_diff = (logit_diff @ W_U )[:,:-2]  # exclude sep and mask token logits

print("logit_diff shape:", logit_diff.shape)




# check accuracy of patched o2 prediction
with torch.no_grad():
    # Run model and cache patterns over the whole val set
    base_o1 = logits[:, -2, :]  # [B, V]
    base_o2 = logits[:, -1, :]  # [B, V]

    # Patch o1 digits to predict o2
    patched_o2_digits = base_o1[:, :-2] + logit_diff
    pred_patched_o2 = patched_o2_digits.argmax(dim=-1)        # [B]

    # Baselines
    pred_naive_o2_from_o1 = base_o1.argmax(dim=-1)            # reuse o1 argmax as o2
    pred_true_o2          = base_o2.argmax(dim=-1)
    pred_true_o1          = base_o1.argmax(dim=-1)

    # Targets
    tgt_o1 = val_targets[:, -2]
    tgt_o2 = val_targets[:, -1]

    # Accuracies (token-level)
    acc_true_o1   = (pred_true_o1 == tgt_o1).float().mean().item()
    acc_true_o2   = (pred_true_o2 == tgt_o2).float().mean().item()
    acc_o1_as_o2  = (pred_naive_o2_from_o1 == tgt_o2).float().mean().item()
    acc_o2_as_o1 = (pred_patched_o2 == tgt_o1).float().mean().item()
    acc_patched_o2= (pred_patched_o2 == tgt_o2).float().mean().item()

    # Optional: sequence accuracy for both outputs using [o1, o1+diff]
    pred_o1 = base_o1.argmax(dim=-1)
    both_right = ((pred_o1 == tgt_o1) & (pred_patched_o2 == tgt_o2)).float().mean().item()
    generic_acc = (acc_true_o1 + acc_patched_o2) / 2.0

print("--- Using o1 and o1+logit_diff to predict o2 ---")
print(f"Baseline o2 acc (model o2 logits):    {acc_true_o2:.4f}")
print(f"Naive acc (o2 := argmax(o1)):         {acc_o1_as_o2:.4f}")
print(f"Patched acc (o2 := argmax(o1+diff)):  {acc_patched_o2:.4f}")
# print(f"Both-correct seq acc [o1, o1+diff]:   {both_right:.4f}")
print(f"[original_o1, o1+diff]:               {generic_acc:.4f} <-- (should be 0.9198)")
print(f"True o1 acc:                          {acc_true_o1:.4f} (baseline)")
print(f"o2 as o1 acc:                         {acc_o2_as_o1:.4f} (should be low)")

# %%
# verify that logit_diff is correct

# Add logit_diff to o1 logits (digits only) and compare with o2
with torch.no_grad():

    # Sanity on logit_diff length
    assert logit_diff.shape[-1] == N_DIGITS, f"Expected logit_diff over {N_DIGITS} digits, got {logit_diff.shape}"

    base_o1 = logits[sample_idx, -2].detach().clone()
    base_o2 = logits[sample_idx, -1].detach().clone()

    # Add only to digit columns; leave MASK/SEP unchanged
    patched_o1 = base_o1.clone()
    delta = logit_diff[sample_idx].to(device=patched_o1.device, dtype=patched_o1.dtype)
    patched_o1[:N_DIGITS] += delta

    pred_o1_before = int(base_o1.argmax().item())
    pred_o2 = int(base_o2.argmax().item())
    pred_o1_after = int(patched_o1.argmax().item())
    gold_o2 = int(val_targets[sample_idx, -1].item()) # ground truth for o2

    # How close are patched o1 digit logits to o2 digit logits?
    diff_vec = (patched_o1[:N_DIGITS] - base_o2[:N_DIGITS])
    l2 = float(diff_vec.norm().item())
    max_abs = float(diff_vec.abs().max().item())

print("Sample:", sample_idx, "seq:", val_inputs[sample_idx].cpu().numpy())
print(f"o1 pred before: {pred_o1_before}")
print(f"o2 pred (orig): {pred_o2} | gold o2: {gold_o2}")
print(f"o1 pred after +logit_diff: {pred_o1_after}")
print("Matches o2 prediction? ", pred_o1_after == pred_o2)
print("Matches o2 gold?       ", pred_o1_after == gold_o2)
print(f"Digit-logits closeness to o2 -> L2: {l2:.4e}, max|diff|: {max_abs:.4e}")

# %%
# --- PCA grid of cumulative o1 + logit_diff term additions, with SVM per panel ---
from math import ceil
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit, cross_val_score
from sklearn.svm import LinearSVC

@torch.no_grad()
def _pca_two_sets(A: torch.Tensor, B: torch.Tensor):
    # A,B: [Na,D], [Nb,D] (float)
    A2 = A.reshape(-1, A.shape[-1]).float().cpu()
    B2 = B.reshape(-1, B.shape[-1]).float().cpu()
    X_all = torch.cat([A2, B2], dim=0)
    mean = X_all.mean(0, keepdim=True)
    Xc = X_all - mean
    U, S, Vh = torch.linalg.svd(Xc, full_matrices=False)
    comps = Vh[:2, :].T
    var_ratio = (S**2 / (S**2).sum())[:2]
    return (A2 - mean) @ comps, (B2 - mean) @ comps, var_ratio  # -> [Na,2], [Nb,2], [2]

with torch.no_grad():
    # Base o1 digit logits
    base_o1_digits = logits[:, -2, :N_DIGITS]  # [B, N_DIGITS]

    # Build individual term contributions in d_model
    term_pos_o2   =     pos_o2
    term_pos_o1   = -beta_o2_sep * pos_o1
    term_big_d1   = -beta_o2_o1 * alpha_sep_d1 * big_d1
    term_big_d2   = -beta_o2_o1 * alpha_sep_d2 * big_d2
    term_sep      = -beta_o2_o1 * sep_embed
    term_mask     =  beta_o2_o1 * mask_embed

    # Project each term to digit logits
    W_U_t = getattr(model, "W_U", model.unembed.W_U)
    def unembed_digits(x): return (x @ W_U_t)[:, :N_DIGITS]

    contrib_pos_o2 = unembed_digits(term_pos_o2)
    contrib_pos_o1 = unembed_digits(term_pos_o1)
    contrib_big_d1 = unembed_digits(term_big_d1)
    contrib_big_d2 = unembed_digits(term_big_d2)
    contrib_sep    = unembed_digits(term_sep)
    contrib_mask   = unembed_digits(term_mask)

    # Order of cumulative additions
    steps = [
        ("o1 (base)",                      base_o1_digits, []),
        ("+ pos_o2",                       base_o1_digits, [contrib_pos_o2]),
        ("+ pos_o2 -(1-β)·pos_o1",         base_o1_digits, [contrib_pos_o2, contrib_pos_o1]),
        ("+ … -β·α·d1",                    base_o1_digits, [contrib_pos_o2, contrib_pos_o1, contrib_big_d1]),
        ("+ … -β·α·d2",                    base_o1_digits, [contrib_pos_o2, contrib_pos_o1, contrib_big_d1, contrib_big_d2]),
        ("+ … -β·sep",                     base_o1_digits, [contrib_pos_o2, contrib_pos_o1, contrib_big_d1, contrib_big_d2, contrib_sep]),
        ("+ … +β·mask",                    base_o1_digits, [contrib_pos_o2, contrib_pos_o1, contrib_big_d1, contrib_big_d2, contrib_sep, contrib_mask]),
        ("o1 + logit_diff (≈ o2)",         base_o1_digits, [contrib_pos_o2, contrib_pos_o1, contrib_big_d1, contrib_big_d2, contrib_sep, contrib_mask]),
    ]

    # Labels/masks: keep non-duplicate (d1!=d2) and only cases where tgt o2 is one of {d1,d2}
    d1 = d1_tok
    d2 = d2_tok
    tgt_o2 = val_targets[:, -1]
    mask_unique = (d1 != d2) & ((tgt_o2 == d1) | (tgt_o2 == d2))
    if mask_unique.sum() < 10:
        print("Not enough non-duplicate examples to evaluate.")
        raise SystemExit

    # Classes for plotting
    m0 = mask_unique & (tgt_o2 == d1)  # class 0: o2=d1
    m1 = mask_unique & (tgt_o2 == d2)  # class 1: o2=d2
    y_cls = (tgt_o2[mask_unique] == d2[mask_unique]).long()

    # Build patched matrices per step
    patched_per_step = []
    titles = []
    for title, base, add_list in steps:
        patched = base.clone()
        if add_list:
            for c in add_list:
                patched = patched + c
        # The last panel is identical to the previous because both include all terms;
        # keep title though for clarity
        patched_per_step.append(patched)
        titles.append(title)

    # Grid layout
    n_panels = len(patched_per_step)
    ncols = 4
    nrows = ceil(n_panels / ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5*ncols, 4.2*nrows))
    axes = np.array(axes).reshape(nrows, ncols)

    # Choose how to color points:
    #   "o2_true"    -> color by ground-truth o2 class (current behavior)
    #   "panel_pred" -> color by this panel's argmax between {d1,d2}
    color_by = "panel_pred"

    cA, cB = "#1f77b4", "#ff7f0e"
    for i, (patched, title) in enumerate(zip(patched_per_step, titles)):
        r, c = divmod(i, ncols)
        ax = axes[r, c]

        if color_by == "panel_pred":
            idx = torch.arange(patched.shape[0], device=patched.device)
            score_d1 = patched[idx, d1]
            score_d2 = patched[idx, d2]
            pred_is_d2 = score_d2 > score_d1  # [B]
            m1_plot = mask_unique & pred_is_d2
            m0_plot = mask_unique & (~pred_is_d2)
            legend_labels = ("pred o2=d1", "pred o2=d2")
        else:
            m0_plot, m1_plot = m0, m1
            legend_labels = ("o2=d1", "o2=d2")

        # Two sets for PCA
        A = patched[m1_plot]  # class 1
        B = patched[m0_plot]  # class 0
        if A.numel() == 0 or B.numel() == 0:
            ax.axis("off")
            ax.set_title(f"{title}\n(no samples for one class)")
            continue

        A_pc, B_pc, var_ratio = _pca_two_sets(A, B)

        ax.scatter(A_pc[:, 0], A_pc[:, 1], s=10, color=cB, alpha=0.55, label=legend_labels[1], linewidths=0, rasterized=True)
        ax.scatter(B_pc[:, 0], B_pc[:, 1], s=10, color=cA, alpha=0.55, label=legend_labels[0], linewidths=0, rasterized=True)
        ax.set_title(title)
        ax.set_xlabel(f"PC1 ({var_ratio[0].item():.1%})")
        ax.set_ylabel(f"PC2 ({var_ratio[1].item():.1%})")
        ax.grid(True, linestyle=":", alpha=0.3)
        if r == 0 and c == 0:
            ax.legend(frameon=False, loc="best")

    # Hide any empty axes
    for j in range(n_panels, nrows*ncols):
        r, c = divmod(j, ncols)
        axes[r, c].axis("off")

    plt.tight_layout()
    plt.show()


# %%
# --- Mechanistic Decomposition of logit_diff (for cases where target is d2) ---
# Helper to gather the specific logits for d1 and d2 from a logit matrix
def gather_pair_cols(logits, d1, d2):
    # logits: [B, N_DIGITS], d1: [B], d2: [B]
    d1_logits = logits.gather(-1, d1.unsqueeze(-1))
    d2_logits = logits.gather(-1, d2.unsqueeze(-1))
    return torch.cat([d1_logits, d2_logits], dim=-1) # -> [B, 2]

# --- Data Calculation (Unchanged) ---
m = m1 

base_diff_d2_d1 = (
    gather_pair_cols(base_o1_digits[m], d1[m], d2[m])[:, 1] - 
    gather_pair_cols(base_o1_digits[m], d1[m], d2[m])[:, 0]
).mean().cpu()

contribs = [contrib_pos_o2, contrib_pos_o1, contrib_big_d1, contrib_big_d2, contrib_sep, contrib_mask]
diff_contributions = []
for c in contribs:
    c_d2 = gather_pair_cols(c[m], d1[m], d2[m])[:, 1]
    c_d1 = gather_pair_cols(c[m], d1[m], d2[m])[:, 0]
    diff_contributions.append((c_d2 - c_d1).mean().item())

final_calc_diff = base_diff_d2_d1 + sum(diff_contributions)
actual_diff = (
    gather_pair_cols(logits[:, -1, :N_DIGITS][m], d1[m], d2[m])[:, 1] -
    gather_pair_cols(logits[:, -1, :N_DIGITS][m], d1[m], d2[m])[:, 0]
).mean().cpu()

# --- Labels Matching the Paper (Unchanged) ---
base_name_paper = r'$\ell_{o_1}$ (base)'
contrib_names_paper = [
    r'$P(o_2)$', r'$-\beta_{o_2 \to s} P(o_1)$', r'$-\beta_{o_2 \to o_1}\alpha_{s \to d_1}D_1$',
    r'$-\beta_{o_2 \to o_1}\alpha_{s \to d_2}D_2$', r'$-\beta_{o_2 \to o_1}S$', r'$+\beta_{o_2 \to o_1}E(m)$'
]


# --- FINAL PLOTTING CODE ---
fig, ax = plt.subplots(figsize=(13, 7))

all_names = [base_name_paper] + contrib_names_paper
all_values = [base_diff_d2_d1.item()] + diff_contributions
cumulative_values = np.cumsum(all_values)
step_plot_values = np.insert(cumulative_values, 0, base_diff_d2_d1.item())

colors = ['grey'] + ['#377eb8' if v > 0 else '#e41a1c' for v in diff_contributions]

# --- Main bars ---
bars = ax.barh(all_names, all_values, color=colors, alpha=0.9, zorder=2)
ax.invert_yaxis()

# --- Text Annotations (Value only) ---
for bar, value in zip(bars, all_values):
    ha = 'left' if value >= 0 else 'right'
    offset = 0.5
    ax.text(bar.get_width() + (offset if value >=0 else -offset), bar.get_y() + bar.get_height()/2,
            f'{value:.2f}', va='center', ha=ha, fontsize=10)

# --- Top Axis Step Plot ---
ax_top = ax.twiny()
y_steps = np.arange(len(all_names) + 1) - 0.5
# Add a label to the step plot for the legend
step_line, = ax_top.step(step_plot_values, y_steps, color='dimgray', where='post', 
                         linestyle='--', linewidth=1.5, label='Cumulative Sum')
ax_top.set_xlabel(r"Cumulative $\Delta\ell$", fontsize=11, color='dimgray')
ax_top.tick_params(axis='x', colors='dimgray')
ax_top.spines['right'].set_visible(False)
ax_top.tick_params(axis='y', right=False, labelright=False)

# --- Robust Vertical Line ---
if np.isclose(final_calc_diff, actual_diff):
    y_limits = ax.get_ylim()
    # Add a label to the plot line for the legend
    final_line, = ax.plot([final_calc_diff, final_calc_diff], y_limits, 
                           color='purple', linestyle='--', linewidth=2, 
                           label='Final / Actual Sum', zorder=10)
    ax.text(final_calc_diff, y_limits[1] - 0.2, f' {final_calc_diff:.2f}', color='purple', 
            ha='left', va='bottom', fontsize=10)

# --- NEW: COMBINED LEGEND ---
# Get handles from both axes and combine them into one legend
handles = [step_line, final_line]
ax.legend(handles=handles, frameon=False, loc="lower left", fontsize=10)

# --- General Aesthetics ---
# A more formal title for a paper
# ax.set_title(r"Mechanistic Decomposition of $\Delta\ell$ (for cases where target is $d_2$)", fontsize=14)
# Updated x-axis label using delta_ell and clarifying directionality
ax.set_xlabel(r"Contribution to $\ell_{o_2}$ (Negative favours $d_1$ $\leftarrow$ | $\rightarrow$ Positive favours $d_2$)", fontsize=12)
ax.grid(axis='x', linestyle=':', alpha=0.6, zorder=1)
ax.axvline(0, color='dimgray', linestyle='-', linewidth=1.2, zorder=1) # Prominent zero line
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_position(('outward', 5))
plt.yticks(fontsize=12)
ax_top.spines['top'].set_position(('outward', 5)) 

# Set limits and layout
ax.set_xlim(left=-25, right=55)
ax_top.set_xlim(ax.get_xlim()) 
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()

# %%


# %% [markdown]
# ### W_E, W_U, W_pos

# %%
from sklearn.metrics.pairwise import cosine_similarity

def calculate_8d_pairwise_angles(model):
    """
    Calculates pairwise angles (degrees) for:
      - W_E (token embeddings)
      - W_U (unembeddings)
      - W_pos (positional embeddings for the used sequence length)
    """
    print(f"Calculating pairwise angles in the original {D_MODEL}D space...")
    model.eval()

    # Token embeddings/unembeddings
    w_e = model.embed.W_E.detach().cpu().numpy()
    w_u = model.unembed.W_U.T.detach().cpu().numpy()
    token_labels = [str(d) for d in DIGITS] + ['SEP', 'MASK']

    # Positional embeddings (first SEQ_LEN positions actually used)
    w_pos = model.W_pos[:SEQ_LEN].detach().cpu().numpy()
    pos_labels = [f'd{i+1}' for i in range(LIST_LEN)] + ['SEP'] + [f'o{i+1}' for i in range(LIST_LEN)]

    def cos_to_angles_df(mat, labels):
        cs = cosine_similarity(mat)
        cs = np.clip(cs, -1.0, 1.0)
        ang = np.rad2deg(np.arccos(cs))
        return pd.DataFrame(ang, index=labels, columns=labels)

    df_e = cos_to_angles_df(w_e, token_labels)
    df_u = cos_to_angles_df(w_u, token_labels)
    df_pos = cos_to_angles_df(w_pos, pos_labels)

    # print(f"\n--- Pairwise Angles (Degrees) for W_E (Embeddings) in {D_MODEL}D ---")
    # print(df_e.to_markdown(floatfmt=".1f"))

    # print(f"\n--- Pairwise Angles (Degrees) for W_U (Unembeddings) in {D_MODEL}D ---")
    # print(df_u.to_markdown(floatfmt=".1f"))

    print(f"\n--- Pairwise Angles (Degrees) for W_pos (Positional Embeddings) in {D_MODEL}D ---")
    print(df_pos.to_markdown(floatfmt=".1f"))

    return df_e, df_u, df_pos

# Calculate and display the pairwise angles from the d_model space
df_e_angles_8d, df_u_angles_8d, df_pos_angles_8d = calculate_8d_pairwise_angles(model)


# %%
def analyze_spacing_invariant(df_angles, name, subset_labels=None):
    """
    Permutation-invariant spacing analysis: for each vector, find the two nearest
    neighbors (smallest angles) within subset_labels and report mean/std of those angles.
    - df_angles: square DataFrame of pairwise angles (degrees).
    - name: label for printing.
    - subset_labels: iterable of index/column labels to restrict analysis. If None,
      defaults to:
        * token dfs -> first len(DIGITS) digit labels ("0".."N-1")
        * positional dfs -> all labels in df_angles
    """
    if subset_labels is None:
        digit_label_candidates = [str(d) for d in DIGITS]
        if set(digit_label_candidates).issubset(set(df_angles.index)):
            # Token case: restrict to digits only
            subset_labels = digit_label_candidates
        else:
            # Positional case: use all labels present (e.g., d1, d2, SEP, o1, o2)
            subset_labels = list(df_angles.index)

    sub = df_angles.loc[subset_labels, subset_labels].copy()

    all_neighbor_angles = []
    print(f"\n--- Permutation-Invariant Spacing Analysis for {name} ---")

    for lbl in subset_labels:
        angles_from_lbl = sub.loc[lbl].drop(lbl)  # drop self
        sorted_angles = angles_from_lbl.sort_values()
        all_neighbor_angles.extend(sorted_angles.iloc[:2].values)  # two nearest

    neighbor_angles = np.array(all_neighbor_angles, dtype=float)
    print(f"Mean neighbor angle: {neighbor_angles.mean():.2f}°")
    print(f"Std Dev of neighbor angles: {neighbor_angles.std():.2f}°")

# Example calls
digit_labels = [str(d) for d in DIGITS]
pos_labels = [f'd{i+1}' for i in range(LIST_LEN)] + ['SEP'] + [f'o{i+1}' for i in range(LIST_LEN)]

analyze_spacing_invariant(df_e_angles_8d,  "W_E (Embeddings)", subset_labels=digit_labels)
analyze_spacing_invariant(df_u_angles_8d,  "W_U (Unembeddings)", subset_labels=digit_labels)
analyze_spacing_invariant(df_pos_angles_8d,"W_pos (Positional Embeddings)", subset_labels=pos_labels)

# %% [markdown]
# ### Attention

# %%

# try setting specific attention positions to zero
layer_to_ablate = 1 # output digits do nothijg in layer 0
print(f"Layer {layer_to_ablate} Ablation")
# Define which specific attention position you want to zero out
query_pos, key_pos = 4,3    # row=q, col=k
renorm_rows = True           # set False to keep the magnitude shrink
# ^ If you intentionally want to reduce the head’s contribution magnitude (e.g., test reliance on that single edge plus scale effects), set renorm_rows=False.

def specific_attention_ablation_hook(pattern, hook):
    # pattern: [batch, n_heads, Q, K]
    out = pattern.clone()
    print(f'BEFORE Ablation:\n{out[0, head_index_to_ablate, :, :].cpu().numpy()}')

    # Zero the specific edge
    out[:, head_index_to_ablate, query_pos, key_pos] = 0.0
    testval = 0.0
    # out[:, head_index_to_ablate, 2, 0] = testval
    # out[:, head_index_to_ablate, 2, 1] = 3*testval
    # out[:, head_index_to_ablate, :3, :] = 0.0
    # out[:, head_index_to_ablate, 3:, :] = 0.0
    # out[:, head_index_to_ablate, 2, :2] = 0.5

    if renorm_rows:
        # Renormalize just the affected row for that head (per batch)
        row = out[:, head_index_to_ablate, query_pos, :]              # [B, K]
        s = row.sum(dim=-1, keepdim=True).clamp_min(1e-12)            # avoid div-by-zero
        out[:, head_index_to_ablate, query_pos, :] = row / s

    print(f'AFTER Ablation:\n{out[0, head_index_to_ablate, :, :].cpu().numpy()}')
    print("-" * 45)

    return out

# Get the attention pattern hook name
attn_pattern_hook_name = utils.get_act_name("pattern", layer_to_ablate)

# --- Calculate Ablated Loss on last 2 digits ---
with torch.no_grad():
    ablated_logits = model.run_with_hooks(
        val_inputs,
        return_type="logits",  # Get logits instead of loss
        fwd_hooks=[(attn_pattern_hook_name, specific_attention_ablation_hook)],
    )
    # Slice to get logits for the last two positions
    output_logits_ablated = ablated_logits[:, LIST_LEN+1:]
    # Calculate the loss
    ablated_loss = loss_fn(
        output_logits_ablated.reshape(-1, VOCAB), output_targets.reshape(-1)
    )

    # Calculate accuracy
    ablated_predictions = ablated_logits.argmax(dim=-1)
    ablated_output_predictions = ablated_predictions[:, LIST_LEN+1:]
    ablated_accuracy = (ablated_output_predictions == output_targets).float().mean()

print("\n--- Performance Metrics (on last 2 digits) ---")
print(f"{'':<12} | {'Original':<10} | {'Ablated':<10}")
print("-" * 45)
print(f"{'Loss:':<12} | {original_loss.item():<10.3f} | {ablated_loss.item():<10.3f}")
print(f"{'Accuracy:':<12} | {original_accuracy.item():<10.3f} | {ablated_accuracy.item():<10.3f}")
print("-" * 45)

# Get the predicted tokens from the ablated logits
ablated_predictions = ablated_logits.argmax(dim=-1)

print(f"\n--- Prediction Comparison (Sample {sample_idx}) ---")
print(f"Original sequence:   {val_inputs[sample_idx].cpu().numpy()}")
print(f"Original prediction: {original_predictions[sample_idx].cpu().numpy()}")
print(f"Ablated prediction:  {ablated_predictions[sample_idx].cpu().numpy()}")
print("-" * 45)

# %%
assert ablated_output_predictions.shape == output_targets.shape, "Shape mismatch"
assert ablated_output_predictions.ndim == 2 and ablated_output_predictions.shape[1] == 2, "Assumes LIST_LEN == 2"
assert ablated_output_predictions.dtype == output_targets.dtype, "Dtype mismatch"

eq = (ablated_output_predictions == output_targets)  # [B, 2] bool

num_sequences, two = eq.shape
num_tokens = eq.numel()
num_correct_tokens = eq.sum().item()
token_acc = num_correct_tokens / num_tokens

per_seq_correct = eq.sum(dim=1)
both_right = (per_seq_correct == 2).sum().item()
one_right  = (per_seq_correct == 1).sum().item()
both_wrong = (per_seq_correct == 0).sum().item()

# Wrong counts per position
o1_wrong = (~eq[:, 0]).sum().item()
o2_wrong = (~eq[:, 1]).sum().item()

seq_failures = (per_seq_correct < 2).sum().item()
seq_acc = both_right / num_sequences if num_sequences > 0 else 0.0

print(f"Token acc: {token_acc:.3f}  ({num_correct_tokens}/{num_tokens})")
print(f"Both right: {both_right}, One right: {one_right}, Both wrong: {both_wrong}\n")

print("Wrong Counts Table")
print("------------------")
print(f"{'Category':<12} | {'Count':>6}")
print("------------------")
print(f"{'o1 wrong':<12} | {o1_wrong:>6}")
print(f"{'o2 wrong':<12} | {o2_wrong:>6}")
print(f"{'both wrong':<12} | {both_wrong:>6}")
print("------------------")
print(f"{'TOTAL wrong':<12} | {(num_sequences - both_right):>6}")
print()

print(f"Seq acc (both outputs correct): {seq_acc:.3f}  -> failures: {seq_failures}/{num_sequences}\n")

# Dupes table (predicted o1 == o2)
same_pred_mask = (ablated_output_predictions[:, 0] == ablated_output_predictions[:, 1])
dupes_total = same_pred_mask.sum().item()
dupes_correct = ((same_pred_mask) & eq[:, 0] & eq[:, 1]).sum().item()
dupes_wrong = ((same_pred_mask) & ~eq[:, 0] & ~eq[:, 1]).sum().item()
dupes_o1_right = ((same_pred_mask) & eq[:, 0] & ~eq[:, 1]).sum().item()
dupes_o2_right = ((same_pred_mask) & eq[:, 1] & ~eq[:, 0]).sum().item()

print("Dupes Table (o1 == o2)")
print("----------------------")
print(f"{'Category':<16} | {'Count':>6}")
print("----------------------")
print(f"{'dupes correct':<16} | {dupes_correct:>6}")
print(f"{'o1 right only':<16} | {dupes_o1_right:>6}")
print(f"{'o2 right only':<16} | {dupes_o2_right:>6}")
print(f"{'dupes wrong':<16} | {dupes_wrong:>6}")
print(f"{'dupes total':<16} | {dupes_total:>6}")
print()

# %%
def analyze_o2_errors(preds, targets, inputs=None, top_k=10):
    """
    preds, targets: [B, 2] (o1, o2)
    inputs: [B, 5] full token seq [d1, d2, SEP, o1, o2] (optional)
    """
    assert preds.shape == targets.shape and preds.ndim == 2 and preds.shape[1] == 2
    B = preds.shape[0]
    o1_pred, o2_pred = preds[:, 0], preds[:, 1]
    o1_true, o2_true = targets[:, 0], targets[:, 1]

    o2_wrong_mask = (o2_pred != o2_true)
    num_o2_wrong = int(o2_wrong_mask.sum().item())
    if num_o2_wrong == 0:
        print("No o2 errors.")
        return

    # Relations to o1
    dupes_rate = (o2_pred[o2_wrong_mask] == o1_pred[o2_wrong_mask]).float().mean().item()
    equals_o1_true_rate = (o2_pred[o2_wrong_mask] == o1_true[o2_wrong_mask]).float().mean().item()
    o1_correct_given_o2_wrong = (o1_pred[o2_wrong_mask] == o1_true[o2_wrong_mask]).float().mean().item()

    # Relations to inputs (if provided)
    if inputs is not None:
        d1, d2 = inputs[:, 0], inputs[:, 1]
        eq_d1 = int((o2_pred[o2_wrong_mask] == d1[o2_wrong_mask]).sum().item())
        eq_d2 = int((o2_pred[o2_wrong_mask] == d2[o2_wrong_mask]).sum().item())
        eq_d1_all = int((o2_pred == d1).sum().item())
    else:
        eq_d1 = eq_d2 = None
        eq_d1_all = None

    # Frequency of o2 predictions when wrong
    vals = o2_pred[o2_wrong_mask]
    counts = torch.bincount(vals, minlength=VOCAB).cpu()
    top_idx = counts.argsort(descending=True)[:top_k]

    def tok_label(t):
        t = int(t)
        if t < N_DIGITS: return f"{t}"
        if t == MASK: return "MASK"
        if t == SEP: return "SEP"
        return f"tok{t}"

    print(f"o2 wrong: {num_o2_wrong}/{B} ({num_o2_wrong/B:.2%})")
    print(f"P(o2_pred == o1_pred | o2 wrong): {dupes_rate:.2%}")
    print(f"P(o2_pred == o1_true | o2 wrong): {equals_o1_true_rate:.2%}")
    print(f"P(o1 correct | o2 wrong): {o1_correct_given_o2_wrong:.2%}")
    if eq_d1 is not None:
        print(f"P(o2_pred == d1): {eq_d1_all/B:.2%} ({eq_d1_all})")
        print(f"P(o2_pred == d1 | o2 wrong): {eq_d1/num_o2_wrong:.2%} ({eq_d1})")
        print(f"P(o2_pred == d2 | o2 wrong): {eq_d2/num_o2_wrong:.2%} ({eq_d2})")

    print("\nTop o2 predictions when wrong:")
    for t in top_idx.tolist():
        c = int(counts[t].item())
        if c == 0: continue
        print(f"  {tok_label(t):>4}: {c} ({c/num_o2_wrong:.2%})")

    # Show a few concrete examples
    show = min(5, num_o2_wrong)
    idxs = torch.nonzero(o2_wrong_mask).squeeze(-1)[:show].cpu().tolist()
    print("\nExamples (d1,d2) -> (o1_true,o2_true) | (o1_pred,o2_pred):")
    for i in idxs:
        if inputs is not None:
            d1i, d2i = int(inputs[i, 0]), int(inputs[i, 1])
            left = f"({d1i},{d2i}) -> ({int(o1_true[i])},{int(o2_true[i])})"
        else:
            left = f"-> ({int(o1_true[i])},{int(o2_true[i])})"
        print(f"  {left} | ({int(o1_pred[i])},{int(o2_pred[i])})")

analyze_o2_errors(ablated_output_predictions, output_targets, inputs=val_inputs)

# %%
renorm_rows = False
ablate_in_l0 = [
                # (2,0),
                # (2,1),
                # (4,2),
                # (3,2),
                # below don't change acc
                (4,3),
                (0,0),
                (1,0)
                ]
ablate_in_l1 = [
                # (3,2),
                # (4,2),
                # (4,3),
                # below don't change acc
                (0,0),
                (1,0),
                (2,0),
                (2,1)
                
                ]

ablate_in_l2 = [(0,0),(1,0),(2,0), (2,1), (3,0),  (4,0), (4,1), (4,2), (4,3)]

# Try ablating multiple layer attention patterns at same time
def build_qk_mask(positions=None, queries=None, keys=None, seq_len=SEQ_LEN):
    """
    Create a boolean mask of shape (seq_len, seq_len) where True means "ablate this (q,k)".
    You can pass:
      - positions: list of (q, k) tuples
      - or queries: iterable of q, and keys: iterable of k (outer-product mask)
    """
    mask = torch.zeros(seq_len, seq_len, dtype=torch.bool)
    if positions is not None:
        for q, k in positions:
            mask[q, k] = True
    else:
        if queries is None:
            queries = range(seq_len)
        if keys is None:
            keys = range(seq_len)
        for q in queries:
            mask[q, keys] = True
    return mask

def make_pattern_hook(mask_2d: torch.Tensor, head_index=None, set_to=0.0, renorm=True, eps=1e-12):
    """
    Returns a fwd hook for the 'pattern' activation that:
      - sets masked entries to set_to (default 0.0)
      - optionally renormalizes rows so they sum to 1 again (per head, per batch, per query row)
    Args:
      mask_2d: Bool tensor [Q, K]
      head_index: int to affect a single head, or None to affect all heads
      set_to: value to write into masked entries (usually 0.0)
      renorm: whether to renormalize rows after masking
    """
    mask_2d = mask_2d.detach()

    def hook(pattern, hook):
        # pattern: [batch, n_heads, Q, K]
        B, H, Q, K = pattern.shape
        m4_all = mask_2d.to(pattern.device).view(1, 1, Q, K)  # broadcastable
        # Keep a copy for safe fallback in renorm
        pre = pattern.clone()
        print(f"\nLayer {hook.layer()} Ablation")
        print(f'BEFORE Ablation:\n{pattern[0, head_index, :, :].cpu().numpy()}')
        # print(f'Mask:\n{m4_all[0, 0, :, :].cpu().numpy()}')
        

        if head_index is None:
            pattern = torch.where(m4_all, torch.as_tensor(set_to, device=pattern.device), pattern)
        else:
            m3 = m4_all.squeeze(1)  # [1, Q, K]
            ph = pattern[:, head_index]  # [B, Q, K]
            ph = torch.where(m3, torch.as_tensor(set_to, device=pattern.device), ph)
            pattern[:, head_index] = ph

        if renorm:
            # Renormalize only rows whose query index has any masked key
            rows_to_fix = mask_2d.any(dim=-1)  # [Q]
            if rows_to_fix.any():
                rows_idx = rows_to_fix.nonzero(as_tuple=False).squeeze(-1)  # [Nr]
                heads = range(H) if head_index is None else [head_index]
                for h in heads:
                    # p: [B, Nr, K]
                    p = pattern[:, h, rows_idx, :]
                    s = p.sum(dim=-1, keepdim=True).clamp_min(eps)   # [B, Nr, 1]
                    pattern[:, h, rows_idx, :] = p / s

        print(f'AFTER Ablation:\n{pattern[0, head_index, :, :].cpu().numpy()}')
        return pattern

    return hook

# Example usage:
# Define what to ablate per layer:
# - As explicit (q,k) pairs
# - Or as queries/keys sets (outer-product)
layers_to_ablate = {
    0: build_qk_mask(positions=ablate_in_l0, seq_len=SEQ_LEN),
    1: build_qk_mask(positions=ablate_in_l1, seq_len=SEQ_LEN),
    # 2: build_qk_mask(positions=ablate_in_l2, seq_len=SEQ_LEN),
}

# Apply to a single head or all heads
head = None  # Set to None to affect all heads, or specify a head index (e.g., 0)

# Build hooks
fwd_hooks = []
for layer_idx, mask in layers_to_ablate.items():
    hook_name = utils.get_act_name("pattern", layer_idx)
    fwd_hooks.append((hook_name, make_pattern_hook(mask, head_index=head, set_to=0.0, renorm=renorm_rows)))

# Run with hooks and evaluate on last two positions
with torch.no_grad():
    logits_multi = model.run_with_hooks(val_inputs, return_type="logits", fwd_hooks=fwd_hooks)

output_logits_multi = logits_multi[:, LIST_LEN+1:]
ablated_output_predictions = output_logits_multi.argmax(dim=-1)
output_targets = val_targets[:, LIST_LEN+1:]

ablated_loss = loss_fn(output_logits_multi.reshape(-1, VOCAB), val_targets[:, LIST_LEN+1:].reshape(-1))
ablated_acc = (ablated_output_predictions == val_targets[:, LIST_LEN+1:]).float().mean()

print("\n--- Performance Metrics ---")
print(f"Multi-layer attention ablation -> Loss: {ablated_loss.item():.3f}, Acc: {ablated_acc.item():.3f}")

# Optional: inspect a sample
idx = sample_idx
print("Sample sequence:", val_inputs[idx].cpu().numpy())
print("Original:", original_predictions[idx].cpu().numpy())
print("Ablated: ", logits_multi.argmax(dim=-1)[idx].cpu().numpy())

# %% [markdown]
# ## Positional embedding swap test for o1 and o2
# 
# This cell temporarily swaps the learned positional embeddings for the first two output positions (o1 and o2), evaluates predictions, and checks if outputs swap accordingly. It restores the original embeddings afterward to avoid side effects.

# %%
# Swap o1/o2 positional embeddings, test effect on predictions, then restore.
import torch

# Reuse existing globals if available, else derive
assert 'model' in globals(), "Expected a trained HookedTransformer in variable `model`."

# Derive positions from model config: seq = [d1..dL, SEP, o1..oL]
L = (model.cfg.n_ctx - 1) // 2
pos_o1, pos_o2 = L + 1, L + 2

# Access learned positional embedding table
pos_table = getattr(getattr(model, 'pos_embed', None), 'W_pos', None)
if pos_table is None:
    raise RuntimeError("Model does not use learned positional embeddings (W_pos not found). Cannot swap.")

# Prefer existing validation data if present; otherwise generate a small set
if 'val_inputs' in globals() and 'val_targets' in globals():
    X_val, Y_val = val_inputs, val_targets
else:
    # Infer N_DIGITS from vocab: digits + {MASK, SEP}
    d_vocab = getattr(model.cfg, 'd_vocab', None)
    if d_vocab is None:
        d_vocab = model.cfg.d_vocab_out
    N_DIGITS = d_vocab - 2
    from patrick.grid_search import make_validation_set
    device = next(model.parameters()).device
    X_val, Y_val = make_validation_set(
        n_examples=2048,
        n_digits=N_DIGITS,
        list_len=L,
        device=device,
        seed=1234,
    )

@torch.no_grad()
def preds_and_acc(x, y):
    logits = model(x)[:, L + 1 :, :]  # [B, L, V]
    preds = logits.argmax(dim=-1)     # [B, L]
    gold = y[:, L + 1 :]
    acc = (preds == gold).float().mean().item()
    return preds, acc

# 1) Baseline predictions
with torch.no_grad():
    preds_before, acc_before = preds_and_acc(X_val, Y_val)

# 2) Swap pos embeddings for o1/o2, test, then restore
row_o1 = pos_table[pos_o1].detach().clone()
row_o2 = pos_table[pos_o2].detach().clone()

try:
    with torch.no_grad():
        pos_table[pos_o1].copy_(row_o2)
        pos_table[pos_o2].copy_(row_o1)

    preds_after, acc_after = preds_and_acc(X_val, Y_val)

finally:
    # Restore to avoid side effects
    with torch.no_grad():
        pos_table[pos_o1].copy_(row_o1)
        pos_table[pos_o2].copy_(row_o2)

# 3) Measure swap effect specifically on the two outputs (first two positions after SEP)
if L >= 2:
    o1_swapped = (preds_after[:, 0] == preds_before[:, 1])
    o2_swapped = (preds_after[:, 1] == preds_before[:, 0])
    both_swapped_rate = (o1_swapped & o2_swapped).float().mean().item()
else:
    both_swapped_rate = float('nan')

print(f"Acc before: {acc_before:.4f} | Acc after swap: {acc_after:.4f}")
print(f"Exact o1<->o2 swap rate: {both_swapped_rate:.4f} (on first two output positions)")

# Optional: peek at a few examples
for i in range(min(15, X_val.shape[0])):
    print(f"ex{i}: before={preds_before[i, :min(L,2)].tolist()} -> after={preds_after[i, :min(L,2)].tolist()}")


# %%
