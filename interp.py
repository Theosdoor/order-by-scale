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

# Configure plotly to use static rendering if widgets fail
import plotly.io as pio
pio.renderers.default = "notebook"

float_formatter = "{:.5f}".format
np.set_printoptions(formatter={'float_kind':float_formatter})


# %% [markdown]
# ## Model

# %%
# ---------- constants ----------
MODEL_NAME = 'v2_2layer_100dig_128d'
MODEL_PATH = "models/" + MODEL_NAME + ".pt"

DATASET_NAME = "listlen2_digits100_dupes"
# listlen2_digits10_dupes
# listlen2_digits10_nodupes
# listlen2_digits100_dupes_traindupesonly
# listlen2_digits100_dupes
# listlen2_digits100_nodupes

LIST_LEN = 2 # [d1, d2]
SEQ_LEN = LIST_LEN * 2 + 1 # [d1, d2, SEP, o1, o2]

N_DIGITS = 100
DIGITS = list(range(N_DIGITS)) # 100 digits from 0 to 99
PAD = N_DIGITS # special padding token
SEP = N_DIGITS + 1 # special seperator token for the model to think about the input (+1 to avoid confusion with the last digit)
VOCAB = len(DIGITS) + 2  # + the special tokens

# For backward compatibility with older versions
if MODEL_NAME.startswith('v1_'):
    USE_PAD = False  # whether to use the PAD token in the input sequences (or just SEP)
    VOCAB -= 1  # -1 for the PAD token
else:
    USE_PAD = True

D_MODEL = 128
N_HEAD = 1 # 1
N_LAYER = 2 # 2
USE_LN = False # use layer norm in model
USE_BIAS = False # use bias in model
FREEZE_WV = True # no value matrix in attn 
FREEZE_WO = True # no output matrix in attn (i.e. attn head can only copy inputs to outputs)

DEV = (
    "cuda"
    if torch.cuda.is_available()
    else ("mps" if torch.backends.mps.is_available() else "cpu")
)
device = DEV
torch.manual_seed(0)


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


# %%
# Check train set
train_ds[:5]

# %% [markdown]
# 
# 
# Bias not needed

# %%
# LOAD existing model
if os.path.exists(MODEL_PATH):
    model = load_model(MODEL_PATH, device=DEV)
else:
    raise FileNotFoundError(f"Model file {MODEL_PATH} does not exist. Please train the model first.")

# from torchinfo import summary
# summary(model) 
accuracy(model)

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


sample = val_ds[0][0] # Example input sequence
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

print("Accuracy with avg-attn:", accuracy(model_with_avg_attn))

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
# First, get all the data and run the model to get cache and predictions
all_inputs = val_ds.tensors[0].to(DEV)
all_targets = val_ds.tensors[1].to(DEV)

# Run model on all validation data
with torch.no_grad():
    all_logits, all_cache = model.run_with_cache(all_inputs, return_type="logits")

# Calculate which predictions are correct
all_predictions = all_logits.argmax(dim=-1)[:, -LIST_LEN:]  # Last LIST_LEN positions
all_correct = (all_predictions == all_targets[:, -LIST_LEN:]).all(dim=-1).cpu().numpy()

# Extract the relevant attention scores (not patterns)
# Layer 0, query=SEP (position 2), keys=d1,d2 (positions 0,1)
attn_vals = all_cache["attn_scores", 0].squeeze()[:, 2, [0, 1]].cpu().numpy()

# Scatter plot: x = attention to d1 (pos 0), y = attention to d2 (pos 1), color by correctness
plt.figure(figsize=(6, 6))
plt.scatter(attn_vals[:, 0], attn_vals[:, 1], c=all_correct, cmap="coolwarm", alpha=0.5)
plt.xlabel("Attention to d1 (pos 0)")
plt.ylabel("Attention to d2 (pos 1)")
plt.title("Scatter plot of Layer 0, Query=SEP, Attention to d1 vs d2")
plt.grid(True)
plt.tight_layout()
plt.colorbar(label="Correct (1=True, 0=False)")
plt.show()

print(f"Total samples: {len(all_correct)}")
print(f"Correct predictions: {all_correct.sum()}")
print(f"Accuracy: {all_correct.mean():.3f}")

# %%
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

# Collect residual stream (embed + post-resid after each layer)
resid_keys = ["hook_embed"] + [f"blocks.{l}.hook_resid_post" for l in range(model.cfg.n_layers)]
resid_values = torch.stack([cache[k] for k in resid_keys], dim=0)  # [L+1, 1, seq, d_model]

# Get W_U (compatibly)
W_U = getattr(model, "W_U", model.unembed.W_U)

# Logit lens: decode most likely token at each position after each layer
position_tokens = (resid_values @ W_U).squeeze(1).argmax(-1)  # [L+1, seq]

L, N, _ = att.shape

# Simple layered layout
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
threshold = 0.05  # ignore tiny weights to reduce clutter
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
ax.set_xticklabels(["Input"] + [f"After L{l+1}" for l in range(model.cfg.n_layers)])
ax.set_yticks(y_positions)
position_names = ["d1", "d2", "SEP", "o1", "o2"]
ax.set_yticklabels(position_names)
ax.set_xlim(-0.5, L + 0.5)
ax.set_ylim(-0.5, N - 0.5)
ax.set_title("Attention Flow Across Layers")
ax.grid(False)
ax.set_aspect("auto")
for spine in ax.spines.values():
    spine.set_visible(False)

# Legend: dotted = residual stream, solid = attention
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
plt.show()


# %% [markdown]
# ### W_pos

# %%
# --- Positional Encoding Ablation ---

print("--- Positional Encoding Ablation Results ---")
print(f"Original Loss: {original_loss.item():.4f}, Original Accuracy: {original_accuracy.item():.4f}")
print(f"Original prediction (sample {sample_idx}): {original_predictions[sample_idx].cpu().numpy()}")
print("-" * 60)

# Hook to subtract positional encodings from the residual stream
def ablate_pos_encoding_hook(resid, hook):
    # resid shape: [batch, seq_pos, d_model]
    # W_pos shape: [seq_pos, d_model]
    # We subtract the positional embeddings from the residual stream.
    # W_pos is automatically broadcast across the batch dimension.
    result = resid - model.W_pos
    # Restore some positions to their original values
    idx = 2
    result[:, idx] = resid[:, idx]
    return result

# Define the ablation experiments: a description and the hook point
ablation_points = [
    ("Before Layer 0", "blocks.0.hook_resid_pre"),
    ("After Layer 0", "blocks.0.hook_resid_post"), # i.e. pre layer 1
    ("After Layer 1", "blocks.1.hook_resid_post"),
]

# --- Perform Ablation for Each Case ---
for description, hook_name in ablation_points:
    with torch.no_grad():
        # Run the model with the ablation hook
        ablated_logits = model.run_with_hooks(
            val_inputs,
            return_type="logits",
            fwd_hooks=[(hook_name, ablate_pos_encoding_hook)]
        )
    
    # Calculate ablated loss on the output tokens
    output_logits_ablated = ablated_logits[:, LIST_LEN+1:]
    ablated_loss = loss_fn(output_logits_ablated.reshape(-1, VOCAB), output_targets.reshape(-1))
    
    # Calculate ablated accuracy on the output tokens
    ablated_predictions = ablated_logits.argmax(dim=-1)
    ablated_output_predictions = ablated_predictions[:, LIST_LEN+1:]
    ablated_accuracy = (ablated_output_predictions == output_targets).float().mean()
    
    print(f"Ablating Positional Encodings {description}:")
    print(f"  Ablated Loss: {ablated_loss.item():.4f}")
    print(f"  Ablated Accuracy: {ablated_accuracy.item():.4f}")
    print(f"  Ablated prediction (sample {sample_idx}):  {ablated_predictions[sample_idx].cpu().numpy()}")
    print("-" * 60)

# %% [markdown]
# ### Residual stream

# %%
def ablate_skip_connection(layer_to_ablate):
    """
    Ablates the skip connection over the attention block for a specific layer.
    """
    
    # This dictionary will store the input residual stream for the layer
    captured_resid_pre = {}

    # Hook to capture the input to the attention block
    def capture_resid_pre_hook(resid, hook):
        captured_resid_pre['value'] = resid
        return resid

    # Hook to ablate the skip connection by subtracting the captured input
    def ablate_skip_hook(resid, hook):
        # resid here is resid_pre + attn_out
        # We subtract resid_pre to leave only attn_out
        result = resid - captured_resid_pre['value']
        # result[:, 3:] = resid[:, 3:]
        return result

    resid_pre_hook_name = f"blocks.{layer_to_ablate}.hook_resid_pre"
    hook_attn_out_name = f"blocks.{layer_to_ablate}.hook_attn_out"
    resid_post_hook_name = f"blocks.{layer_to_ablate}.hook_resid_post"

    with torch.no_grad():
        ablated_logits = model.run_with_hooks(
            val_inputs,
            return_type="logits",
            fwd_hooks=[
                (resid_pre_hook_name, capture_resid_pre_hook),
                (resid_post_hook_name, ablate_skip_hook)
            ]
        )
    return ablated_logits

print(f"--- Attention Skip Connection Ablation Results ---")
print(f"Sample sequence: {val_inputs[sample_idx].cpu().numpy()}") # last sample in validation set
print(f"Original Loss: {original_loss.item():.4f}")
print(f"Original Accuracy: {original_accuracy.item():.4f}")
print(f"Original predictions: {original_predictions[sample_idx].cpu().numpy()}")
print("-" * 50)

# --- Perform Ablation for Each Layer ---
for l in range(N_LAYER):
    ablated_logits = ablate_skip_connection(l)
    output_logits_ablated = ablated_logits[:, LIST_LEN+1:]
    ablated_loss = loss_fn(output_logits_ablated.reshape(-1, VOCAB), output_targets.reshape(-1))
    ablated_predictions = ablated_logits.argmax(dim=-1)
    
    # --- Calculate Ablated Accuracy ---
    ablated_output_predictions = ablated_predictions[:, LIST_LEN+1:]
    ablated_accuracy = (ablated_output_predictions == output_targets).float().mean()

    print(f"Ablating Skip Connection at Layer {l}:")
    print(f"  Ablated Loss: {ablated_loss.item():.4f}")
    print(f"  Loss Increase: {(ablated_loss - original_loss).item():.4f}")
    print(f"  Ablated Accuracy: {ablated_accuracy.item():.4f}")
    print(f"  Ablated predictions: {ablated_predictions[sample_idx].cpu().numpy()}")
    print("-" * 50)

# %%
# --- Ablate Skip Connections for Layer 0 and 1 Simultaneously ---

# This dictionary will store the input residual streams for each layer
captured_resid_pres = {}

# Hook to capture the input to an attention block
def capture_resid_pre_hook(resid, hook):
    layer_idx = hook.layer()
    captured_resid_pres[layer_idx] = resid
    return resid

# Hook to ablate the skip connection by subtracting the captured input
def ablate_skip_hook(resid, hook):
    layer_idx = hook.layer()
    # Subtract the captured resid_pre for the corresponding layer
    result = resid - captured_resid_pres[layer_idx]
    idx = -2
    # result[:, -2:] = resid[:, -2:] # keep some tokens intact (last 2)
    return result

# Define the hooks for both layers
fwd_hooks = []
for l in range(model.cfg.n_layers):
    resid_pre_hook_name = f"blocks.{l}.hook_resid_pre"
    resid_post_hook_name = f"blocks.{l}.hook_resid_post"
    fwd_hooks.extend([
        (resid_pre_hook_name, capture_resid_pre_hook),
        (resid_post_hook_name, ablate_skip_hook)
    ])

# Run the model with both skip connections ablated
with torch.no_grad():
    ablated_logits = model.run_with_hooks(
        val_inputs,
        return_type="logits",
        fwd_hooks=fwd_hooks
    )
    output_logits_ablated = ablated_logits[:, LIST_LEN+1:]
    ablated_loss = loss_fn(output_logits_ablated.reshape(-1, VOCAB), output_targets.reshape(-1))
    ablated_predictions = ablated_logits.argmax(dim=-1)
    
    # --- Calculate Ablated Accuracy ---
    ablated_output_predictions = ablated_predictions[:, LIST_LEN+1:]
    ablated_accuracy = (ablated_output_predictions == output_targets).float().mean()


print(f"--- Ablating All Skip Connections ---")
print(f"Validation set size: {len(val_inputs)} samples")
print("-" * 50)
print(f"{'Metric':<12} | {'Original':<10} | {'Ablated':<10}")
print("-" * 50)
print(f"{'Loss':<12} | {original_loss.item():<10.4f} | {ablated_loss.item():<10.4f}")
print(f"{'Accuracy':<12} | {original_accuracy.item():<10.4f} | {ablated_accuracy.item():<10.4f}")
print("-" * 50)
print(f"Example from {sample_idx}th validation sample:")
print(f"  Sample sequence:      {val_inputs[sample_idx].cpu().numpy()}")
print(f"  Original predictions: {original_predictions[sample_idx].cpu().numpy()}")
print(f"  Ablated predictions:  {ablated_predictions[sample_idx].cpu().numpy()}")

# %%
# --- Mean Ablation of Skip Connections ---

# 1. Cache the 'resid_pre' activations for each layer across the validation set
resid_pre_hook_names = [f"blocks.{l}.hook_resid_pre" for l in range(model.cfg.n_layers)]
with torch.no_grad():
    _, cache = model.run_with_cache(val_inputs, names_filter=lambda name: name in resid_pre_hook_names)

# 2. Calculate the mean of these activations
mean_resid_pres = {}
for l in range(model.cfg.n_layers):
    mean_resid_pres[l] = cache[resid_pre_hook_names[l]].mean(dim=(0, 1))

# --- Define hooks for mean ablation ---
captured_resid_pre = {}

def capture_resid_pre_hook(resid, hook):
    """Saves the current resid_pre to be subtracted in the next hook."""
    captured_resid_pre[hook.layer()] = resid
    return resid

def mean_ablate_skip_hook(resid, hook):
    """Replaces the skip connection with its mean value."""
    layer_idx = hook.layer()
    # resid_post = resid_pre + block_output
    # We want: mean_resid_pre + block_output
    # So we calculate: resid_post - resid_pre + mean_resid_pre
    return resid - captured_resid_pre[layer_idx] + mean_resid_pres[layer_idx]

# --- Function to calculate loss and accuracy ---
loss_fn = torch.nn.CrossEntropyLoss()

def calculate_metrics(logits, targets):
    """Calculates loss and accuracy for the output tokens."""
    output_logits = logits[:, LIST_LEN+1:]
    output_targets = targets[:, LIST_LEN+1:]
    
    loss = loss_fn(output_logits.reshape(-1, VOCAB), output_targets.reshape(-1)).item()
    
    predictions = output_logits.argmax(dim=-1)
    accuracy = (predictions == output_targets).float().mean().item()
    
    return loss, accuracy

# --- Evaluate metrics for each ablation case ---

print("--- Skip Connection Mean Ablation Metrics ---")

# Original metrics
with torch.no_grad():
    original_logits = model(val_inputs)
    og_loss, original_acc = calculate_metrics(original_logits, val_targets)
print(f"Original -> Loss: {og_loss:.4f}, Accuracy: {original_acc:.2%}")
print("-" * 50)

# Ablate each layer individually
for l in range(model.cfg.n_layers):
    fwd_hooks = [
        (f"blocks.{l}.hook_resid_pre", capture_resid_pre_hook),
        (f"blocks.{l}.hook_resid_post", mean_ablate_skip_hook)
    ]
    with torch.no_grad():
        ablated_logits = model.run_with_hooks(val_inputs, fwd_hooks=fwd_hooks)
        ablated_loss, ablated_acc = calculate_metrics(ablated_logits, val_targets)
    print(f"Ablating Layer {l} Skip -> Loss: {ablated_loss:.4f}, Accuracy: {ablated_acc:.2%}")

# Ablate all layers simultaneously
fwd_hooks = []
for l in range(model.cfg.n_layers):
    fwd_hooks.extend([
        (f"blocks.{l}.hook_resid_pre", capture_resid_pre_hook),
        (f"blocks.{l}.hook_resid_post", mean_ablate_skip_hook)
    ])
with torch.no_grad():
    ablated_logits = model.run_with_hooks(val_inputs, fwd_hooks=fwd_hooks)
    ablated_loss, ablated_acc = calculate_metrics(ablated_logits, val_targets)
print(f"Ablating All Skips -> Loss: {ablated_loss:.4f}, Accuracy: {ablated_acc:.2%}")
print("-" * 50)

# %% [markdown]
# ### W_E, W_U, W_pos

# %%
# https://umap-learn.readthedocs.io/en/latest/parameters.html
N_DIM_VIS = 2  # <-- CHANGE THIS VALUE to 2 or 3 to switch visualizations
umap_n_neighbors = min(5, VOCAB - 1)  # Use smaller n_neighbors for small dataset (max VOCAB-1 = 10)
umap_min_dist = 0.1  # Spread points out more
umap_metric = 'euclidean' # default: euclidean 

def visualize_w_e_and_w_u(model):
    """
    Extracts W_E and W_U, applies UMAP to get 2D or 3D projections,
    and creates side-by-side interactive plots based on N_DIM_VIS.
    """
    if N_DIM_VIS not in [2, 3]:
        raise ValueError("N_DIM_VIS must be set to 2 or 3.")
    

    print(f"\nStarting {N_DIM_VIS}D UMAP visualization  for W_E and W_U... \n(n_neighbors={umap_n_neighbors}, min_dist={umap_min_dist}, metric={umap_metric})")
    model.eval()

    w_e = model.embed.W_E.detach().cpu().numpy()
    w_u = model.unembed.W_U.T.detach().cpu().numpy()
    
    reducer = umap.UMAP(
        n_neighbors=umap_n_neighbors,
        min_dist=umap_min_dist,
        n_components=N_DIM_VIS,
        random_state=42, # UMAP is stochastic, so we set a seed for reproducibility
        metric=umap_metric,  # Use Euclidean distance for UMAP
        # verbose=True,
    )

    w_e_proj = reducer.fit_transform(w_e)
    w_u_proj = reducer.fit_transform(w_u)
    labels = [str(d) for d in DIGITS] + ['SEP']

    # --- Find common axis ranges across both projections ---
    all_proj = np.vstack([w_e_proj, w_u_proj])
    min_vals = all_proj.min(axis=0)
    max_vals = all_proj.max(axis=0)
    
    # Add a 10% margin for better visualization
    margin = (max_vals - min_vals) * 0.1
    ranges = [(min_v - m, max_v + m) for min_v, max_v, m in zip(min_vals, max_vals, margin)]

    if N_DIM_VIS == 3:
        fig = make_subplots(
            rows=1, cols=2,
            specs=[[{'type': 'scene'}, {'type': 'scene'}]],
            subplot_titles=('3D UMAP of W_E (Embeddings)', '3D UMAP of W_U (Unembeddings)')
        )
        fig.add_trace(go.Scatter3d(
            x=w_e_proj[:, 0], y=w_e_proj[:, 1], z=w_e_proj[:, 2],
            mode='markers+text', text=labels, textfont=dict(size=10, color='black'),
            marker=dict(size=5, color=list(range(VOCAB)), colorscale='viridis'),
            hoverinfo='text',
            hovertext=[f'Token: {l}<br>x: {x:.2f}, y: {y:.2f}, z: {z:.2f}' for l, x, y, z in zip(labels, w_e_proj[:, 0], w_e_proj[:, 1], w_e_proj[:, 2])],
            showlegend=False
        ), row=1, col=1)
        fig.add_trace(go.Scatter3d(
            x=w_u_proj[:, 0], y=w_u_proj[:, 1], z=w_u_proj[:, 2],
            mode='markers+text', text=labels, textfont=dict(size=10, color='black'),
            marker=dict(
                size=5, color=list(range(VOCAB)), colorscale='viridis', showscale=True,
                colorbar=dict(title="Token ID", tickvals=list(range(VOCAB)), ticktext=labels)
            ),
            hoverinfo='text',
            hovertext=[f'Token: {l}<br>x: {x:.2f}, y: {y:.2f}, z: {z:.2f}' for l, x, y, z in zip(labels, w_u_proj[:, 0], w_u_proj[:, 1], w_u_proj[:, 2])]
        ), row=1, col=2)
        fig.update_layout(title_text='3D UMAP Projections', height=700, width=1400)
        # Apply the same axis ranges to both 3D scenes
        fig.update_scenes(
            xaxis_title_text='Dim 1', yaxis_title_text='Dim 2', zaxis_title_text='Dim 3',
            xaxis_range=ranges[0], yaxis_range=ranges[1], zaxis_range=ranges[2]
        )
    else:  # N_DIM_VIS == 2
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('2D UMAP of W_E (Embeddings)', '2D UMAP of W_U (Unembeddings)')
        )
        fig.add_trace(go.Scatter(
            x=w_e_proj[:, 0], y=w_e_proj[:, 1],
            mode='markers+text', text=labels, textposition='top center',
            marker=dict(size=10, color=list(range(VOCAB)), colorscale='viridis'),
            hoverinfo='text',
            hovertext=[f'Token: {l}<br>x: {x:.3f}, y: {y:.3f}' for l, x, y in zip(labels, w_e_proj[:, 0], w_e_proj[:, 1])],
            showlegend=False
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=w_u_proj[:, 0], y=w_u_proj[:, 1],
            mode='markers+text', text=labels, textposition='top center',
            marker=dict(
                size=10, color=list(range(VOCAB)), colorscale='viridis', showscale=True,
                colorbar=dict(title="Token ID", tickvals=list(range(VOCAB)), ticktext=labels)
            ),
            hoverinfo='text',
            hovertext=[f'Token: {l}<br>x: {x:.3f}, y: {y:.3f}' for l, x, y in zip(labels, w_u_proj[:, 0], w_u_proj[:, 1])]
        ), row=1, col=2)
        fig.update_layout(title_text='2D UMAP Projections', height=600, width=1200, template='plotly_white')
        # Apply the same axis ranges to both 2D plots
        fig.update_xaxes(title_text="UMAP Dim 1", range=ranges[0])
        fig.update_yaxes(title_text="UMAP Dim 2", range=ranges[1])

    fig.show()

# visualize_w_e_and_w_u(model)


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
    token_labels = [str(d) for d in DIGITS] + ['SEP', 'PAD']
    if not USE_PAD:  # backward compatibility
        token_labels.remove('PAD')

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
layer_to_ablate = 0 # output digits do nothijg in layer 0
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
    # out[:, head_index_to_ablate, :2, :] = 0.0
    out[:, head_index_to_ablate, 3:, :] = 0.0
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
# token-wise equality matrix [N, 2]
eq = (ablated_output_predictions == output_targets)

num_tokens = eq.numel()
num_correct_tokens = eq.sum().item()
token_acc = num_correct_tokens / num_tokens

both_right = (eq.sum(dim=1) == 2).sum().item()
one_right  = (eq.sum(dim=1) == 1).sum().item()
both_wrong = (eq.sum(dim=1) == 0).sum().item()

# Breakdown of which single position is correct
o1_only_mask = eq[:, 0] & ~eq[:, 1]  # o1 correct, o2 wrong
o2_only_mask = eq[:, 1] & ~eq[:, 0]  # o2 correct, o1 wrong
o1_only = o1_only_mask.sum().item()
o2_only = o2_only_mask.sum().item()

num_sequences = eq.shape[0]
seq_failures = (eq.sum(dim=1) < 2).sum().item()
seq_acc = both_right / num_sequences

print(f"Token acc: {token_acc:.3f}  ({num_correct_tokens}/{num_tokens})")
print(f"Both right: {both_right}, One right: {one_right}, Both wrong: {both_wrong}")
print(f"  - o1 correct, o2 wrong: {o1_only}")
print(f"  - o2 correct, o1 wrong: {o2_only}")
print(f"Seq acc (both outputs correct): {seq_acc:.3f}  -> failures: {seq_failures}/{num_sequences}")

# %%
# --- Analyze Failure Cases ---
# Find indices where the ablated prediction is incorrect
is_incorrect = (ablated_output_predictions != output_targets).any(dim=1)
error_indices = torch.where(is_incorrect)[0]

print(f"\n--- Analysis of {len(error_indices)} Failure Cases ---")
if len(error_indices) > 0:
    # Limit the number of printed examples for readability
    n_examples_to_show = min(10, len(error_indices))
    print(f"Showing the first {n_examples_to_show} incorrect predictions:")
    
    for i in range(n_examples_to_show):
        idx = int(error_indices[i])
        full_sequence = val_inputs[idx].cpu().numpy()
        input_digits = full_sequence[:LIST_LEN]
        correct_output = output_targets[idx].cpu().numpy()
        predicted_output = ablated_output_predictions[idx].cpu().numpy()


        # if correct_output[0] == correct_output[1]:
        #     continue
        # else:
        #     print(predicted_output)

        print(f"\nExample {i+1} (Index: {idx}):")
        print(f"  Input Digits:     {input_digits}")
        print(f"  Correct Output:   {correct_output}")
        print(f"  Predicted Output: {predicted_output} <--- ERROR")


    # TEST
    bad_preds = ablated_output_predictions[error_indices].cpu().numpy()
    bad_preds_2 = [] # non duped
    c = 0
    for p in bad_preds:
        if p[0] == p[1]:
            c+=1
        else:
            bad_preds_2.append(p)
    print(f'{c} duped')
else:
    print("No incorrect predictions found after ablation.")

len(bad_preds_2) # not duped

# %%
is_correct = (ablated_output_predictions == output_targets).all(dim=1)
is_correct_ids = torch.where(is_correct)[0]
n_eg = min(5, is_correct_ids.numel())
print(len(is_correct_ids), "correct predictions in the validation set")

for i in range(n_eg):
    idx = int(is_correct_ids[i])
    full_sequence = val_inputs[idx].cpu().numpy()
    input_digits = full_sequence[:LIST_LEN]
    correct_output = output_targets[idx].cpu().numpy()
    predicted_output = ablated_output_predictions[idx].cpu().numpy()
    
    # if correct_output[0] == correct_output[1]:
    #     continue
    # else:
    #     print(predicted_output)
    print(f"\nExample {i+1} (Index: {idx}):")
    print(f"  Input Digits:     {input_digits}")
    print(f"  Correct Output:   {correct_output}")
    print(f"  Predicted Output: {predicted_output} <--- CORRECT")

# %%
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
    # Layer 1: zero attention for queries o1/o2 (positions 3,4) to keys d1/d2 (positions 0,1)
    1: build_qk_mask(positions=[(4, 2), (4,3)], seq_len=SEQ_LEN),
    # Layer 2: zero attention at specific entries (q=4,k=3) and (q=3,k=2) as an example
    0: build_qk_mask(positions=[(4, 2), (4,3)], seq_len=SEQ_LEN),
}

# Apply to a single head or all heads
head = None  # Set to None to affect all heads, or specify a head index (e.g., 0)

# Build hooks
fwd_hooks = []
for layer_idx, mask in layers_to_ablate.items():
    hook_name = utils.get_act_name("pattern", layer_idx)
    fwd_hooks.append((hook_name, make_pattern_hook(mask, head_index=head, set_to=0.0, renorm=True)))

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


