# %%
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import os
import copy

import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import pandas as pd, itertools
from tqdm.auto import tqdm

from transformer_lens import HookedTransformer, HookedTransformerConfig, utils
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA

# Configure plotly to use static rendering if widgets fail
import plotly.io as pio

pio.renderers.default = "notebook"

float_formatter = "{:.5f}".format
np.set_printoptions(formatter={"float_kind": float_formatter})

# %%

# ---------- constants ----------
LIST_LEN = 2  # [d1, d2]
SEQ_LEN = LIST_LEN * 2 + 1  # [d1, d2, SEP, o1, o2]

N_DIGITS = 100
DIGITS = list(range(N_DIGITS))  # 100 digits from 0 to 99
PAD = N_DIGITS  # special padding token
SEP = (
    N_DIGITS + 1
)  # special seperator token for the model to think about the input (+1 to avoid confusion with the last digit)
VOCAB = len(DIGITS) + 2  # + the special tokens

D_MODEL = 128
N_HEAD = 1  # 1
N_LAYER = 2  # 2
USE_LN = False  # use layer norm in model
USE_BIAS = False  # use bias in model
FREEZE_WV = True  # no value matrix in attn
FREEZE_WO = (
    True  # no output matrix in attn (i.e. attn head can only copy inputs to outputs)
)
WEIGHT_DECAY = 0.01  # default 0.01

TRAIN_SPLIT = 0.8  # 80% train, 20% test
MAX_TRAIN_STEPS = 300_000  # max training steps

# model name for saving and loading
# MODEL_NAME = f'{N_DIGITS}dig_{D_MODEL}d'
MODEL_NAME = "v2_3layer_100dig_16d"
MODEL_PATH = "models/" + MODEL_NAME + ".pt"

USE_CHECKPOINTING = True  # whether to use checkpointing for training

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

mask_bias = torch.triu(
    torch.ones(SEQ_LEN, SEQ_LEN) * float("-inf")
)  # upper triangular bias mask (lead_diag & above = -inf, rest = 0)
mask_bias[0, 0] = (
    0.0  # don't want a full row of -inf! otherwise we get nan erros & training breaks
)
mask_bias[LIST_LEN + 1 :, :LIST_LEN] = float(
    "-inf"
)  # stop output tokens from attending to input tokens
mask_bias = mask_bias.unsqueeze(0).unsqueeze(
    0
)  # (1,1,T,T) broadcastable across batch and heads

print(mask_bias.cpu()[0][0])

# %%

def get_data():
    # Create all possible combinations of digits
    all_data = list(itertools.product(DIGITS, repeat=LIST_LEN))
    n_data = len(all_data)
    all_data = torch.tensor(all_data, dtype=torch.int64)

    # Create sequences of the form [d1, d2, SEP, d1, d2]
    all_targets = torch.full((n_data, SEQ_LEN), SEP)
    all_targets[:, :LIST_LEN] = all_data
    all_targets[:, LIST_LEN + 1 :] = all_data

    # Create input sequences of the form [d1, d2, SEP, PAD, PAD]
    all_inputs = all_targets.clone()
    all_inputs[:, LIST_LEN + 1 :] = PAD
    return(all_data, all_inputs)


# ---------- data ----------
if __name__ == "__main__":
    all_targets, all_inputs = get_data()
    n_data = all_targets.size(0)

    # Shuffle the dataset (inputs and targets together)
    torch.manual_seed(42)
    perm = torch.randperm(n_data)
    all_inputs = all_inputs[perm]
    all_targets = all_targets[perm]

    train_ds = TensorDataset(
        all_inputs[: int(TRAIN_SPLIT * n_data)], all_targets[: int(TRAIN_SPLIT * n_data)]
    )  # 80% for training
    val_ds = TensorDataset(
        all_inputs[int(TRAIN_SPLIT * n_data) :], all_targets[int(TRAIN_SPLIT * n_data) :]
    )  # 20% for validation
    train_batch_size = min(
        128, len(train_ds)
    )  # Use a batch size of 128 or less if dataset is smaller
    val_batch_size = min(
        256, len(val_ds)
    )  # Use a batch size of 256 or less if dataset is smaller
    train_dl = DataLoader(train_ds, train_batch_size, shuffle=True, drop_last=True)
    val_dl = DataLoader(val_ds, val_batch_size, drop_last=False)

    print("Input:", train_ds[0][0])  # Example input: [d1, d2, SEP, SEP, SEP]
    print("Target:", train_ds[0][1])  # Example target: [d1, d2, SEP, d1, d2]
    len(train_ds), len(val_ds)  # Should be 80% for train and 20% for validation

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
    attn_biases = ["b_Q", "b_K", "b_V", "b_O"]
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


def make_model(
    n_layers=N_LAYER,
    n_heads=N_HEAD,
    d_model=D_MODEL,
    ln=USE_LN,
    use_bias=USE_BIAS,
    freeze_wv=FREEZE_WV,
    freeze_wo=FREEZE_WO,
):
    cfg = HookedTransformerConfig(
        n_layers=n_layers,
        n_heads=n_heads,
        d_model=d_model,
        d_head=d_model // n_heads,
        n_ctx=SEQ_LEN,
        d_vocab=VOCAB,
        attn_only=True,  # no MLP!
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
def save_model(model, path=MODEL_PATH):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")


def load_model(path=MODEL_PATH, device=DEV):
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
            logits = m(inputs.to(DEV))[:, LIST_LEN + 1 :]  # (batch, 2, vocab)
            preds = logits.argmax(-1)
            hits += (preds == targets[:, LIST_LEN + 1 :].to(DEV)).sum().item()
            tots += preds.numel()
    return hits / tots


def train(
    m,
    max_steps=10_000,
    early_stop_acc=0.999,
    checkpoints=False,
    weight_decay=WEIGHT_DECAY,
    verbose=True,
):
    opt = torch.optim.AdamW(m.parameters(), 1e-3, weight_decay=weight_decay)
    ce = torch.nn.CrossEntropyLoss()
    dl = itertools.cycle(train_dl)  # infinite iterator
    for step in tqdm(range(max_steps), desc="Training"):
        inputs, targets = next(dl)
        # get logits/loss for output tokens only
        logits = m(inputs.to(DEV))[:, LIST_LEN + 1 :].reshape(-1, VOCAB)
        loss = ce(logits, targets[:, LIST_LEN + 1 :].reshape(-1).to(DEV))
        loss.backward()
        opt.step()
        opt.zero_grad()
        if (step + 1) % 100 == 0:
            acc = accuracy(m)
            if acc >= early_stop_acc:
                print(
                    f"Early stopping at step {step + 1} with accuracy {acc:.2%} >= {early_stop_acc:.2%}"
                )
                break
            update_every = max(min(10_000, 0.05 * max_steps), 1000)
            if verbose and (step + 1) % update_every == 0:
                print(f"Step {step + 1}, Loss: {loss.item():.4f}, Accuracy: {acc:.2%}")
            if checkpoints and (step + 1) % 50_000 == 0:
                save_model(m, MODEL_PATH)

    print(f"Final accuracy: {accuracy(m):.2%}")


# %%
# LOAD existing or train and SAVE new model

if __name__ == "__main__":
    if os.path.exists(MODEL_PATH):
        model = load_model(MODEL_PATH, device=DEV)
    else:
        print("Training model")
        model = make_model()
        train(model, max_steps=50000, early_stop_acc=1.0, checkpoints=USE_CHECKPOINTING)
        save_model(model, MODEL_PATH)

# %%

# --- Model Parameters Overview ---

if __name__ == "__main__":
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

# --- Using Plotly for visualization ---


def check_attention(m, dataloader, eps=1e-3):
    for inputs, _ in dataloader:
        with torch.no_grad():
            _, cache = m.run_with_cache(inputs.to(DEV))
        for l in range(m.cfg.n_layers):
            pat = cache["pattern", l][:, 0]  # (batch, Q, K)
            leak = pat[:, LIST_LEN + 1 :, :LIST_LEN].sum(
                dim=-1
            )  # mass on forbidden keys
            if (leak > eps).any():
                raise ValueError(
                    f"❌ Layer {l}: output tokens attend to x₁/x₂ by >{eps:.0e}"
                )
    print("✅ no attention leakage onto x₁/x₂")


if __name__ == "__main__":
    sample = val_ds[0][0]  # Example input sequence
    print(
        f"Sample sequence: {sample.cpu().numpy()}"
    )  # Print the sample sequence for reference
    _, cache = model.run_with_cache(sample.unsqueeze(0).to(DEV))

    # --- Create Plotly visualization ---
    token_labels = (
        [f"d{i+1}" for i in range(LIST_LEN)]
        + ["SEP"]
        + [f"o{i+1}" for i in range(LIST_LEN)]
    )
    subplot_titles = [f"Layer {l} Attention Pattern" for l in range(model.cfg.n_layers)]

    fig = make_subplots(
        rows=1,
        cols=model.cfg.n_layers,
        subplot_titles=subplot_titles,
        horizontal_spacing=0.08,  # Add spacing between plots
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
                showscale=(
                    l == model.cfg.n_layers - 1
                ),  # Show colorbar only for the last plot
            ),
            row=1,
            col=l + 1,
        )

    fig.update_layout(
        title_text="Attention Patterns for a Sample Sequence",
        width=1200,
        height=450,
        template="plotly_white",
    )

    # Apply settings to all axes
    fig.update_xaxes(title_text="Key Position")
    fig.update_yaxes(title_text="Query Position", autorange="reversed")

    fig.show()

    check_attention(model, val_dl)

# %% [markdown]
# There is variance in the attention pattern in only 2 positions in each layer,
# with the rest of the attention pattern being fixed
# %%

if __name__ == "__main__":
    all_logits, all_cache = model.run_with_cache(all_inputs.to(DEV), return_type="logits")

    l0_var = all_cache["blocks.0.attn.hook_pattern"].squeeze().var(dim=0, correction=0)
    l1_var = all_cache["blocks.1.attn.hook_pattern"].squeeze().var(dim=0, correction=0)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    im0 = axes[0].imshow(l0_var.cpu(), cmap="viridis")
    axes[0].set_title("Layer 0 Attention Pattern Variance")
    axes[0].set_xlabel("Key Position")
    axes[0].set_ylabel("Query Position")
    fig.colorbar(im0, ax=axes[0])

    im1 = axes[1].imshow(l1_var.cpu(), cmap="viridis")
    axes[1].set_title("Layer 1 Attention Pattern Variance")
    axes[1].set_xlabel("Key Position")
    axes[1].set_ylabel("Query Position")
    fig.colorbar(im1, ax=axes[1])

    plt.tight_layout()
    plt.show()

# %%
# --- Mean Attention Patterns ---

if __name__ == "__main__":
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
    token_labels = (
        [f"d{i+1}" for i in range(LIST_LEN)]
        + ["SEP"]
        + [f"o{i+1}" for i in range(LIST_LEN)]
    )
    subplot_titles = [f"Layer {l} Average Attention" for l in range(model.cfg.n_layers)]

    fig = make_subplots(
        rows=1,
        cols=model.cfg.n_layers,
        subplot_titles=subplot_titles,
        horizontal_spacing=0.08,
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
                showscale=(
                    l == model.cfg.n_layers - 1
                ),  # Show colorbar only for the last plot
            ),
            row=1,
            col=l + 1,
        )

    fig.update_layout(
        title_text="Average Attention Patterns Across Validation Set",
        width=1200,
        height=450,
        template="plotly_white",
    )
    fig.update_xaxes(title_text="Key Position")
    fig.update_yaxes(title_text="Query Position", autorange="reversed")
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
# Visualise a PCA of the embedding matrix: basically nothing interesting to see here.
# %%

# --- PCA on model.W_E embedding matrix ---

if __name__ == "__main__":
    # Get the embedding weights (VOCAB, D_MODEL)
    W_E = model.W_E.detach().cpu().numpy()

    # Run PCA to reduce to 2D
    pca = PCA(n_components=2)
    W_E_2d = pca.fit_transform(W_E)

    # Prepare labels: 0..99, INP, SEP
    labels = [str(i) for i in range(N_DIGITS)] + ["INP", "SEP"]

    # Plot using matplotlib
    plt.figure(figsize=(8, 6))
    plt.scatter(W_E_2d[:N_DIGITS, 0], W_E_2d[:N_DIGITS, 1], c="blue", label="Digits")
    plt.scatter(W_E_2d[N_DIGITS, 0], W_E_2d[N_DIGITS, 1], c="orange", label="INP")
    plt.scatter(W_E_2d[N_DIGITS + 1, 0], W_E_2d[N_DIGITS + 1, 1], c="red", label="SEP")

    # Annotate points
    for i, label in enumerate(labels):
        plt.annotate(label, (W_E_2d[i, 0], W_E_2d[i, 1]), fontsize=8, alpha=0.7)

    plt.title("PCA of Embedding Matrix $W_E$")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend()
    plt.tight_layout()
    plt.show()

# %%

# --- PCA on model.W_pos positional embedding matrix ---

if __name__ == "__main__":
    # Get the positional embedding weights (SEQ_LEN, D_MODEL)
    W_pos = model.W_pos.detach().cpu().numpy()

    # Run PCA to reduce to 2D
    pca_pos = PCA(n_components=2)
    W_pos_2d = pca_pos.fit_transform(W_pos)

    # Prepare labels for each position: [d1, d2, SEP, o1, o2]
    pos_labels = (
        [f"d{i+1}" for i in range(LIST_LEN)]
        + ["SEP"]
        + [f"o{i+1}" for i in range(LIST_LEN)]
    )

    # Plot using matplotlib
    plt.figure(figsize=(8, 6))
    plt.scatter(W_pos_2d[:, 0], W_pos_2d[:, 1], c="green")

    # Annotate points
    for i, label in enumerate(pos_labels):
        plt.annotate(label, (W_pos_2d[i, 0], W_pos_2d[i, 1]), fontsize=10, alpha=0.8)

    plt.title("PCA of Positional Embedding Matrix $W_{pos}$")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.tight_layout()
    plt.show()

# %%

if __name__ == "__main__":
    all_predictions = all_logits.argmax(dim=-1)[:, -2:]
    all_correct = (all_predictions.cpu() == all_targets[:, -2:].cpu()).all(dim=-1)

    # The first attention pattern works strictly on positional embeddings not on
    # token embeddings.
    with torch.no_grad():
        embeds = (
            (all_cache["blocks.0.hook_resid_pre"] @ model.W_K[0].squeeze())
            # (all_cache["blocks.0.hook_resid_pre"] @ model.W_K[0].squeeze())
            .cpu()
            .numpy()[:, :2]
        )
    flat_embeds = embeds.reshape(-1, embeds.shape[-1])  

    pca = PCA(n_components=2)
    embeds_2d = pca.fit_transform(flat_embeds)  
    embeds_2d = embeds_2d.reshape(embeds.shape[0], embeds.shape[1], 2)  

    colors = ["blue", "orange", "green", "red", "purple"]
    labels = [f"pos {i}" for i in range(embeds.shape[1])]

    plt.figure(figsize=(8, 6))
    for pos in range(embeds.shape[1]):
        plt.scatter(
            embeds_2d[:, pos, 0],
            embeds_2d[:, pos, 1],
            s=5,
            color=colors[pos],
            label=labels[pos],
            alpha=0.5,
        )
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("PCA of hook_embed (colored by position index)")
    plt.legend()
    plt.tight_layout()
    plt.show()


# %% [markdown]
# Plot of the attention scores for sep to d1 and d2, coloured by whether the
# model predicted correctly or not. Wondering why the model maps to the diagonal 
# when this results in incorrect answers?

# %%

if __name__ == "__main__":
    # Extract the relevant attention values
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

# %% [markdown]
# SEP, d2, and o2 positional embeddings decode to similar logit distributions,
# that are far from o1 and d1 (which are also very far from one another)

# Compute pos_U
if __name__ == "__main__":
    pos_U = model.W_pos @ model.W_U

    # Run PCA to reduce to 2D
    pca_pos_U = PCA(n_components=2)
    pos_U_2d = pca_pos_U.fit_transform(pos_U.detach().cpu().numpy())

    # Plot using matplotlib
    plt.figure(figsize=(8, 6))
    plt.scatter(pos_U_2d[:, 0], pos_U_2d[:, 1], c="purple")

    # Annotate points with their position labels
    for i, label in enumerate(pos_labels):
        plt.annotate(label, (pos_U_2d[i, 0], pos_U_2d[i, 1]), fontsize=10, alpha=0.8)

    plt.title("PCA of $W_{pos} W_U$")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.tight_layout()
    plt.show()

# %%

# %% Demonstrate the two-slot algorithm on the validation set
if __name__ == "__main__":
    import torch
    import torch.nn.functional as F

    model.eval()

    def cos_batch(a, b):  # (B,d)
        return F.cosine_similarity(a, b, dim=-1)

    wU = model.W_U  # (d_model, vocab)

    write_cos, read_o1_cos, read_o2_cos = [], [], []
    sep_pos, d1_pos, d2_pos, o1_pos, o2_pos = 2, 0, 1, 3, 4

    contrib = {
        "o1_sep_to_d1": [],
        "o1_sep_to_d2": [],
        "o2_sep_to_d1": [],
        "o2_sep_to_d2": [],
        "o2_o1_to_d1": [],
        "o2_o1_to_d2": [],
    }
    acc_o2_orig, acc_o2_no_o1edge = 0, 0
    tots = 0

    with torch.no_grad():
        for inputs, targets in val_dl:
            inputs = inputs.to(DEV)

            logits, cache = model.run_with_cache(inputs, return_type="logits")

            # Residuals
            r0 = cache["hook_embed"] + cache["hook_pos_embed"]                 # (B,T,d)
            rL0 = cache["blocks.0.hook_resid_post"]                            # (B,T,d)
            rL1 = cache["blocks.1.hook_resid_post"]                            # (B,T,d)

            # Attention patterns
            pat0 = cache["blocks.0.attn.hook_pattern"][:, 0]                   # (B,Q,K)
            pat1 = cache["blocks.1.attn.hook_pattern"][:, 0]                   # (B,Q,K)

            # --- Layer 0: write both digits into SEP
            a1 = pat0[:, sep_pos, d1_pos].unsqueeze(-1)                        # (B,1)
            a2 = pat0[:, sep_pos, d2_pos].unsqueeze(-1)                        # (B,1)

            r_sep_pre  = r0[:, sep_pos, :]
            r_d1_pre   = r0[:, d1_pos, :]
            r_d2_pre   = r0[:, d2_pos, :]
            M_true     = rL0[:, sep_pos, :]                                    # r_SEP^(1)
            M_pred     = r_sep_pre + a1*r_d1_pre + a2*r_d2_pre                 # predicted by copy rule

            write_cos.append(cos_batch(M_true, M_pred).cpu())

            # --- Layer 1: read M into o1 and o2
            b1  = pat1[:, o1_pos, sep_pos].unsqueeze(-1)                       # o1<-SEP
            c1  = pat1[:, o2_pos, sep_pos].unsqueeze(-1)                       # o2<-SEP
            d1w = pat1[:, o2_pos, o1_pos].unsqueeze(-1)                        # o2<-o1

            r_o1_pre  = rL0[:, o1_pos, :]
            r_o2_pre  = rL0[:, o2_pos, :]
            r_o1_post = rL1[:, o1_pos, :]
            r_o2_post = rL1[:, o2_pos, :]

            r_o1_pred = r_o1_pre + b1*M_true
            r_o2_pred = r_o2_pre + c1*M_true + d1w*r_o1_pre

            read_o1_cos.append(cos_batch(r_o1_post, r_o1_pred).cpu())
            read_o2_cos.append(cos_batch(r_o2_post, r_o2_pred).cpu())

            # --- Direct logit attribution (per-source contributions)
            d1_tok = inputs[:, d1_pos]                                         # (B,)
            d2_tok = inputs[:, d2_pos]

            w_d1 = wU[:, d1_tok].T                                             # (B,d)
            w_d2 = wU[:, d2_tok].T

            def dot(v, w):  # (B,d)·(B,d)->(B,)
                return (v * w).sum(dim=-1)

            # o1 logits
            contrib["o1_sep_to_d1"].append(dot(b1*M_true, w_d1).cpu())
            contrib["o1_sep_to_d2"].append(dot(b1*M_true, w_d2).cpu())

            # o2 logits
            contrib["o2_sep_to_d1"].append(dot(c1*M_true,    w_d1).cpu())
            contrib["o2_sep_to_d2"].append(dot(c1*M_true,    w_d2).cpu())
            contrib["o2_o1_to_d1"].append(dot(d1w*r_o1_pre,  w_d1).cpu())
            contrib["o2_o1_to_d2"].append(dot(d1w*r_o1_pre,  w_d2).cpu())

            # --- Counterfactual: remove the o2<-o1 edge
            r_o2_no_o1 = r_o2_post - d1w*r_o1_pre
            logits_o2_cf = r_o2_no_o1 @ wU
            pred_o2_cf = logits_o2_cf.argmax(dim=-1)

            pred_o2 = logits[:, o2_pos, :].argmax(dim=-1)
            acc_o2_orig     += (pred_o2    == d2_tok).sum().item()
            acc_o2_no_o1edge += (pred_o2_cf == d2_tok).sum().item()
            tots += inputs.size(0)

    # --- Summary
    write_cos = torch.cat(write_cos)
    read_o1_cos = torch.cat(read_o1_cos)
    read_o2_cos = torch.cat(read_o2_cos)

    print(f"L0 write recon cos  : mean {write_cos.mean():.4f}  median {write_cos.median():.4f}")
    print(f"L1 read o1 recon cos: mean {read_o1_cos.mean():.4f}  median {read_o1_cos.median():.4f}")
    print(f"L1 read o2 recon cos: mean {read_o2_cos.mean():.4f}  median {read_o2_cos.median():.4f}")

    for k in contrib:
        contrib[k] = torch.cat(contrib[k])
    print("\nDirect logit contributions (means):")
    print(f"  o1: SEP→d1 {contrib['o1_sep_to_d1'].mean():.4f}, SEP→d2 {contrib['o1_sep_to_d2'].mean():.4f}")
    print(f"  o2: SEP→d2 {contrib['o2_sep_to_d2'].mean():.4f}, SEP→d1 {contrib['o2_sep_to_d1'].mean():.4f}")
    print(f"  o2: o1 →d1 {contrib['o2_o1_to_d1'].mean():.4f}  (expect <0),  o1→d2 {contrib['o2_o1_to_d2'].mean():.4f}")

    frac_neg = (contrib["o2_o1_to_d1"] < 0).float().mean().item()
    print(f"\nFrac(o2: o1→d1 contribution < 0): {frac_neg:.3f}")

    print(f"\nSecond-digit accuracy:")
    print(f"  original : {acc_o2_orig/tots:.4f}")
    print(f"  no o2←o1 : {acc_o2_no_o1edge/tots:.4f}  (counterfactual by removing that path)")
