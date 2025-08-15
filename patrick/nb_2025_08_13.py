# %% [markdown]
# # List copying in superposition
# We investigate the problem of copying lists with a transformer in a setting where the list must be compressed into a single activation, i.e. superposing the input digit values.

# %%
import itertools
import numpy as np
import torch
import torch
import numpy as np
import os
from functools import partial
%matplotlib widget
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from tqdm.auto import tqdm

from nb_2025_08_12 import MODEL_PATH, get_data, LIST_LEN
from train import load_model
from grid_search import make_validation_set
from sklearn.decomposition import PCA
import pickle
from sklearn.svm import SVC

# Configure plotly to use static rendering if widgets fail
import plotly.io as pio

pio.renderers.default = "notebook"

float_formatter = "{:.5f}".format
np.set_printoptions(formatter={"float_kind": float_formatter})

DEV = (
    "cuda"
    if torch.cuda.is_available()
    else ("mps" if torch.backends.mps.is_available() else "cpu")
)
device = DEV
torch.manual_seed(0)

# %% [markdown]
# ## Toy model
# A 3-layer transformer can perfectly copy a 2-digit list. However, decreasing the number of layers to 2 decreases the model performance to around 90%.

# %%


def val(pred, target):
    return (pred[:, -LIST_LEN:] == target[:, -LIST_LEN:]).float().mean().item()


MODEL_PATH = "models/copytask_L2_ND100_DM128_H1_NL2_LN0_B0_FWV1_FWO1_WD0p01_RUN0.pt"

if __name__ == "__main__":
    model, _, _ = load_model(MODEL_PATH)
    inputs, data = make_validation_set(8192, 100, 2, device, 0)
    logits, cache = model.run_with_cache(inputs, return_type="logits")

    assert val(data, data) == 1.0
    assert val(inputs, data) == 0.0
    original_performance = val(logits.argmax(-1), data)
    print(f"Original performance: {original_performance}")

    correct = (logits.argmax(-1)[:, -LIST_LEN:] == data[:, -LIST_LEN:]).all(-1).cpu()

# %% [markdown]
# Plotting the attention scores by layer of the model lets us see how information flows through the model.

# %%


def plot_attention_flow(model, inputs, device, position_names=None, threshold=0.05):
    """
    Plots the attention flow across layers for a single input sequence.

    Args:
        model: The transformer model.
        inputs: The input tensor (should be a batch, e.g., shape [B, L]).
        device: The device to run the model on.
        position_names: Optional list of token names for y-axis.
        threshold: Minimum attention weight to draw an edge.
    """
    # Run model and get cache for a single example
    _, cache = model.run_with_cache(
        inputs[3479].unsqueeze(0).to(device), return_type="logits"
    )

    att = (
        torch.stack(
            [
                cache[f"blocks.{layer}.attn.hook_pattern"]
                for layer in range(model.cfg.n_layers)
            ],
            dim=0,
        )
        .cpu()
        .numpy()
        .squeeze()
    )

    # Get pre and post residuals for all layers
    resid_keys = ["hook_embed"] + [
        f"blocks.{l}.hook_resid_post" for l in range(model.cfg.n_layers)
    ]
    resid_values = torch.stack([cache[k] for k in resid_keys], dim=0)

    position_tokens = (resid_values @ model.W_U).squeeze().argmax(-1)

    L, N, _ = att.shape

    # Layout
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
    ax.set_xticklabels(["Input"] + [f"After L{l+1}" for l in range(model.cfg.n_layers)])
    ax.set_yticks(y_positions)
    if position_names is None:
        position_names = [str(i) for i in range(N)]
    ax.set_yticklabels(position_names)
    ax.set_xlim(-0.5, L + 0.5)
    ax.set_ylim(-0.5, N - 0.5)
    ax.set_title("Attention Flow Across Layers (with Logit Lens)")
    ax.grid(False)
    ax.set_aspect("auto")
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Legend: dotted = residual stream, solid = attention
    legend_elements = [
        Line2D(
            [0],
            [0],
            linestyle="--",
            color="gray",
            lw=1.5,
            label="Residual stream (dotted)",
        ),
        Line2D(
            [0], [0], linestyle="-", color="black", lw=1.5, label="Attention (solid)"
        ),
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


POSITION_NAMES = ["d1", "d2", "SEP", "o1", "o2"]

if __name__ == "__main__":
    plot_attention_flow(model, inputs, device, position_names=POSITION_NAMES)

# %% [markdown]
# We can ablate a number of attention values with no impact on model performance.

# %%


def ablate_qk_hook(pttn, hook=None, q=None, k=None):
    assert q is not None and k is not None, "q and k must be specified"

    pttn[:, 0, q, k] = 0.0
    return pttn


def ablate_pttn_hook(pttn, hook=None, ablate_pttn=None):
    assert ablate_pttn is not None, "ablate_pttn must be specified"

    pttn[:, :, ablate_pttn] = 0.0
    return pttn


if __name__ == "__main__":
    all_patterns = torch.cat(
        [cache["pattern", i] for i in range(LIST_LEN)], dim=1
    )  # [N, L, S, S]
    ablated = torch.zeros_like(all_patterns[0]).to(torch.bool)

    for l, q, k in itertools.product(*[range(s) for s in ablated.shape]):
        pttn_ablate_logits = model.run_with_hooks(
            inputs,
            fwd_hooks=[
                (f"blocks.{l}.attn.hook_pattern", partial(ablate_qk_hook, q=q, k=k)),
            ],
        )
        pttn_ablate_perf = val(pttn_ablate_logits.argmax(-1), data)
        if abs(pttn_ablate_perf - original_performance) < 1e-4:
            ablated[l, q, k] = True

    ablate_pttn_logits = model.run_with_hooks(
        inputs,
        fwd_hooks=[
            (
                f"blocks.{l}.attn.hook_pattern",
                partial(ablate_pttn_hook, ablate_pttn=ablated[l]),
            ),
        ],
    )
    perf_hit = original_performance - val(ablate_pttn_logits.argmax(-1), data)
    print(f"Ablated performance hit: {perf_hit:.2g}")

    # Permanently add the hook
    model.add_hook(
        "blocks.0.attn.hook_pattern",
        partial(ablate_pttn_hook, ablate_pttn=ablated[0]),
    )
    model.add_hook(
        "blocks.1.attn.hook_pattern",
        partial(ablate_pttn_hook, ablate_pttn=ablated[1]),
    )
    logits, cache = model.run_with_cache(inputs, return_type="logits")
    outputs = logits.argmax(-1)


# %% [markdown]
# This gives us a simplified graph, where we see only how information flows from the input digits through the separator token into the output positions. Possibly surprising here is that the attention that the output positions pay to the separator token cannot be ablated.

# %%

if __name__ == "__main__":
    plot_attention_flow(
        model, inputs, device, position_names=["d1", "d2", "SEP", "o1", "o2"]
    )

# %% [markdown]
# We start by looking at the attention pattern for layer 0. In layer 0, o1 and o2 attend strictly to <SEP>, and <SEP> shares it attention between d1 and d2.

# %%

if __name__ == "__main__":
    pattern_layer0 = cache["pattern", 0].mean(0).squeeze().cpu().numpy()
    plt.figure(figsize=(6, 5))
    plt.imshow(pattern_layer0, cmap="viridis", aspect="auto")
    plt.colorbar(label="Attention Weight")
    plt.xlabel("Key Position")
    plt.ylabel("Query Position")
    plt.title("Attention Pattern Heatmap (Layer 0)")
    plt.tight_layout()
    plt.show()

# %% [markdown]
# Earlier research found that RNNs use a fixed attention pattern where the embeddings are projected into an "onion ring" pattern. However, in our model, the attention pattern to each of the input digits is roughly normally distributed, for each of the positions (though always summing to 1).
# %%

if __name__ == "__main__":
    # Extract attention weights for SEP (position 2) attending to d1 (0) and d2 (1) in layer 0
    attn_sep_d1 = cache["pattern", 0].squeeze()[:, 2, 0].cpu().numpy()
    attn_sep_d2 = cache["pattern", 0].squeeze()[:, 2, 1].cpu().numpy()

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.hist(attn_sep_d1, bins=30, color="skyblue", edgecolor="black")
    plt.xlabel("Attention to d1 (pos 0)")
    plt.ylabel("Count")
    plt.title("Histogram: SEP → d1 (Layer 0)")

    plt.subplot(1, 2, 2)
    plt.hist(attn_sep_d2, bins=30, color="salmon", edgecolor="black")
    plt.xlabel("Attention to d2 (pos 1)")
    plt.ylabel("Count")
    plt.title("Histogram: SEP → d2 (Layer 0)")

    plt.tight_layout()
    plt.show()

# %% [markdown]
# Switching to mean attention destroys performance.
# %%

if __name__ == "__main__":
    mean_pttn = cache["pattern", 0].mean(0).squeeze()

    def mean_pttn_hook(pttn, hook=None):
        pttn[:, :] = mean_pttn
        return pttn

    logits = model.run_with_hooks(
        inputs, fwd_hooks=[("blocks.0.attn.hook_pattern", mean_pttn_hook)]
    )
    mean_pttn_performance = val(logits.argmax(-1), data)
    perf_hit = original_performance - mean_pttn_performance
    print(f"Ablated performance hit: {perf_hit:.2g}")

# %% [markdown]
# There isn't a clear relationship between the digit values and the performance of the model. If d1 == d2 led to failures, we would expect to see a diagonal band of incorrect predictions in the scatter plot.
# %%

if __name__ == "__main__":
    # Only show incorrect values (where correct == 0)
    incorrect_mask = correct == 0
    plt.figure(figsize=(6, 6))
    scatter = plt.scatter(
        inputs[incorrect_mask, 0].cpu(),
        inputs[incorrect_mask, 1].cpu(),
        c="blue",
        alpha=0.5,
        label="Incorrect",
    )
    plt.xlabel("d1 value")
    plt.ylabel("d2 value")
    plt.title("Scatter plot of incorrect predictions by d1 and d2 values")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# %% [markdown]
# When plotting the attention scores though, we see that there is a band of attention value combinations that always leads to incorrect predictions. This raises the question of whether the failures occur when any similar values are used.
# %%

if __name__ == "__main__":
    # Extract the relevant attention values
    attn_vals = cache["attn_scores", 0].squeeze()[:, 2, [0, 1]].cpu().numpy()

    # Scatter plot: x = attention to d1 (pos 0), y = attention to d2 (pos 1), color by correctness
    plt.figure(figsize=(6, 6))
    scatter = plt.scatter(
        attn_vals[:, 0],
        attn_vals[:, 1],
        c=correct,
        cmap="coolwarm",
        alpha=0.5,
        label=None,
    )
    plt.xlabel("Attention to d1 (pos 0)")
    plt.ylabel("Attention to d2 (pos 1)")
    plt.title("Scatter plot of Layer 0, Query=SEP, Attention to d1 vs d2")
    plt.grid(True)
    plt.tight_layout()

    # Add legend for correct/incorrect
    legend_elements = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label="Incorrect",
            markerfacecolor=plt.cm.coolwarm(0.0),
            markersize=8,
            alpha=0.5,
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label="Correct",
            markerfacecolor=plt.cm.coolwarm(1.0),
            markersize=8,
            alpha=0.5,
        ),
    ]
    plt.legend(handles=legend_elements, title="Correctness")
    plt.show()

# %% [markdown]
# Much, but not all of the performance of the model can be recovered by fixing the attention scores to certain values.

# %%


def fixed_attention_score_hook(attn, hook=None, attn_0=None, attn_1=None):
    assert (attn_0 is not None) and (
        attn_1 is not None
    ), "attn_0 and attn_1 must be specified"

    attn[:, :, 2, 0] = attn_0
    attn[:, :, 2, 1] = attn_1
    return attn


if __name__ == "__main__":
    steps = 100
    scores = torch.linspace(0, 1, steps=steps)
    cache_file = "fixed_attention_perfs.pkl"

    if os.path.exists(cache_file):
        with open(cache_file, "rb") as f:
            perfs = pickle.load(f)
    else:
        perfs = torch.zeros((steps, steps))
        for i, j in tqdm(itertools.product(range(steps), range(steps)), total=steps**2):
            l = model.run_with_hooks(
                inputs,
                fwd_hooks=[
                    (
                        "blocks.0.attn.hook_pattern",
                        partial(
                            fixed_attention_score_hook,
                            attn_0=scores[i],
                            attn_1=scores[j],
                        ),
                    )
                ],
            )
            perfs[i, steps - j - 1] = val(l.argmax(-1), data)
        with open(cache_file, "wb") as f:
            pickle.dump(perfs, f)

    plt.figure(figsize=(6, 5))
    plt.imshow(perfs.T.cpu(), cmap="viridis", aspect="auto")
    plt.colorbar(label="Performance")
    plt.xlabel("<SEP> -> d1 attention score")
    plt.ylabel("<SEP> -> d2 attention score")
    plt.title("Model validation performance for fixed attention scores")
    plt.tight_layout()
    plt.show()

# %% [markdown]
# In all failure cases, the predicted o2 == d1.

# %%

if __name__ == "__main__":
    print((data[~correct, 0] == outputs[~correct, -1]).all().item())

# %%

# Run PCA on model.W_pos and plot the first two principal components
if __name__ == "__main__":
    W_pos = model.W_pos.detach().cpu().numpy()
    pca = PCA(n_components=2)
    W_pos_pca = pca.fit_transform(W_pos)
    plt.figure(figsize=(6, 6))
    plt.scatter(
        W_pos_pca[:, 0],
        W_pos_pca[:, 1],
        c=np.arange(W_pos.shape[0]),
        cmap="tab10",
        s=80,
    )
    for i, (x, y) in enumerate(W_pos_pca):
        plt.text(x + 0.03, y + 0.03, POSITION_NAMES[i], fontsize=10)
    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")
    plt.title("PCA of model.W_pos")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# %%



# %% [markdown]
# The PCA of Q and K shows clear separation between the two input digits (d1 and d2) arising from the positional encoding.
# %%

if __name__ == "__main__":
    # Residuals: [batch, seq, d_model] or [seq, d_model]
    resid = cache["resid_post", 1].detach().cpu().numpy()
    if resid.ndim == 2:  # no batch
        resid = resid[None, ...]
    if resid.shape[1] < 2:
        raise ValueError("Need sequence length >= 2 to compare -1 and -2.")

    # Last and second-to-last token vectors
    vecs_last = resid[:, -1, :]   # [B, d_model]
    vecs_prev = resid[:, -2, :]   # [B, d_model]

    # Positional matrix and projections
    W_pos = model.W_pos.detach().cpu().numpy()  # [n_positions, d_model]
    projs_last = vecs_last @ W_pos.T  # [B, n_positions]
    projs_prev = vecs_prev @ W_pos.T  # [B, n_positions]

    # Mean projections across data points
    mean_last = projs_last.mean(axis=0)
    mean_prev = projs_prev.mean(axis=0)

    fig, axs = plt.subplots(1, 2, figsize=(14, 4), sharey=True)

    axs[0].boxplot([projs_last[:, j] for j in range(projs_last.shape[1])], showfliers=False)
    step = max(1, projs_last.shape[1] // 20)
    axs[0].set_xticks(ticks=np.arange(1, projs_last.shape[1] + 1, step),
                      labels=np.arange(0, projs_last.shape[1], step))
    axs[0].set_xlabel("Position index")
    axs[0].set_ylabel("Dot product")
    axs[0].set_title("Distribution across data: last token (-1)")

    axs[1].boxplot([projs_prev[:, j] for j in range(projs_prev.shape[1])], showfliers=False)
    step2 = max(1, projs_prev.shape[1] // 20)
    axs[1].set_xticks(ticks=np.arange(1, projs_prev.shape[1] + 1, step2),
                      labels=np.arange(0, projs_prev.shape[1], step2))
    axs[1].set_xlabel("Position index")
    axs[1].set_title("Distribution across data: second-last token (-2)")

    fig.tight_layout()
    plt.show()

# %%

if __name__ == "__main__":
    # Residuals: [batch, seq, d_model] or [seq, d_model]
    resid = cache["resid_post", 1].detach().cpu().numpy()
    if resid.ndim == 2:  # no batch
        resid = resid[None, ...]
    if resid.shape[1] < 2:
        raise ValueError("Need sequence length >= 2 to compare -1 and -2.")

    # Last and second-to-last token vectors
    vecs_last = resid[:, -1, :]  # [B, d_model]
    vecs_prev = resid[:, -2, :]  # [B, d_model]

    # Positional matrix and projections
    W_pos = model.W_pos.detach().cpu().numpy()
    projs_last = (vecs_last @ W_pos.T)[:]
    projs_prev = (vecs_prev @ W_pos.T)[:]

    fig, ax = plt.subplots(figsize=(6, 5))
        
    ax.scatter(projs_last[:, 0], projs_last[:, 2], s=8, alpha=0.7, color="tab:blue", label="o1 (last)")
    ax.scatter(projs_prev[:, 0], projs_prev[:, 2], s=8, alpha=0.7, color="tab:orange", label="o2 (prev)")

    ax.set_title("Projection of o1 and o2 onto positional dims 2 and 3")
    ax.set_xlabel("pos dim 2")
    ax.set_ylabel("pos dim 3")
    ax.grid(True, linestyle=":")
    ax.legend(frameon=False)

    plt.tight_layout()
    plt.show()

# %% [markdown]
# The projections of the last and second-to-last token vectors onto the positional dimensions are linearly separable, revealing that this composition of positional encodings is key to separating the tokens back out.

# %%
 
# # X: points, y: labels
# X = np.vstack([projs_last, projs_prev])
# y = np.hstack([np.ones(len(projs_last)), -np.ones(len(projs_prev))])

# # hard-margin SVM via large C
# clf = SVC(kernel="linear", C=1e6).fit(X, y)

# separable = clf.score(X, y) > 0.99
# print("Linearly separable:", separable)

# if separable:
#     w = clf.coef_[0]
#     b = clf.intercept_[0]
#     margin = 1.0 / np.linalg.norm(w)
#     print("Margin:", margin)

# %% [markdown]
# This is clear from the PCA where there is a substantial gap between the two clusters on the PCA plot.

# %%

def plot_pca_two_sets(A: torch.Tensor, B: torch.Tensor, labels=("last", "prev")):
    with torch.no_grad():
        # Flatten leading dims -> [N, D]
        A2 = A.reshape(-1, A.shape[-1]).float().cpu()
        B2 = B.reshape(-1, B.shape[-1]).float().cpu()

        # Fit PCA on A∪B using SVD
        X_all = torch.cat([A2, B2], dim=0)                  # [N_total, D]
        mean = X_all.mean(0, keepdim=True)                  # [1, D]
        Xc = X_all - mean
        # Xc = U @ diag(S) @ Vh
        U, S, Vh = torch.linalg.svd(Xc, full_matrices=False)
        comps = Vh[:2, :].T                                 # [D, 2]
        var_ratio = (S**2 / (S**2).sum())[:2]               # explained variance ratio (approx)

        # Project each set with the shared components
        A_pc = (A2 - mean) @ comps                          # [NA, 2]
        B_pc = (B2 - mean) @ comps                          # [NB, 2]

    # Plot
    plt.figure(figsize=(6, 5))
    plt.scatter(A_pc[:, 0], A_pc[:, 1], s=10, label=labels[0], alpha=0.7)
    plt.scatter(B_pc[:, 0], B_pc[:, 1], s=10, label=labels[1], alpha=0.7)
    plt.xlabel(f"PC1 ({var_ratio[0].item():.1%})")
    plt.ylabel(f"PC2 ({var_ratio[1].item():.1%})")
    plt.legend(frameon=False)
    plt.title("PCA of unembeds")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    with torch.no_grad():
        last_pos_projs_unembed = torch.from_numpy(projs_last) @ model.W_pos.cpu() @ model.W_U.cpu()
        prev_pos_projs_unembed = torch.from_numpy(projs_prev) @ model.W_pos.cpu() @ model.W_U.cpu()

    plot_pca_two_sets(last_pos_projs_unembed, prev_pos_projs_unembed,
                      labels=("last", "prev"))


