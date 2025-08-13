"""
- Attention flow plots for the original attention pattern and the ablated one
  the 2L model.
"""
# %%
import itertools
import numpy as np
import torch
import torch
import numpy as np
import os
from functools import partial
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from nb_2025_08_12 import load_model, MODEL_PATH, get_data, LIST_LEN

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


# %%


def val(pred, target):
    return (pred[:, -LIST_LEN:] == target[:, -LIST_LEN:]).float().mean().item()


if os.path.exists(MODEL_PATH):
    model = load_model(MODEL_PATH, device=DEV)
    data, inputs = get_data()
    logits, cache = model.run_with_cache(inputs, return_type="logits")

    assert val(data, data) == 1.0
    assert val(inputs, data) == 0.0
    original_performance = val(logits.argmax(-1).cpu(), data)
    print(f"Original performance: {original_performance}")

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


if __name__ == "__main__":
    plot_attention_flow(
        model, inputs, device, position_names=["d1", "d2", "SEP", "o1", "o2"]
    )

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
        pttn_ablate_perf = val(pttn_ablate_logits.argmax(-1).cpu(), data)
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
    perf_hit = original_performance - val(ablate_pttn_logits.argmax(-1).cpu(), data)
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


# %%

if __name__ == "__main__":
    plot_attention_flow(
        model, inputs, device, position_names=["d1", "d2", "SEP", "o1", "o2"]
    )
