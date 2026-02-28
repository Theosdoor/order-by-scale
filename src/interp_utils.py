# ---- Ablation of Specific Attention Edges ----
import torch
import numpy as np
from tqdm.auto import tqdm
from transformer_lens import utils
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from matplotlib.lines import Line2D
import matplotlib.patheffects as pe

from .model_utils import build_attention_mask


def get_valid_attention_positions(mask_bias, mask_bias_l0, seq_len, n_layers):
    """
    Get all valid (non-masked) attention positions for each layer.
    
    Args:
        mask_bias: Attention mask for layers 1+ with shape (1, 1, seq_len, seq_len)
        mask_bias_l0: Attention mask for layer 0 with shape (1, 1, seq_len, seq_len)
        seq_len: Sequence length
        n_layers: Number of layers in model
        
    Returns:
        Dict mapping layer_idx -> list of (query, key) tuples that are NOT masked
    """
    valid_positions = {}
    
    for layer in range(n_layers):
        mask = mask_bias_l0 if layer == 0 else mask_bias
        # mask is (1, 1, Q, K), extract the (Q, K) portion
        mask_2d = mask.squeeze(0).squeeze(0)  # (seq_len, seq_len)
        
        # Valid positions are where mask is NOT -inf (i.e., mask == 0)
        valid = (mask_2d == 0).nonzero(as_tuple=False)  # (N, 2) tensor of (q, k) pairs
        valid_positions[layer] = [(int(q), int(k)) for q, k in valid.tolist()]
    
    return valid_positions


def _make_pattern_hook(mask_2d, head_index=None, set_to=0.0, renorm=False, eps=1e-12):
    """
    Returns a forward hook that ablates specified attention positions.
    
    Args:
        mask_2d: Bool tensor [Q, K]
        head_index: int to affect a single head, or None to affect all heads
        set_to: value to write into masked entries (usually 0.0)
        renorm: whether to renormalize rows after masking
        eps: epsilon for numerical stability when renormalizing
    """
    mask_2d = mask_2d.detach()

    def hook(pattern, hook):
        # pattern: [batch, n_heads, Q, K]
        B, H, Q, K = pattern.shape
        m4_all = mask_2d.to(pattern.device).view(1, 1, Q, K)

        if head_index is None:
            pattern = torch.where(m4_all, torch.as_tensor(set_to, device=pattern.device), pattern)
        else:
            m3 = m4_all.squeeze(1)
            ph = pattern[:, head_index]
            ph = torch.where(m3, torch.as_tensor(set_to, device=pattern.device), ph)
            pattern[:, head_index] = ph

        if renorm:
            rows_to_fix = mask_2d.any(dim=-1)
            if rows_to_fix.any():
                rows_idx = rows_to_fix.nonzero(as_tuple=False).squeeze(-1)
                heads = range(H) if head_index is None else [head_index]
                for h in heads:
                    p = pattern[:, h, rows_idx, :]
                    s = p.sum(dim=-1, keepdim=True).clamp_min(eps)
                    pattern[:, h, rows_idx, :] = p / s

        return pattern

    return hook


def _build_qk_mask_from_positions(positions, seq_len):
    """Create a boolean mask where True = ablate this (q, k) position."""
    return build_qk_mask(positions=positions, seq_len=seq_len)


def _run_ablation_and_get_accuracy(model, inputs, targets, ablation_dict, seq_len, list_len, 
                                    vocab_size, head_index=None, renorm=False):
    """
    Run model with specified ablations and return accuracy.
    
    Args:
        model: The transformer model
        inputs: Input tensor [B, seq_len]
        targets: Target tensor [B, seq_len]
        ablation_dict: Dict mapping layer_idx -> list of (q, k) positions to ablate
        seq_len: Sequence length
        list_len: Number of input digits
        vocab_size: Vocabulary size
        head_index: Head to ablate (None = all heads)
        renorm: Whether to renormalize attention rows after ablation
        
    Returns:
        Accuracy (float)
    """
    fwd_hooks = []
    for layer_idx, positions in ablation_dict.items():
        if not positions:
            continue
        mask = _build_qk_mask_from_positions(positions, seq_len)
        hook_name = utils.get_act_name("pattern", layer_idx)
        fwd_hooks.append((hook_name, _make_pattern_hook(mask, head_index=head_index, 
                                                        set_to=0.0, renorm=renorm)))
    
    with torch.no_grad():
        if fwd_hooks:
            logits = model.run_with_hooks(inputs, return_type="logits", fwd_hooks=fwd_hooks)
        else:
            logits = model(inputs)
    
    output_logits = logits[:, list_len + 1:]
    output_targets = targets[:, list_len + 1:]
    predictions = output_logits.argmax(dim=-1)
    accuracy = (predictions == output_targets).float().mean().item()
    
    return accuracy


def systematic_attention_ablation(
    model,
    inputs,
    targets,
    mask_bias,
    mask_bias_l0,
    seq_len,
    list_len,
    vocab_size,
    accuracy_tolerance=0.001,
    head_index=None,
    renorm=True,
    verbose=True,
):
    """
    Systematically test ablating each valid attention position and classify them as
    critical (cannot ablate) or non-critical (can ablate without hurting accuracy).
    
    This function:
    1. Tests each valid (layer, query, key) position individually
    2. Identifies positions that, when ablated, reduce accuracy beyond tolerance
    3. Verifies that all non-critical positions can be ablated simultaneously
    
    Args:
        model: HookedTransformer model
        inputs: Input tensor [B, seq_len]
        targets: Target tensor [B, seq_len]
        mask_bias: Attention mask for layers 1+ (1, 1, seq_len, seq_len)
        mask_bias_l0: Attention mask for layer 0 (1, 1, seq_len, seq_len)
        seq_len: Sequence length
        list_len: Number of input digits (LIST_LEN)
        vocab_size: Vocabulary size (VOCAB)
        accuracy_tolerance: Maximum accuracy drop allowed (default 0.001 = 0.1%)
        head_index: Which head to ablate (None = all heads)
        renorm: Whether to renormalize attention rows after ablation
        verbose: Whether to print progress
        
    Returns:
        dict with keys:
            'critical': List of (layer, q, k) positions that CANNOT be ablated
            'non_critical': List of (layer, q, k) positions that CAN be ablated
            'original_accuracy': Baseline accuracy
            'ablated_accuracy': Accuracy when ablating all non-critical positions
            'individual_results': Dict of (layer, q, k) -> {'accuracy': float, 'delta': float}
    """
    n_layers = model.cfg.n_layers
    
    # Get valid positions for each layer
    valid_positions = get_valid_attention_positions(mask_bias, mask_bias_l0, seq_len, n_layers)
    
    # Calculate original accuracy
    original_acc = _run_ablation_and_get_accuracy(
        model, inputs, targets, {}, seq_len, list_len, vocab_size, head_index, renorm
    )
    
    if verbose:
        print(f"Original accuracy: {original_acc:.4f}")
        total_positions = sum(len(v) for v in valid_positions.values())
        print(f"Testing {total_positions} valid attention positions across {n_layers} layers...")
    
    # Test each position individually
    individual_results = {}
    critical = []
    non_critical = []
    
    all_positions = [(layer, q, k) for layer, positions in valid_positions.items() 
                     for q, k in positions]
    
    for layer, q, k in tqdm(all_positions, desc="Testing positions", disable=not verbose):
        ablation_dict = {layer: [(q, k)]}
        acc = _run_ablation_and_get_accuracy(
            model, inputs, targets, ablation_dict, seq_len, list_len, vocab_size, head_index, renorm
        )
        delta = acc - original_acc
        individual_results[(layer, q, k)] = {'accuracy': acc, 'delta': delta}
        
        if delta < -accuracy_tolerance:
            critical.append((layer, q, k))
        else:
            non_critical.append((layer, q, k))
    
    if verbose:
        print(f"\nResults: {len(critical)} critical, {len(non_critical)} non-critical positions")
    
    # Verify: ablate all non-critical positions simultaneously
    non_critical_by_layer = {}
    for layer, q, k in non_critical:
        if layer not in non_critical_by_layer:
            non_critical_by_layer[layer] = []
        non_critical_by_layer[layer].append((q, k))
    
    combined_acc = _run_ablation_and_get_accuracy(
        model, inputs, targets, non_critical_by_layer, seq_len, list_len, vocab_size, head_index, renorm
    )
    
    if verbose:
        print(f"Accuracy when ablating all {len(non_critical)} non-critical positions: {combined_acc:.4f}")
        combined_delta = combined_acc - original_acc
        if combined_delta < -accuracy_tolerance:
            print(f"WARNING: Combined ablation reduces accuracy by {-combined_delta:.4f}!")
            print("Some positions may have interaction effects.")
    
    return {
        'critical': critical,
        'non_critical': non_critical,
        'original_accuracy': original_acc,
        'ablated_accuracy': combined_acc,
        'individual_results': individual_results,
        'non_critical_by_layer': non_critical_by_layer,
    }


def format_ablation_results(results, position_names=None):
    """
    Format ablation results for display.
    
    Args:
        results: Output from systematic_attention_ablation
        position_names: Optional list of position names (e.g., ['d1', 'd2', 'SEP', 'o1', 'o2'])
    
    Returns:
        Formatted string report
    """
    lines = []
    lines.append("=" * 60)
    lines.append("ATTENTION ABLATION ANALYSIS")
    lines.append("=" * 60)
    lines.append(f"Original accuracy: {results['original_accuracy']:.4f}")
    lines.append(f"Accuracy with all non-critical ablated: {results['ablated_accuracy']:.4f}")
    lines.append("")
    
    def pos_label(q, k):
        if position_names:
            return f"{position_names[q]}<-{position_names[k]}"
        return f"q{q}<-k{k}"
    
    # Group by layer
    critical_by_layer = {}
    for layer, q, k in results['critical']:
        if layer not in critical_by_layer:
            critical_by_layer[layer] = []
        critical_by_layer[layer].append((q, k))
    
    lines.append("CRITICAL POSITIONS (cannot ablate):")
    lines.append("-" * 40)
    for layer in sorted(critical_by_layer.keys()):
        positions = critical_by_layer[layer]
        pos_strs = [pos_label(q, k) for q, k in positions]
        delta_strs = [f"{results['individual_results'][(layer, q, k)]['delta']*100:+.1f}%" 
                      for q, k in positions]
        lines.append(f"  Layer {layer}: {', '.join(f'{p} ({d})' for p, d in zip(pos_strs, delta_strs))}")
    
    lines.append("")
    lines.append("NON-CRITICAL POSITIONS (can ablate):")
    lines.append("-" * 40)
    for layer in sorted(results['non_critical_by_layer'].keys()):
        positions = results['non_critical_by_layer'][layer]
        pos_strs = [pos_label(q, k) for q, k in positions]
        lines.append(f"  Layer {layer}: {', '.join(pos_strs)}")
    
    return "\n".join(lines)


def format_ablation_as_matrices(results, model, example_input, list_len, position_names=None):
    """
    Format ablation results as minimal attention pattern matrices per layer.
    Shows actual attention values for critical edges, 0 for masked/non-critical edges.
    
    Args:
        results: Output from systematic_attention_ablation or find_critical_attention_edges
        model: HookedTransformer model
        example_input: Single input sequence tensor [seq_len] or [1, seq_len]
        list_len: Number of input digits
        position_names: Optional list of position names (e.g., ['d1', 'd2', 'SEP', 'o1', 'o2'])
    
    Returns:
        Formatted string with matrices for each layer
    """
    seq_len = list_len * 2 + 1
    n_layers = model.cfg.n_layers
    
    # Handle input shape
    if example_input.dim() == 1:
        example_input = example_input.unsqueeze(0)
    device = next(model.parameters()).device
    example_input = example_input.to(device)
    
    # Run model and cache attention patterns
    with torch.no_grad():
        _, cache = model.run_with_cache(example_input, return_type="logits")
    
    # Collect attention patterns [L, H, Q, K]
    att = torch.stack(
        [cache[f"blocks.{layer}.attn.hook_pattern"] for layer in range(n_layers)],
        dim=0,
    ).cpu().numpy().squeeze()  # [L, Q, K] if single head
    
    # Handle multi-head case
    if att.ndim == 4:
        att = att.mean(axis=1)  # Average across heads: [L, Q, K]
    
    # Build set of critical edges
    critical_edges = set(results['critical'])
    
    # Generate position labels
    if position_names is None:
        position_names = [f"d{i+1}" for i in range(list_len)] + ["SEP"] + [f"o{i+1}" for i in range(list_len)]
    
    # Build matrices
    lines = []
    lines.append("=" * 60)
    lines.append("MINIMAL ATTENTION PATTERNS (Critical Edges Only)")
    lines.append("=" * 60)
    lines.append(f"Original accuracy: {results['original_accuracy']:.4f}")
    lines.append(f"Accuracy with non-critical ablated: {results['ablated_accuracy']:.4f}")
    lines.append("")
    
    for layer in range(n_layers):
        lines.append(f"Layer {layer}:")
        lines.append("-" * 60)
        
        # Create minimal attention matrix
        minimal_att = np.zeros((seq_len, seq_len))
        for q in range(seq_len):
            for k in range(seq_len):
                if (layer, q, k) in critical_edges:
                    minimal_att[q, k] = att[layer, q, k]
        
        # Format as table with row/column labels
        # Column header
        col_width = 7
        header = "      " + "".join(f"{name:>{col_width}s}" for name in position_names)
        lines.append(header)
        
        # Each row with label
        for q in range(seq_len):
            row_vals = "".join(f"{minimal_att[q, k]:>{col_width}.3f}" for k in range(seq_len))
            lines.append(f"{position_names[q]:>5s} {row_vals}")
        
        # Show critical edges for this layer
        critical_this_layer = [(q, k) for (l, q, k) in critical_edges if l == layer]
        if critical_this_layer:
            edge_strs = []
            for q, k in critical_this_layer:
                delta = results['individual_results'][(layer, q, k)]['delta']
                edge_strs.append(f"{position_names[q]}←{position_names[k]} ({delta*100:+.1f}%)")
            lines.append("")
            lines.append(f"  Critical edges ({len(critical_this_layer)}): {', '.join(edge_strs)}")
        
        lines.append("")
    
    return "\n".join(lines)


def find_critical_attention_edges(
    model,
    inputs,
    targets,
    list_len,
    vocab_size=None,
    accuracy_tolerance=0.001,
    head_index=None,
    renorm=True,
    verbose=True,
):
    """
    Find critical attention edges that cannot be ablated without reducing accuracy.
    
    This is a simplified wrapper around systematic_attention_ablation that 
    automatically builds attention masks and derives seq_len from list_len.
    
    Args:
        model: HookedTransformer model
        inputs: Input tensor [B, seq_len]
        targets: Target tensor [B, seq_len]
        list_len: Number of input digits (e.g., 2 for [d1, d2, SEP, o1, o2])
        vocab_size: Vocabulary size (inferred from model if None)
        accuracy_tolerance: Maximum accuracy drop allowed (default 0.001 = 0.1%)
        head_index: Which head to ablate (None = all heads)
        renorm: Whether to renormalize attention rows after ablation
        verbose: Whether to print progress
        
    Returns:
        dict with keys:
            'critical': List of (layer, q, k) positions that CANNOT be ablated
            'non_critical': List of (layer, q, k) positions that CAN be ablated
            'original_accuracy': Baseline accuracy
            'ablated_accuracy': Accuracy when ablating all non-critical positions
            'individual_results': Dict of (layer, q, k) -> {'accuracy': float, 'delta': float}
            'non_critical_by_layer': Dict of layer -> list of (q, k) positions
    """
    seq_len = list_len * 2 + 1  # [d1, ..., dn, SEP, o1, ..., on]
    if vocab_size is None:
        vocab_size = model.cfg.d_vocab
    
    mask_bias, mask_bias_l0 = build_attention_mask(list_len=list_len, seq_len=seq_len)
    
    return systematic_attention_ablation(
        model=model,
        inputs=inputs,
        targets=targets,
        mask_bias=mask_bias,
        mask_bias_l0=mask_bias_l0,
        seq_len=seq_len,
        list_len=list_len,
        vocab_size=vocab_size,
        accuracy_tolerance=accuracy_tolerance,
        head_index=head_index,
        renorm=renorm,
        verbose=verbose,
    )


# Legacy variables for backward compatibility
renorm_rows = False  # whether to renormalize rows after ablation
ablate_in_l0 = [
    (4, 3),
    (0, 0),
    (1, 0)
]
ablate_in_l1 = [
    (0, 0),
    (1, 0),
    (2, 0),
    (2, 1),
    (4, 3)
]
ablate_in_l2 = [(0, 0), (1, 0), (2, 0), (2, 1), (3, 0), (4, 0), (4, 1), (4, 2), (4, 3)]


# Try ablating multiple layer attention patterns at same time
def build_qk_mask(positions=None, queries=None, keys=None, seq_len=5):
    """
    Create a boolean mask of shape (seq_len, seq_len) where True means "ablate this (q,k)".
    You can pass:
      - positions: list of (q, k) tuples
      - or queries: iterable of q, and keys: iterable of k (outer-product mask)
    """
    mask = torch.zeros(seq_len, seq_len, dtype=torch.bool)
    if positions is not None:
        for q, k in positions:
            if 0 <= q < seq_len and 0 <= k < seq_len:
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
    Returns a fwd hook for the 'pattern' activation that ablates specified positions.
    
    This is a wrapper around _make_pattern_hook with renorm=True by default for
    backward compatibility with legacy code.
    
    Args:
        mask_2d: Bool tensor [Q, K]
        head_index: int to affect a single head, or None to affect all heads
        set_to: value to write into masked entries (usually 0.0)
        renorm: whether to renormalize rows after masking (default True)
        eps: epsilon for numerical stability when renormalizing
    """
    return _make_pattern_hook(mask_2d, head_index=head_index, set_to=set_to, renorm=renorm, eps=eps)


def gen_attn_flow(
    model,
    example_input,
    list_len,
    ablation_results=None,
    inputs_for_ablation=None,
    targets_for_ablation=None,
    accuracy_tolerance=0.001,
    show_delta_labels=True,
    attention_threshold=0.04,
    figsize=(6.8, 4.2),
    dpi=300,
    show_plot=True,
    return_fig=False,
):
    """
    Generate an attention flow diagram showing critical attention edges.
    
    Creates a publication-ready figure showing:
    - Nodes for each token position at each layer
    - Residual stream connections (dotted lines)
    - Attention edges (arrows) with optional delta-accuracy labels
    - Only shows edges that CANNOT be ablated (critical edges) if ablation_results provided
    
    Args:
        model: HookedTransformer model
        example_input: Single input sequence tensor [seq_len] or [1, seq_len]
        list_len: Number of input digits (seq_len = list_len * 2 + 1)
        ablation_results: Pre-computed results from find_critical_attention_edges.
                          If None and inputs_for_ablation/targets_for_ablation provided,
                          will compute automatically. If None and no ablation inputs,
                          shows all attention edges.
        inputs_for_ablation: Inputs for ablation analysis (if ablation_results is None)
        targets_for_ablation: Targets for ablation analysis (if ablation_results is None)
        accuracy_tolerance: Tolerance for classifying edges as critical
        show_delta_labels: Whether to show accuracy delta labels on edges
        attention_threshold: Minimum attention weight to draw an edge
        figsize: Figure size tuple
        dpi: Figure DPI
        show_plot: Whether to call plt.show()
        return_fig: Whether to return (fig, ax) tuple
        
    Returns:
        If return_fig=True: (fig, ax) tuple
        Otherwise: None
    """
    seq_len = list_len * 2 + 1
    n_layers = model.cfg.n_layers
    
    # Handle input shape
    if example_input.dim() == 1:
        example_input = example_input.unsqueeze(0)
    device = next(model.parameters()).device
    example_input = example_input.to(device)
    
    # Run ablation analysis if needed
    if ablation_results is None and inputs_for_ablation is not None:
        ablation_results = find_critical_attention_edges(
            model=model,
            inputs=inputs_for_ablation,
            targets=targets_for_ablation,
            list_len=list_len,
            accuracy_tolerance=accuracy_tolerance,
            verbose=True,
        )
    
    # Build set of critical edges for fast lookup
    critical_edges = set()
    individual_results = {}
    if ablation_results is not None:
        critical_edges = set(ablation_results['critical'])
        individual_results = ablation_results.get('individual_results', {})
    
    # Run model and cache activations
    with torch.no_grad():
        _, cache = model.run_with_cache(example_input, return_type="logits")
    
    # Collect attention patterns [L, H, Q, K] -> take first head, squeeze batch
    att = torch.stack(
        [cache[f"blocks.{layer}.attn.hook_pattern"] for layer in range(n_layers)],
        dim=0,
    ).cpu().numpy().squeeze()  # [L, Q, K] if single head, [L, H, Q, K] otherwise
    
    # Handle multi-head case: average across heads for visualization
    if att.ndim == 4:
        att = att.mean(axis=1)  # [L, Q, K]
    
    # Collect residual stream for logit lens
    resid_keys = ["hook_embed"] + [f"blocks.{l}.hook_resid_post" for l in range(n_layers)]
    resid_values = torch.stack([cache[k] for k in resid_keys], dim=0)
    
    W_U = getattr(model, "W_U", model.unembed.W_U)
    position_tokens = (resid_values @ W_U).squeeze(1).argmax(-1).cpu()  # [L+1, seq]
    
    # Layout
    L = n_layers
    N = seq_len
    x_positions = np.arange(L + 1)
    y_positions = np.arange(N)[::-1]
    
    # Styling
    plt.rcParams.update({
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    
    # Position names and colors
    position_names = [f"d{i+1}" for i in range(list_len)] + ["SEP"] + [f"o{i+1}" for i in range(list_len)]
    roles = ["input"] * list_len + ["sep"] + ["output"] * list_len
    role_colors = {"input": "#4C78A8", "sep": "#F58518", "output": "#54A24B"}
    node_colors = [role_colors[r] for r in roles]
    
    # Draw nodes
    for lx in range(L + 1):
        xs = np.full(N, lx)
        ax.scatter(xs, y_positions, s=180, c=node_colors,
                   edgecolor="black", linewidth=0.6, zorder=4)
        for i, y in enumerate(y_positions):
            if lx == 0:
                ax.text(lx - 0.14, y, position_names[i], va="center", ha="right",
                        fontsize=9, color="#334155", zorder=5)
            ax.text(lx + 0.14, y, str(position_tokens[lx, i].item()),
                    va="center", ha="left", fontsize=9, fontweight="bold",
                    color="black", zorder=5)
    
    # Draw residual stream (dotted lines)
    for lx in range(L):
        for y in y_positions:
            arrow = FancyArrowPatch((lx, y), (lx + 1, y),
                                    arrowstyle="-", mutation_scale=8,
                                    lw=1.0, linestyle=(0, (2, 2)), color="#94A3B8",
                                    alpha=0.7, zorder=1, clip_on=False)
            ax.add_patch(arrow)
    
    # Helper functions
    arrow_color = "#DC2626"
    arrow_alpha = 0.35
    
    def edge_style(w):
        lw = 0.6 + 2.0 * np.sqrt(float(w))
        return lw, arrow_alpha
    
    def angle_in_display(ax, x0, y0, x1, y1):
        X0, Y0 = ax.transData.transform((x0, y0))
        X1, Y1 = ax.transData.transform((x1, y1))
        return np.degrees(np.arctan2(Y1 - Y0, X1 - X0))
    
    def format_delta_pp(val):
        if val is None:
            return "—"
        sign = "+" if val >= 0 else "−"
        return f"{sign}{abs(val)*100:.1f}%"
    
    # Draw attention edges (only critical ones if ablation_results provided)
    for l in range(L):
        for q in range(N):
            for k in range(N):
                w = att[l, q, k]
                if w <= attention_threshold:
                    continue
                
                # Skip non-critical edges if we have ablation results
                if ablation_results is not None and (l, q, k) not in critical_edges:
                    continue
                
                x0, y0 = l, y_positions[k]
                x1, y1 = l + 1, y_positions[q]
                
                lw, alpha = edge_style(w)
                
                arrow = FancyArrowPatch(
                    (x0, y0), (x1, y1),
                    connectionstyle="arc3,rad=0",
                    arrowstyle="->", mutation_scale=8,
                    lw=lw, color=arrow_color, alpha=alpha,
                    zorder=2, shrinkA=8, shrinkB=8,
                    joinstyle="round", capstyle="round",
                    clip_on=False,
                )
                ax.add_patch(arrow)
                
                # Add delta label if available
                if show_delta_labels and (l, q, k) in individual_results:
                    delta_val = individual_results[(l, q, k)]['delta']
                    label_text = format_delta_pp(delta_val)
                    
                    angle_deg = angle_in_display(ax, x0, y0, x1, y1) - 8
                    mid_x = (x0 + x1) / 2.0
                    mid_y = (y0 + y1) / 2.0
                    
                    ann = ax.annotate(
                        label_text,
                        xy=(mid_x, mid_y),
                        ha="center",
                        va="center",
                        fontsize=7,
                        color="#111827",
                        rotation=angle_deg,
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
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    
    # Legend
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
    
    if show_plot:
        plt.show()
    
    if return_fig:
        return fig, ax


def plot_sep_attention_vs_accuracy(
    model,
    val_inputs,
    val_targets,
    list_len,
    layer=0,
    sep_position=None,
    d1_position=0,
    d2_position=1,
    position_pairs=None,
    figsize=(6.8, 6.2),
    dpi=300,
    layout='row',
    show_plot=True,
    return_fig=False,
):
    """
    Create scatter plot(s) of SEP attention scores colored by correctness.
    
    Can create either a single plot (default) or multiple subplots arranged
    in a row or column when position_pairs is provided.
    
    Args:
        model: HookedTransformer model
        val_inputs: Validation input tensor [B, seq_len]
        val_targets: Validation target tensor [B, seq_len]
        list_len: Number of input digits
        layer: Which layer to extract attention scores from (default 0)
        sep_position: Position of SEP token (default: list_len)
        d1_position: Position of first digit (default: 0, ignored if position_pairs provided)
        d2_position: Position of second digit (default: 1, ignored if position_pairs provided)
        position_pairs: List of (d1_pos, d2_pos, title) tuples to create multiple plots
        figsize: Figure size tuple
        dpi: Figure DPI
        layout: 'row' for horizontal layout, 'column' for vertical (default: 'row')
        show_plot: Whether to call plt.show()
        return_fig: Whether to return (fig, axes, stats) tuple
        
    Returns:
        If return_fig=True: (fig, axes, stats) tuple where stats contains accuracy info
        Otherwise: None
    """
    if sep_position is None:
        sep_position = list_len
    
    device = next(model.parameters()).device
    val_inputs = val_inputs.to(device)
    val_targets = val_targets.to(device)
    
    # Run model on all validation data
    with torch.no_grad():
        logits, cache = model.run_with_cache(val_inputs, return_type="logits")
    
    # Correctness per sample (both outputs correct)
    predictions = logits.argmax(dim=-1)[:, -list_len:]
    targets = val_targets[:, -list_len:]
    all_correct = (predictions == targets).all(dim=-1).cpu().numpy()
    
    # Extract attention SCORES: specified layer, q=SEP
    scores = cache["attn_scores", layer][:, 0]  # [B, Q, K] (head=0)
    
    # Use position_pairs if provided, otherwise create single plot
    if position_pairs is None:
        position_pairs = [(d1_position, d2_position, None)]
    
    n_plots = len(position_pairs)
    if n_plots == 1:
        fig, axes = plt.subplots(1, 1, figsize=figsize, dpi=dpi)
        axes = [axes]
    else:
        if layout == 'column':
            fig, axes = plt.subplots(n_plots, 1, figsize=figsize, dpi=dpi)
        else:  # 'row'
            fig, axes = plt.subplots(1, n_plots, figsize=figsize, dpi=dpi)
    
    ok = all_correct.astype(bool)
    c_ok, c_bad = "#1f77b4", "#DC2626"  # blue, red
    ms, a_ok, a_bad = 24, 0.35, 0.55
    
    # Plot each pair
    for idx, (d1_pos, d2_pos, title) in enumerate(position_pairs):
        ax = axes[idx]
        
        attn_vals = scores[:, sep_position, [d1_pos, d2_pos]].detach().cpu().numpy()  # [B, 2]
        x, y = attn_vals[:, 0], attn_vals[:, 1]
        x_ok, y_ok = x[ok], y[ok]
        x_bad, y_bad = x[~ok], y[~ok]
        
        # Axes limits (robust to outliers)
        x_lo, x_hi = np.percentile(x, [1, 99])
        y_lo, y_hi = np.percentile(y, [1, 99])
        pad_x = 0.06 * max(1e-6, x_hi - x_lo)
        pad_y = 0.06 * max(1e-6, y_hi - y_lo)
        
        # Scatter
        if len(x_ok):
            ax.scatter(x_ok, y_ok, s=ms, c=c_ok, alpha=a_ok, edgecolors="none", label="Correct")
        if len(x_bad):
            ax.scatter(x_bad, y_bad, s=ms, c=c_bad, alpha=a_bad, edgecolors="none", label="Incorrect")
        
        # Reference lines at 0
        ax.axvline(0, color="#94A3B8", lw=0.8, ls="--", alpha=0.8)
        ax.axhline(0, color="#94A3B8", lw=0.8, ls="--", alpha=0.8)
        
        # Labels, limits, legend
        ax.set_xlabel(f"SEP → d{d1_pos+1} attention score")
        ax.set_ylabel(f"SEP → d{d2_pos+1} attention score")
        ax.set_xlim(x_lo - pad_x, x_hi + pad_x)
        ax.set_ylim(y_lo - pad_y, y_hi + pad_y)
        ax.grid(True, linestyle=":", alpha=0.8)
        if title:
            ax.set_title(title)
        if idx == 0:
            ax.legend(frameon=True, loc="best")
    
    plt.tight_layout()
    
    if show_plot:
        plt.show()
        # Print statistics
        acc = all_correct.mean()
        print(f"Total samples: {len(all_correct)}")
        print(f"Correct predictions: {all_correct.sum()}")
        print(f"Accuracy: {acc:.3f}")
    
    if return_fig:
        stats = {
            'accuracy': all_correct.mean(),
            'total': len(all_correct),
            'correct': all_correct.sum(),
            'incorrect': (~all_correct).sum(),
        }
        return fig, axes if n_plots > 1 else axes[0], stats
