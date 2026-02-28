import os
import torch
from transformer_lens import HookedTransformer, HookedTransformerConfig, utils

from .runtime import _RUNTIME, configure_runtime  # single source of truth

__all__ = [
	"configure_runtime",
	"build_attention_mask",
	"attach_custom_mask",
	"strip_bias",
	"set_WV_identity_and_freeze",
	"set_WO_identity_and_freeze",
	"make_model",
	"accuracy",
	"save_model",
	"load_model",
	"parse_model_name",
	"parse_model_name_safe",
	"infer_model_config",
]

# ---------- mask ----------
# attention mask for [d1, d2, SEP, o1, o2] looks like this:
# -    d1    d2    SEP    o1    o2   (keys)
# d1  -inf  -inf   -inf  -inf  -inf
# d2   0    -inf   -inf  -inf  -inf
# SEP  0      0    -inf  -inf  -inf
# o1  -inf  -inf    0    -inf   -inf
# o2  -inf  -inf    0      0    -inf
# (queries)
def build_attention_mask(list_len = None, seq_len = None):
	"""Build attention masks, defaulting to configured runtime values.

	Returns a tuple of (mask_bias, mask_bias_l0), both with shape (1,1,T,T).
	"""
	if list_len is None:
		list_len = _RUNTIME.list_len
	if seq_len is None:
		seq_len = _RUNTIME.seq_len
	assert list_len is not None and seq_len is not None, "list_len and seq_len must be provided or configured via configure_runtime()"

	mask_bias = torch.triu(torch.ones(seq_len, seq_len) * float("-inf")) # upper triangular bias mask (lead_diag & above = -inf, rest = 0)
	mask_bias[0, 0] = 0. # don't want a full row of -inf! otherwise we get nan errors & training breaks
	mask_bias[list_len+1:, :list_len] = float("-inf") # stop output tokens from attending to input tokens
	mask_bias = mask_bias.unsqueeze(0).unsqueeze(0) # (1,1,T,T) broadcastable across batch and heads

	# L0: keep outputs self-only and allow SEP->digits; avoid all -inf rows
	mask_bias_l0 = mask_bias.clone()
	mask_bias_l0[..., list_len+1:, :] = float("-inf") # block all for outputs
	idx = torch.arange(list_len+1, seq_len)  # re-enable self for outputs
	mask_bias_l0[..., idx, idx] = 0.0

	return mask_bias, mask_bias_l0

def attach_custom_mask(model, list_len = None, seq_len = None):
	"""Attach masking hooks to a model, defaulting to runtime-config lengths."""
	if list_len is None:
		list_len = _RUNTIME.list_len
	if seq_len is None:
		seq_len = _RUNTIME.seq_len
	assert list_len is not None and seq_len is not None, "list_len and seq_len must be provided or configured via configure_runtime()"

	# Pre-compute masks used by hooks (define before closures for clarity and static analyzers)
	mask_bias, mask_bias_l0 = build_attention_mask(list_len, seq_len)

	def _mask(scores, hook=None):
		# scores: (batch, heads, Q, K)
		return scores + mask_bias.to(scores.device)

	def _mask_l0(scores, hook=None):
		# layer-0 special mask: o1/o2 only self; SEP can read d1/d2
		return scores + mask_bias_l0.to(scores.device)

	# Completely suppress attention for oi in L0 (safe: zero pattern rows, not -inf scores)
	def _zero_o_rows(pattern, hook=None):
		# pattern: [B, H, Q, K]
		start_o = list_len + 1  # first o_i index
		if start_o < seq_len:
			pattern = pattern.clone()
			pattern[..., start_o:seq_len, :] = 0.0
		return pattern

	# register the same mask hook on every layer
	for i, block in enumerate(model.blocks):
		if i == 0:
			block.attn.hook_attn_scores.add_perma_hook(_mask_l0, dir="fwd")
			block.attn.hook_pattern.add_perma_hook(_zero_o_rows, dir="fwd")
		else:
			block.attn.hook_attn_scores.add_perma_hook(_mask, dir="fwd")

# ---------- config helpers ----------

def strip_bias(m):
	for mod in m.modules():
		if hasattr(mod, "bias") and mod.bias is not None:
			mod.bias.requires_grad_(False)
			torch.nn.init.zeros_(mod.bias)

	# remove biases from attention layers
	attn_biases = ['b_Q', 'b_K', 'b_V', 'b_O']
	for block in m.blocks:
		for b in attn_biases:
			mod = getattr(block.attn, b, None)
			if mod is not None:
				mod.requires_grad_(False)
				torch.nn.init.zeros_(mod)

	# remove unembed bias
	b_U = getattr(m, "b_U", None)
	if b_U is None and hasattr(m, "unembed"):
		b_U = getattr(m.unembed, "b_U", None)
	if b_U is not None:
		b_U.requires_grad_(False)
		torch.nn.init.zeros_(b_U)

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
	*,
	n_layers,
	n_heads,
	d_model,
	ln = False,
	use_bias = False,
	# wv/wo: use_wv/use_wo is the newer API (True = learn the matrix);
	# freeze_wv/freeze_wo is the legacy API (True = freeze to identity).
	# use_wv/use_wo take precedence if provided.
	use_wv = None,
	use_wo = None,
	freeze_wv = None,
	freeze_wo = None,
	attn_only = True,
	seq_len = None,
	vocab = None,
	list_len = None,
	device = None,
):
	"""Construct a HookedTransformer with our custom mask.

	Args:
		seq_len: total sequence length (must match inputs)
		vocab: vocabulary size
		list_len: number of input digits (used by mask)
		device: device to place model on (defaults to DEV)
		use_wv: if True, learn W_V (default False = freeze to identity)
		use_wo: if True, learn W_O (default False = freeze to identity)
		attn_only: if True, no MLP layers (default True)
	"""
	# Resolve wv/wo: use_wv/use_wo take precedence; freeze_wv/freeze_wo are legacy aliases
	if use_wv is None:
		use_wv = not freeze_wv if freeze_wv is not None else False
	if use_wo is None:
		use_wo = not freeze_wo if freeze_wo is not None else False

	# Resolve from explicit args or runtime config
	if seq_len is None:
		seq_len = _RUNTIME.seq_len
	if vocab is None:
		vocab = _RUNTIME.vocab
	if list_len is None:
		list_len = _RUNTIME.list_len
	dev = device or _RUNTIME.device

	assert seq_len is not None and vocab is not None and list_len is not None, "seq_len, vocab, and list_len must be provided or configured via configure_runtime()"
	cfg = HookedTransformerConfig(
		n_layers=n_layers,
		n_heads=n_heads,
		d_model=d_model,
		d_head=d_model // n_heads,
		n_ctx=seq_len,
		d_vocab=vocab,
		attn_only=attn_only,
		normalization_type=("LN" if ln else None),
	)
	model = HookedTransformer(cfg).to(dev)
	if not use_wv:
		set_WV_identity_and_freeze(model, d_model)
	if not use_wo:
		set_WO_identity_and_freeze(model, d_model)
	if not use_bias:
		strip_bias(model)

	attach_custom_mask(model, list_len, seq_len)
	return model


# metrics
def accuracy(m, val_dl, list_len=None, device=None):
	if list_len is None:
		list_len = _RUNTIME.list_len
	if device is None:
		device = _RUNTIME.device
	assert list_len is not None, "list_len must be provided or configured via configure_runtime()"
	m.eval()
	hits = tots = 0
	with torch.no_grad():
		for inputs, targets in val_dl:
			logits = m(inputs.to(device))[:, list_len + 1 :]  # (batch, 2, vocab)
			preds = logits.argmax(-1)
			hits += (preds == targets[:, list_len + 1 :].to(device)).sum().item()
			tots += preds.numel()
	return hits / tots

# ----- Model saving / loading helpers ------
def save_model(model, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")

def load_model(path, **model_kwargs):
	"""Load weights into a freshly constructed model.

	Accepts any kwargs for make_model. If seq_len/list_len/vocab/device are omitted,
	the configured runtime values will be used.
	"""
	device = model_kwargs.get("device", _RUNTIME.device)
	print("Loading model from", path)
	model = make_model(**model_kwargs)
	model.load_state_dict(
		torch.load(path, map_location=device)
	)  # map weights to target device
	model.eval()
	return model

def infer_model_config(path, device=None):
	"""Infer model configuration from a checkpoint file.

	Returns a dict with keys: d_model, n_layers, n_heads, d_vocab, n_ctx, list_len,
	                          attn_only, use_ln, use_bias, use_wv, use_wo
	"""
	if device is None:
		device = _RUNTIME.device or "cpu"

	checkpoint = torch.load(path, map_location=device, weights_only=True)

	d_model = checkpoint['blocks.0.attn.W_Q'].shape[1]
	n_heads = checkpoint['blocks.0.attn.W_Q'].shape[0]
	n_layers = sum(1 for k in checkpoint.keys() if k.endswith('.attn.W_Q'))
	d_vocab = checkpoint['embed.W_E'].shape[0]
	n_ctx = checkpoint['pos_embed.W_pos'].shape[0]

	if n_ctx % 2 == 0:
		raise ValueError(f"Invalid n_ctx={n_ctx}: expected odd value (n_ctx = list_len * 2 + 1)")
	list_len = (n_ctx - 1) // 2

	attn_only = 'blocks.0.mlp.W_in' not in checkpoint
	use_ln = 'blocks.0.ln1.w' in checkpoint and bool((checkpoint['blocks.0.ln1.w'].abs().sum() > 0).item())

	# W_V frozen to identity if all values match the identity slice
	try:
		d_head = checkpoint['blocks.0.attn.W_V'].shape[-1]
		identity_slice = torch.eye(d_model, d_head)
		wv = checkpoint['blocks.0.attn.W_V'][0]
		use_wv = not torch.allclose(wv, identity_slice, atol=1e-5)
		identity_slice_o = torch.eye(d_head, d_model)
		wo = checkpoint['blocks.0.attn.W_O'][0]
		use_wo = not torch.allclose(wo, identity_slice_o, atol=1e-5)
	except Exception:
		use_wv = False
		use_wo = False

	# Detect bias: check if any b_Q is non-zero
	use_bias = 'blocks.0.attn.b_Q' in checkpoint and bool((checkpoint['blocks.0.attn.b_Q'].abs().sum() > 0).item())

	return dict(
		d_model=d_model, n_layers=n_layers, n_heads=n_heads, d_vocab=d_vocab,
		n_ctx=n_ctx, list_len=list_len, attn_only=attn_only,
		use_ln=use_ln, use_bias=use_bias, use_wv=use_wv, use_wo=use_wo,
	)


# ----- Model name parsing helper ------
def parse_model_name(name: str):
	"""Parse model naming convention to extract (n_layers, n_digits, d_model).

	Supports forms like:
	    '2layer_100dig_64d'
	    '2layer_100dig_64d_20241014-153012'

	Returns:
	    tuple (n_layers:int, n_digits:int, d_model:int)

	Raises:
	    ValueError if pattern not recognized.
	"""
	import re
	# Remove an optional trailing timestamp or run id separated by underscore
	base = name.split('.pt')[0]  # strip accidental file extension
	# Accept additional suffix segments after the first three components
	pattern = r"^(?P<layers>\d+)layer_(?P<digits>\d+)dig_(?P<dmodel>\d+)d(?:_.+)?$"
	m = re.match(pattern, base)
	if not m:
		raise ValueError(f"Model name '{name}' does not match expected pattern '<L>layer_<D>dig_<M>d[_...]' ")
	n_layer = int(m.group('layers'))
	n_digits = int(m.group('digits'))
	d_model = int(m.group('dmodel'))
	print(f"Using model config: {n_layer} layers, {n_digits} digits, {d_model} d_model")
	return n_digits, d_model, n_layer


from dataclasses import dataclass as _dataclass

@_dataclass
class _ParsedModelName:
	list_len: int
	n_layers: int
	n_digits: int
	d_model: int


def parse_model_name_safe(name: str) -> _ParsedModelName:
	"""Parse model name safely, returning a dataclass with defaults.

	Supports both legacy format ('2layer_100dig_64d') and new format ('L2_H1_D64_V100').
	Returns a _ParsedModelName with list_len defaulting to 2 if not encoded in name.
	"""
	import re
	base = name.split('.pt')[0]

	# New format: L{layers}_H{heads}_D{dmodel}_V{vocab}[_len{list_len}][_...]
	m = re.search(r'L(\d+)_H\d+_D(\d+)_V(\d+)', base)
	if m:
		n_layers = int(m.group(1))
		d_model = int(m.group(2))
		n_digits = int(m.group(3)) - 2  # vocab = digits + MASK + SEP
		len_m = re.search(r'len(\d+)', base)
		list_len = int(len_m.group(1)) if len_m else 2
		return _ParsedModelName(list_len=list_len, n_layers=n_layers, n_digits=n_digits, d_model=d_model)

	# Legacy format: {layers}layer_{digits}dig_{dmodel}d
	m = re.match(r'^(\d+)layer_(\d+)dig_(\d+)d', base)
	if m:
		n_layers = int(m.group(1))
		n_digits = int(m.group(2))
		d_model = int(m.group(3))
		return _ParsedModelName(list_len=2, n_layers=n_layers, n_digits=n_digits, d_model=d_model)

	# Fallback: return safe defaults with a warning
	print(f"[parse_model_name_safe] Could not parse '{name}', using defaults.")
	return _ParsedModelName(list_len=2, n_layers=2, n_digits=100, d_model=64)