import os
import torch
from transformer_lens import HookedTransformer, HookedTransformerConfig, utils
from typing import NamedTuple

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
	"count_params",
	"save_model",
	"load_model",
	"ModelConfig",
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
	use_wv = False,
	use_wo = False,
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
		use_wv: if False, freeze W_V to identity (default: False)
		use_wo: if False, freeze W_O to identity (default: False)
		attn_only: if True, no MLP layers (default True)
	"""
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
		act_fn="gelu" if not attn_only else None,
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
	with torch.inference_mode():
		for inputs, targets in val_dl:
			logits = m(inputs.to(device))[:, list_len + 1 :]  # (batch, 2, vocab)
			preds = logits.argmax(-1)
			hits += (preds == targets[:, list_len + 1 :].to(device)).sum().item()
			tots += preds.numel()
	return hits / tots


def count_params(model) -> tuple[int, int]:
	"""Return (total_params, trainable_params)."""
	total = sum(p.numel() for p in model.parameters())
	trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
	return total, trainable

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

	expected_n_ctx = list_len * 2 + 1
	if n_ctx != expected_n_ctx:
		raise ValueError(f"n_ctx={n_ctx} does not match expected value {expected_n_ctx} for list_len={list_len}")

	# Infer use_wv: check if W_V differs from identity across all heads
	W_V = checkpoint['blocks.0.attn.W_V']  # (n_heads, d_model, d_head)
	d_head = W_V.shape[-1]
	identity_wv = torch.eye(d_model, d_head).unsqueeze(0).expand(n_heads, -1, -1)
	use_wv = not torch.allclose(W_V, identity_wv.to(W_V.device), atol=1e-5)

	# Infer use_wo: check if W_O differs from identity across all heads
	W_O = checkpoint['blocks.0.attn.W_O']  # (n_heads, d_head, d_model)
	identity_wo = torch.eye(d_head, d_model).unsqueeze(0).expand(n_heads, -1, -1)
	use_wo = not torch.allclose(W_O, identity_wo.to(W_O.device), atol=1e-5)

	# Detect bias: check if any b_Q is non-zero
	use_bias = 'blocks.0.attn.b_Q' in checkpoint and bool((checkpoint['blocks.0.attn.b_Q'].abs().sum() > 0).item())

	config = dict(
		d_model=d_model, n_layers=n_layers, n_heads=n_heads, d_vocab=d_vocab,
		n_ctx=n_ctx, list_len=list_len, attn_only=attn_only,
		use_ln=use_ln, use_bias=use_bias, use_wv=use_wv, use_wo=use_wo,
	)
	print(f"Inferred config: {config}")
	return config


# ----- Model name parsing helper ------

class ModelConfig(NamedTuple):
	"""Parsed model configuration from model name."""
	n_digits: int
	d_model: int
	n_layers: int
	list_len: int = 2  # default for backwards compatibility


def parse_model_name(name: str) -> ModelConfig:
	"""Parse model naming convention to extract configuration.

	Supports both naming formats:
	    New: 'L2_H1_D64_V100' or 'L2_H1_D64_V100_len3-ln-bias_250120-180326'
	    Old: '2layer_100dig_64d' or '2layer_100dig_64d_20241014-153012'

	Returns:
	    ModelConfig(n_digits, d_model, n_layers, list_len)

	Raises:
	    ValueError if pattern not recognized.
	"""
	import re
	base = name.split('.pt')[0]

	def _extract_list_len(s: str) -> int:
		m = re.search(r'len(\d+)', s)
		return int(m.group(1)) if m else 2

	# New format first: L{layers}_H{heads}_D{dmodel}_V{vocab}[_flags][_timestamp]
	m = re.match(r"^L(?P<layers>\d+)_H(?P<heads>\d+)_D(?P<dmodel>\d+)_V(?P<digits>\d+)(?:_.+)?$", base)
	if m:
		config = ModelConfig(
			n_digits=int(m.group('digits')),
			d_model=int(m.group('dmodel')),
			n_layers=int(m.group('layers')),
			list_len=_extract_list_len(base),
		)
		print(f"Parsed model config (new format): {config.n_layers} layers, {config.n_digits} digits, {config.d_model} d_model, list_len={config.list_len}")
		return config

	# Old format: {layers}layer_{digits}dig_{dmodel}d[_...]
	m = re.match(r"^(?P<layers>\d+)layer_(?P<digits>\d+)dig_(?P<dmodel>\d+)d(?:_.+)?$", base)
	if m:
		config = ModelConfig(
			n_digits=int(m.group('digits')),
			d_model=int(m.group('dmodel')),
			n_layers=int(m.group('layers')),
			list_len=2,
		)
		print(f"Parsed model config (old format): {config.n_layers} layers, {config.n_digits} digits, {config.d_model} d_model, list_len={config.list_len}")
		return config

	raise ValueError(f"Model name '{name}' does not match expected patterns. "
		f"Expected: 'L<L>_H<H>_D<D>_V<V>[_...]' (new) or '<L>layer_<D>dig_<M>d[_...]' (old)")


def parse_model_name_safe(name: str, fallback: ModelConfig = ModelConfig(100, 64, 2)) -> ModelConfig:
	"""Parse model name with fallback on failure.

	Args:
		name: Model name string to parse
		fallback: Default config if parsing fails (default: 100 digits, 64 d_model, 2 layers)

	Returns:
		ModelConfig from parsed name or fallback
	"""
	try:
		return parse_model_name(name)
	except ValueError as e:
		print(f"[parse_model_name_safe] {e}. Using fallback: {fallback}")
		return fallback