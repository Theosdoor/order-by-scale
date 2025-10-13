import os
import torch
from dataclasses import dataclass
from transformer_lens import HookedTransformer, HookedTransformerConfig, utils

DEV = (
    "cuda"
    if torch.cuda.is_available()
    else ("mps" if torch.backends.mps.is_available() else "cpu")
)
torch.manual_seed(0)

# ---------- runtime config (optional) ----------
@dataclass
class _RuntimeConfig:
	list_len: int | None = None
	seq_len: int | None = None
	vocab: int | None = None
	device: str | torch.device = DEV

_RUNTIME = _RuntimeConfig()

def configure_runtime(*, list_len: int, seq_len: int, vocab: int, device: str | torch.device = DEV):
	"""Set module-level runtime configuration so callers can omit repeating constants.

	All arguments are required except device (defaults to detected DEV).
	"""
	_RUNTIME.list_len = list_len
	_RUNTIME.seq_len = seq_len
	_RUNTIME.vocab = vocab
	_RUNTIME.device = device

# ---------- mask ----------
# attention mask for [d1, d2, SEP, o1, o2] looks like this:
# -    d1    d2    SEP    o1    o2   (keys)
# d1  -inf  -inf   -inf  -inf  -inf
# d2   0    -inf   -inf  -inf  -inf
# SEP  0      0    -inf  -inf  -inf
# o1  -inf  -inf    0    -inf   -inf
# o2  -inf  -inf    0      0    -inf
# (queries)
def build_attention_mask(LIST_LEN, SEQ_LEN):
	mask_bias = torch.triu(torch.ones(SEQ_LEN, SEQ_LEN) * float("-inf")) # upper triangular bias mask (lead_diag & above = -inf, rest = 0)
	mask_bias[0, 0] = 0. # don't want a full row of -inf! otherwise we get nan erros & training breaks
	mask_bias[LIST_LEN+1:, :LIST_LEN] = float("-inf") # stop output tokens from attending to input tokens
	mask_bias = mask_bias.unsqueeze(0).unsqueeze(0) # (1,1,T,T) broadcastable across batch and heads

	# L0: keep outputs self-only and allow SEP->digits; avoid all -inf rows
	mask_bias_l0 = mask_bias.clone()
	mask_bias_l0[..., LIST_LEN+1:, :] = float("-inf") # block all for outputs
	idx = torch.arange(LIST_LEN+1, SEQ_LEN)  # re-enable self for outputs
	mask_bias_l0[..., idx, idx] = 0.0

	return mask_bias, mask_bias_l0

def attach_custom_mask(model, LIST_LEN, SEQ_LEN):
	# Pre-compute masks used by hooks (define before closures for clarity and static analyzers)
	mask_bias, mask_bias_l0 = build_attention_mask(LIST_LEN, SEQ_LEN)

	def _mask(scores, hook=None):
		# scores: (batch, heads, Q, K)
		return scores + mask_bias.to(scores.device)

	def _mask_l0(scores, hook=None):
		# layer-0 special mask: o1/o2 only self; SEP can read d1/d2
		return scores + mask_bias_l0.to(scores.device)

	# Completely suppress attention for oi in L0 (safe: zero pattern rows, not -inf scores)
	def _zero_o_rows(pattern, hook=None):
		# pattern: [B, H, Q, K]
		start_o = LIST_LEN + 1  # first o_i index
		if start_o < SEQ_LEN:
			pattern = pattern.clone()
			pattern[..., start_o:SEQ_LEN, :] = 0.0
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
	n_layers: int,
	n_heads: int,
	d_model: int,
	ln: bool = False,
	use_bias: bool = False,
	freeze_wv: bool = True,
	freeze_wo: bool = True,
	seq_len: int | None = None,
	vocab: int | None = None,
	list_len: int | None = None,
	device: str | torch.device | None = None,
):
	"""Construct a HookedTransformer with our custom mask.

	Args:
		seq_len: total sequence length (must match inputs)
		vocab: vocabulary size
		list_len: number of input digits (used by mask)
		device: device to place model on (defaults to DEV)
	"""
	# Resolve from explicit args or runtime config
	if seq_len is None:
		seq_len = _RUNTIME.seq_len
	if vocab is None:
		vocab = _RUNTIME.vocab
	if list_len is None:
		list_len = _RUNTIME.list_len
	dev = device or _RUNTIME.device or DEV

	assert seq_len is not None and vocab is not None and list_len is not None, "seq_len, vocab, and list_len must be provided or configured via configure_runtime()"
	cfg = HookedTransformerConfig(
		n_layers=n_layers,
		n_heads=n_heads,
		d_model=d_model,
		d_head=d_model // n_heads,
		n_ctx=seq_len,
		d_vocab=vocab,
		attn_only=True,  # no MLP!
		normalization_type=("LN" if ln else None),
	)
	model = HookedTransformer(cfg).to(dev)
	if freeze_wv:
		set_WV_identity_and_freeze(model, d_model)
	if freeze_wo:
		set_WO_identity_and_freeze(model, d_model)
	if not use_bias:
		strip_bias(model)

	attach_custom_mask(model, list_len, seq_len)
	return model


# metrics
def accuracy(m, val_dl, list_len: int | None = None, device: str | torch.device | None = None):
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

	Pass the same kwargs you would pass to make_model (including device, seq_len, vocab, list_len, etc.).
	"""
	device = model_kwargs.get("device", DEV)
	print("Loading model from", path)
	model = make_model(**model_kwargs)
	model.load_state_dict(
		torch.load(path, map_location=device)
	)  # map weights to target device
	model.eval()
	return model