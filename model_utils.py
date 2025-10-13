import torch

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
