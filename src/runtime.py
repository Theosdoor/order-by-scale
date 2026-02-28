"""Runtime configuration utilities."""

import torch
from dataclasses import dataclass

__all__ = [
    "configure_runtime",
    "_RUNTIME",
]

torch.manual_seed(0)

# ---------- runtime config (optional) ----------
@dataclass
class _RuntimeConfig:
    list_len = None
    seq_len = None
    vocab = None
    device = None
    seed = None

_RUNTIME = _RuntimeConfig()

def configure_runtime(*, list_len, seq_len, vocab, device, seed=0):
    """Set module-level runtime configuration so callers can omit repeating constants."""
    _RUNTIME.list_len = list_len
    _RUNTIME.seq_len = seq_len
    _RUNTIME.vocab = vocab
    _RUNTIME.device = device
    _RUNTIME.seed = seed
    torch.manual_seed(seed)
