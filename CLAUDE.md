# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Mechanistic interpretability research on attention-only transformers trained on a list-compression task. Models learn to compress `[d1, d2, ..., dn]` into a SEP token and reconstruct it, with sequence structure `[d1, d2, SEP, o1, o2]`. Paper: "Order by Scale: Relative-Magnitude Relational Composition in Attention-Only Transformers" (NeurIPS 2025).

## Commands

```bash
uv sync                          # install dependencies into .venv
python3 train.py                 # train default 2-layer, 1-head, d=64, 100-digit model
python3 train.py --help          # full flag list
python3 train.py --n-layers 3 --max-steps 200000 --wandb
```

Key training flags: `--n-layers`, `--n-heads`, `--d-model`, `--n-digits`, `--list-len`, `--ln`, `--wv`, `--wo`, `--mlp`, `--lr`, `--weight-decay`, `--seed`, `--wandb`.

W&B: copy `.env.example` → `.env` and fill credentials before using `--wandb`.

There is no test suite or linter. Validation runs automatically every 100 training steps.

## Architecture

**`src/runtime.py`** — singleton `_RUNTIME` holding `list_len`, `seq_len`, `vocab`, `device`, `seed`. Call `configure_runtime()` once at startup; all other functions read from it automatically.

**`src/model_utils.py`** — everything model-related:
- `make_model(...)` — factory for `HookedTransformer` (TransformerLens). By default freezes W_V and W_O to identity (`use_wv=False`, `use_wo=False`) and strips biases.
- `attach_custom_mask` / `build_attention_mask` — registers perma-hooks enforcing the causal structure. Layer 0 is special: output tokens (`o_i`) only self-attend; SEP can read all digits. Layers 1+: standard causal mask with outputs blocked from attending to inputs.
- `infer_model_config(path)` — inspects a checkpoint and returns the full config dict, used by notebooks to load models without hard-coding config.
- `ModelConfig` (NamedTuple) + `parse_model_name` / `parse_model_name_safe` — parse config from filename. Two naming conventions: new `L2_H1_D64_V100[_len3][_flags]` and legacy `2layer_100dig_64d`.
- `accuracy`, `count_params`, `save_model`, `load_model`.

**`src/datasets.py`** — generates train/val split. Inputs mask the output positions; targets are full `[d1, d2, SEP, o1, o2]`. 80/20 split by default. Important: always use default `train_split=0.8` when calling `accuracy` to avoid evaluating on training data.

**`src/interp_utils.py`** — ablation framework: systematically zero-out attention positions and re-run the model to identify critical edges, with optional attention renormalization.

**`interp.ipynb` / `interp.py`** — main interpretability analysis. Loads pre-trained checkpoints from `models/` via `infer_model_config` + `load_model`; no training required.

## Key Invariants

- `n_ctx = list_len * 2 + 1` (always odd; `infer_model_config` validates this)
- `vocab = n_digits + 2` (MASK token + SEP token)
- Output logits are sliced at `[:, list_len + 1:]` to skip input and SEP positions
- Masks are attached as **perma-hooks** — they survive `model.reset_hooks()` calls during ablation studies
