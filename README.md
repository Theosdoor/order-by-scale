# Relative‑Magnitude Relational Composition in Attention‑Only Transformers

Code for the paper:
> Farrell, Theo, Patrick Leask, and Noura Al Moubayed. "Order by Scale: Relative‑Magnitude Relational Composition in Attention‑Only Transformers." Socially Responsible and Trustworthy Foundation Models at NeurIPS 2025. https://openreview.net/forum?id=vWRVzNtk7W

<p align="center">
  <img src="figures/attn-flow.png" alt="Information flow in attention-only transformers"/>
  <br/>
  <em>Figure 1: Information flow visualization. Red arrows show attention moving information between token positions (thickness indicates attention weight). The two-layer model (left, 92% accuracy) performs the desired composition at the SEP token, while the three-layer model (right, 100% accuracy) directly copies between positions, avoiding composition. Node numbers show logit lens activations; edge annotations show ablation impact on validation accuracy.</em>
</p>

**TL;DR:** We find a new relational composition method based on relative vector magnitudes in a toy model, challenging the common view that transformer features can be treated as binary on/off switches.

## Overview

This repository implements and analyzes transformers that learn to compress list representations into a special token and then decompress them. The task structure `[d1, d2, SEP, o1, o2]` enables clean mechanistic analysis of information flow through attention layers.

## Architecture

- **Attention-only transformer** (no MLPs)
- **2-3 layers** with single attention head per layer
- **Constrained weights**: Identity value and matrices ($W_V = W_O = I$)
- **Custom attention mask** to enforce causal structure and token-specific attention patterns

## Repository Structure

- `train.ipynb` - Model training with custom attention masks
- `interp_main.ipynb` - Mechanistic interpretability analysis
- `model_utils.py` - Model construction, attention masking, and utilities
- `data.py` - Dataset generation for list compression task
- `models/` - Trained model checkpoints

## Installation

Dependencies are managed with [`uv`](https://docs.astral.sh/uv/). Run `uv sync` to install all dependencies into a local virtual environment.

## Usage

**Training:** Open `train.ipynb` and run cells top-to-bottom. This trains 2- and 3-layer models on the list compression task and saves checkpoints to `models/`.

**Interpretability:** Open `interp_main.ipynb` and run cells top-to-bottom. This notebook loads a trained checkpoint from `models/` and performs mechanistic analysis — attention pattern visualisation, ablation studies, and logit lens inspection. Pre-trained checkpoints are included so you can run the analysis without training from scratch.

## Dependencies

Built with [TransformerLens](https://github.com/TransformerLensOrg/TransformerLens) for mechanistic interpretability.

## Citation

```
@inproceedings{farrell2025order,
  title={Order by Scale: Relative-Magnitude Relational Composition in Attention-Only Transformers},
  author={Theo Farrell and Patrick Leask and Noura Al Moubayed},
  booktitle={Socially Responsible and Trustworthy Foundation Models at NeurIPS 2025},
  year={2025},
  url={https://openreview.net/forum?id=vWRVzNtk7W}
}
```
