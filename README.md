# Mechanistic Interpretability of Simple Attention-Only Transformer

A study of list compression/decompression using a minimal transformer architecture.

PROJECT DIARY (google doc) [here](https://docs.google.com/document/d/1F5Owx5ZKag53pKteuNuVFJ0ztzYpuZwwJYqbJd-9rog/edit?usp=sharing).

## Model Architecture
- **Input format**: `[i1, i2, SPECIAL, o1, o2]`
- **Layer 0**: Compresses list into special token representation
- **Layer 1**: Decompresses special token into output tokens
- **Configuration**: 2-layer, attention-only (no MLP), single head per layer, W_V = I

## Experimental Results so far

### Sample Input
```
[8, 3, 10, 8, 3]
```
Where `10` is the special separator token.

... tbc

## Key Resources
- [Tips for Empirical Alignment Research](https://www.lesswrong.com/posts/dZFpEdKyb9Bf4xYn7/tips-for-empirical-alignment-research)
- [TransformerLens](https://github.com/TransformerLensOrg/TransformerLens)
  - [Main demo](https://colab.research.google.com/github/neelnanda-io/TransformerLens/blob/main/demos/Main_Demo.ipynb#scrollTo=kCXUl7BUPv08)
- [Mechinterp glossary](https://dynalist.io/d/n2ZWtnoYHrU1s4vnFSAQ519J)
- [Grokking mechinterp paper](https://arxiv.org/pdf/2301.05217)