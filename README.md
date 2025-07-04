# Mechanistic Interpretability of Simple Attention-Only Transformer

A study of list compression/decompression using a minimal transformer architecture.

## Model Architecture
- **Input format**: `[i1, i2, SPECIAL, o1, o2]`
- **Layer 0**: Compresses list into special token representation
- **Layer 1**: Decompresses special token into output tokens
- **Configuration**: 2-layer, attention-only (no MLP), single head per layer

## TODO
- [x] Get attention maps for both layers
- [ ] Zero out individual entries in each layer
- [ ] Analyze embedding/unembedding matrices with PCA
- [ ] Test with different list lengths
- [ ] Check Neel's grokking paper formulas

## Experimental Results so far

### Sample Input
```
[8, 3, 10, 8, 3]
```
Where `10` is the special separator token.

### Mean Attention Pattern Analysis

**Layer 0 - Mean Pattern:**
```
Position:  0      1      2      3      4
      0 [1.000  0.000  0.000  0.000  0.000]
      1 [0.014  0.986  0.000  0.000  0.000]
      2 [0.032  0.435  0.533  0.000  0.000]
      3 [0.000  0.000  0.845  0.155  0.000]
      4 [0.000  0.000  0.443  0.245  0.312]
```

**Layer 0 - Sample Pattern:**
```
Position:  0      1      2      3      4
      0 [1.000  0.000  0.000  0.000  0.000]
      1 [0.013  0.987  0.000  0.000  0.000]
      2 [0.030  0.391  0.580  0.000  0.000]
      3 [0.000  0.000  0.789  0.211  0.000]
      4 [0.000  0.000  0.315  0.196  0.489]
```

**Loss Comparison:**
- Original Loss: `0.651`
- Mean Pattern Ablated Loss: `0.652`

**This suggests that...**
- Replacing individual attention patterns with mean patterns has minimal impact on loss
- Suggests the model relies on **positional structure** rather than specific digit values
- The compression mechanism appears to be digit-agnostic

### Zeroing out individual attn scores

<!-- ### Key Findings so far
 -->

## Next Steps
1. Test specific attention position ablations
2. Analyze value vector contributions
3. Investigate the compressed representation structure
4. Map out the algorithmic circuits learned by the model

## Key Resources
- [Tips for Empirical Alignment Research](https://www.lesswrong.com/posts/dZFpEdKyb9Bf4xYn7/tips-for-empirical-alignment-research)
- [TransformerLens](https://github.com/TransformerLensOrg/TransformerLens)
  - [Main demo](https://colab.research.google.com/github/neelnanda-io/TransformerLens/blob/main/demos/Main_Demo.ipynb#scrollTo=kCXUl7BUPv08)
- [Mechinterp glossary](https://dynalist.io/d/n2ZWtnoYHrU1s4vnFSAQ519J)
- [Grokking mechinterp paper](https://arxiv.org/pdf/2301.05217)