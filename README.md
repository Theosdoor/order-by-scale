# Mech interp of a simple attn-only transformer for list compression / decompression

Input = [i1, i2, SPECIAL, o1, o2]
Layer 0 = list compress into special token
Layer 1 = decompress into output tokens


## TODO
* Get attn maps for both layers
* zero out individual entries (in each layer!)


## Results so far
(where sample input is [8, 3, 10, 8, 3])

### Mean attention pattern
Mean pattern calculated: 
[1.         0.         0.         0.         0.        ]
[0.01445366 0.98554623 0.         0.         0.        ]
[0.03167263 0.43538564 0.5329417  0.         0.        ]
[0.         0.         0.84494275 0.15505722 0.        ]
[0.         0.         0.44315127 0.24488218 0.3119666 ]

Original pattern: 
[1.         0.         0.         0.         0.        ]
[0.01287321 0.98712677 0.         0.         0.        ]
[0.02960602 0.3908043  0.5795897  0.         0.        ]
[0.         0.         0.7891883  0.21081167 0.        ]
[0.         0.         0.31488845 0.19573739 0.48937416]

Original Loss: 0.651
Ablated Loss (Using Mean Attention Pattern): 0.652