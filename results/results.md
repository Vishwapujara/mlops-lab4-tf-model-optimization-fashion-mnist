| variant | technique | accuracy | accuracy_drop_pp | worst_class_drop_pp | worst_class | size_kb | gzip_kb | gzip_reduction_x | latency_ms | speedup_x | sparsity | max_unique_values |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline (float32) | post-training quantization | 0.8929 | +0.00 | +0.00 | T-shirt/top | 883.5 | 817.0 | 1.0x | 0.025 | 1.00x | 0% | 204516 |
| baseline (float16) | post-training quantization | 0.8928 | +0.01 | +0.10 | T-shirt/top | 445.3 | 407.6 | 2.0x | 0.027 | 0.91x | 0% | 204516 |
| baseline (dynamic) | post-training quantization | 0.8932 | -0.03 | +0.10 | T-shirt/top | 228.3 | 202.4 | 4.0x | 0.034 | 0.74x | 0% | 204516 |
| baseline (int8) | post-training quantization | 0.8926 | +0.03 | +1.10 | Shirt | 231.0 | 202.7 | 4.0x | 0.021 | 1.20x | 0% | 204516 |
| finetune_control (float32) | control | 0.9064 | +0.00 | +0.00 | T-shirt/top | 883.5 | 817.0 | 1.0x | 0.028 | 0.91x | 0% | 204506 |
| qat (int8) | quantization-aware training | 0.9039 | +0.25 | +1.50 | Shirt | 228.3 | 176.6 | 4.6x | 0.021 | 1.20x | 0% | 204493 |
| prune_polynomial_50 (float32) | pruning (polynomial) | 0.9008 | +0.56 | +1.70 | Pullover | 883.5 | 483.5 | 1.7x | 0.030 | 0.84x | 50% | 102228 |
| prune_polynomial_75 (float32) | pruning (polynomial) | 0.8754 | +3.10 | +6.80 | Shirt | 883.5 | 282.6 | 2.9x | 0.030 | 0.85x | 75% | 51167 |
| prune_polynomial_90 (float32) | pruning (polynomial) | 0.7563 | +15.01 | +53.90 | Shirt | 883.5 | 149.3 | 5.5x | 0.026 | 0.95x | 90% | 20495 |
| prune_constant_50 (float32) | pruning (constant) | 0.9026 | +0.38 | +1.70 | Pullover | 883.5 | 484.1 | 1.7x | 0.030 | 0.84x | 50% | 102235 |
| prune_constant_75 (float32) | pruning (constant) | 0.8717 | +3.47 | +10.00 | Shirt | 883.5 | 285.1 | 2.9x | 0.030 | 0.84x | 75% | 51163 |
| prune_constant_90 (float32) | pruning (constant) | 0.7560 | +15.04 | +48.20 | Shirt | 883.5 | 149.6 | 5.5x | 0.030 | 0.84x | 90% | 20473 |
| prune_2by4 (float32) | structured 2:4 pruning | 0.9020 | +0.44 | +1.20 | Pullover | 883.5 | 496.4 | 1.6x | 0.030 | 0.84x | 50% | 102288 |
| cluster_16 (float32) | weight clustering | 0.9063 | +0.01 | +4.10 | Pullover | 883.5 | 146.6 | 5.6x | 0.030 | 0.85x | 0% | 16 |
| prune75_pqat (int8) | collaborative | 0.8831 | +2.33 | +6.50 | Shirt | 228.3 | 82.4 | 9.9x | 0.021 | 1.19x | 75% | 51163 |
| prune75_cluster16_pcqat (int8) | collaborative | 0.8904 | +1.60 | +4.20 | Coat | 228.3 | 55.7 | 14.7x | 0.021 | 1.20x | 75% | 16 |
