# R7 vLLM Qwen Attention Microbenchmark Summary

## Validation

- Input: `/home/user/flashdec_results/r7_1cc25d4_20260830_2105/vllm_attention.csv`.
- Rows: 50; paired trials: 25.
- Device: NVIDIA GeForce RTX 5070.
- Model shape contract: Qwen2.5-3B-Instruct / bfloat16.
- FlashDec split policy: `auto`.
- PyTorch / Triton / vLLM / PyTorch CUDA: 2.11.0+cu130 / 3.6.0 / 0.25.1 / 13.0.
- Git commit: `1cc25d4df9ffccb5bd804132e19eb1bce01df94e`; clean at start: True.
- Every pair passed full-output cross-backend correctness.

## Paired Results

Ratios are `FlashDec/vLLM Triton`; values below 1 favor FlashDec. Each cell is the median across paired trials.

| case | trials | vLLM Triton p50 ms | FlashDec p50 ms | ratio [min,max] |
| --- | ---: | ---: | ---: | ---: |
| qwen_b1_ctx1024 | 5 | 0.013792 | 0.013792 | 1.0000x [1.0000,1.0000] |
| qwen_b1_ctx128 | 5 | 0.009696 | 0.009952 | 1.0264x [1.0264,1.0297] |
| qwen_b4_ctx1024 | 5 | 0.017888 | 0.017888 | 1.0000x [1.0000,1.0000] |
| qwen_b8_ctx1024 | 5 | 0.030144 | 0.024192 | 0.8025x [0.8017,0.8036] |
| qwen_b8_ctx2048 | 5 | 0.048608 | 0.038528 | 0.7926x [0.7921,0.7931] |

## Frozen Confirmatory Performance Gate

- Required B8 ctx1024/ctx2048 cases <= 0.90x: PASS.
- B1/B4 parity cases <= 1.05x: PASS.
- Every measured case <= 1.05x guardrail: PASS.
- Every case paired-ratio spread <= 0.15: PASS.
- Geometric-mean p50 ratio across cases: 0.9183x.
- Overall external-kernel gate: **PASS**.

## Boundary

This gate compares only single-token decode attention inside the same vLLM KV layout and metadata contract. It is necessary but not sufficient for a model- or serving-level performance claim; those are measured separately with Qwen2.5-3B.
