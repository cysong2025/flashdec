# R7 Qwen2.5-3B vLLM Model Latency Summary

## Validation

- Input: `/home/user/flashdec_results/r7_46c4a4b_20260830_2126/model_latency.csv`.
- Rows: 12; paired process trials: 6.
- Device: NVIDIA GeForce RTX 5070.
- Model: Qwen2.5-3B-Instruct / bfloat16.
- Model config SHA-256: `eed00b17e22553979d090fa492e587e92885e328914c8e0b0b78f0a0d3576b3b`.
- PyTorch / Triton / vLLM / PyTorch CUDA: 2.11.0+cu130 / 3.6.0 / 0.25.1 / 13.0.
- Compilation: `default_inductor_cudagraph`; FlashDec split policy: `auto`.
- Per process: 3 warmup iterations and 5 measured iterations.
- Git commit: `46c4a4be5008baa65a2bff8fa78b04f87d95ba0d`; clean at start: True.

## Paired Results

Ratios are `FlashDec/vLLM Triton`; values below 1 favor FlashDec. Latency is fixed-batch, end-to-end `LLM.generate` time with model loading and compilation excluded.

| case | trials | vLLM p50 ms | FlashDec p50 ms | ratio [min,max] | latency reduction | output TPS uplift |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| qwen_b8_i128_o128 | 3 | 1557.505 | 1551.073 | 0.9958x [0.9952,0.9966] | 0.41% | 0.41% |
| qwen_b8_i2048_o128 | 3 | 3224.064 | 3218.875 | 0.9976x [0.9968,0.9986] | 0.16% | 0.16% |

## Frozen Confirmatory Performance Gate

These pilot-informed thresholds were frozen before the confirmatory three-trial run.
- B8 input2048/output128 target <= 0.995x: FAIL.
- B8 input128/output128 guardrail <= 1.02x: PASS.
- Every case paired-ratio spread <= 0.03: PASS.
- Geometric-mean p50 ratio: 0.9967x.
- Overall external-model gate: **FAIL**.

## Boundary

This is an offline, fixed-batch vLLM model-latency comparison. It includes Qwen transformer execution, scheduling, KV-cache access, sampling, and Python API overhead, but excludes model startup/JIT and does not claim online TTFT/TPOT behavior.
