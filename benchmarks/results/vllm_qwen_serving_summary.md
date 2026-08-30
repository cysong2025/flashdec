# R7 Qwen2.5-3B vLLM Online Serving Summary

## Validation

- Input: `/home/user/flashdec_results/r7_7dcb19c_20260830_2148/serving.csv`.
- Rows: 6; paired server trials: 3.
- Device: NVIDIA GeForce RTX 5070.
- Model: Qwen2.5-3B-Instruct / bfloat16.
- Model config SHA-256: `eed00b17e22553979d090fa492e587e92885e328914c8e0b0b78f0a0d3576b3b`.
- PyTorch / Triton / vLLM / PyTorch CUDA: 2.11.0+cu130 / 3.6.0 / 0.25.1 / 13.0.
- Workload: 128 prompts, concurrency 8, input/output 4096/128, request rate inf.
- Warmups: 8; prefix caching: False; compilation: `default_inductor_cudagraph`.
- Git commit: `7dcb19ccfcca7a3fe1984a06148d62a17336277d`; clean at start: True.
- Every run completed 128/128 requests with zero failures.

## Paired Results

Latency ratios are `FlashDec/vLLM Triton`; throughput ratio is `FlashDec/vLLM Triton`. Values below 1 favor FlashDec latency, while values above 1 favor FlashDec throughput.

| metric | vLLM Triton | FlashDec | paired ratio [min,max] |
| --- | ---: | ---: | ---: |
| median TPOT ms | 31.9200 | 31.8152 | 0.9969x [0.9967,0.9969] |
| p90 TPOT ms | 35.3043 | 35.1923 | 0.9973x [0.9968,0.9974] |
| median TTFT ms | 1113.221 | 1113.990 | 1.0007x [1.0002,1.0016] |
| output throughput tok/s | 197.816 | 197.751 | 1.0019x [0.9994,1.0030] |

## Frozen Confirmatory Performance Gate

These pilot-informed thresholds were frozen before the confirmatory three-trial run.
- Median TPOT ratio <= 0.998x: PASS.
- Output-throughput ratio >= 1.002x: FAIL.
- p90 TPOT ratio <= 1.02x: PASS.
- Median TTFT ratio <= 1.05x: PASS.
- TPOT paired-ratio spread <= 0.02: PASS.
- Overall external-serving gate: **FAIL**.

## Boundary

This is a saturated local HTTP serving comparison on one RTX 5070. It measures the same vLLM scheduler, API server, model, cache policy, and request stream; only the eligible single-token decode attention backend differs. It is not a multi-GPU or distributed-serving claim.
