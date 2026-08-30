# R7 Qwen2.5-3B Cross-backend Correctness

- Device: NVIDIA GeForce RTX 5070.
- PyTorch / vLLM / CUDA: 2.11.0+cu130 / 0.25.1 / 13.0.
- Model config SHA-256: `eed00b17e22553979d090fa492e587e92885e328914c8e0b0b78f0a0d3576b3b`.
- Prompt set SHA-256: `dda43f1a7b3d183a05da444ac8009d390151713640f60d7df23f57547ddaf9f0`.
- First-step greedy top-1 tokens equal: 8/8.
- Full greedy token sequences equal: 5/8 (descriptive only).
- Shared-prefix tokens before autoregressive divergence: 217/256 (descriptive only).
- Result: **PASS**.

## Interpretation Boundary

The pass/fail gate checks the first decode decision from identical model state. Full-rollout identity is reported but is not a gate: a near-tied floating-point decision can change all later inputs.
Elementwise FlashDec/vLLM attention-output agreement is validated separately on the frozen Qwen decode shapes.
