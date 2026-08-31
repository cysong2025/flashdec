# R8 Qwen2.5-3B vLLM Model Latency Summary

## Validation

- Input: `/home/user/flashdec_results/r8_3ba68e3_formal_trials4/model_latency.csv`.
- Rows: 16; paired backend process pairs: 8 (4 trials per case).
- Device: NVIDIA GeForce RTX 5070.
- Model: Qwen2.5-3B-Instruct / bfloat16.
- Model config SHA-256: `eed00b17e22553979d090fa492e587e92885e328914c8e0b0b78f0a0d3576b3b`.
- Prompt dataset: fixed token IDs; seed `20260830`; protocol `sha256-indexed-u64be-mod-model-tokenizer-nonspecial-v2`.
- Decoding: greedy (`temperature=0`, `n=1`), fixed output length (`ignore_eos=True`), and detokenization disabled.
- PyTorch / Triton / vLLM / PyTorch CUDA: 2.11.0+cu130 / 3.6.0 / 0.25.1 / 13.0.
- Compilation: `default_inductor_cudagraph`; FlashDec split policy: `auto`.
- Commit-scoped vLLM cache: `/home/user/flashdec_results/r8_3ba68e3_formal_trials4/vllm-cache/3ba68e3f5317f1a9e2f0a2830697c55de6dfe9d0`.
- Per-process iteration counts: full-length JIT-prime `1`; full-length warmup `1`; measured `1`.
- JIT-prime output hashes are retained in raw worker JSON for audit only; warmup and measured hashes remain the determinism gate.
- Integration guardrail: `qwen_b8_i512_o2` uses the 512-token prompt boundary and generates exactly two tokens; the second generated token covers the first eligible FlashDec split decode.
- Split activation: every CUSTOM worker supplied a unique, canonical engine-process marker proving a successful multi-split FlashDec launch before warmup/timing.
- First observed split occurred during CUDA Graph capture in 8/8 CUSTOM workers (required by the fail-closed activation gate).
- Git commit: `3ba68e3f5317f1a9e2f0a2830697c55de6dfe9d0`; clean at start: True.
- Per-case prompt dataset identities:
  - `qwen_b8_i512_o2`: `1551c91be5c4d7ccaa41abf0884e1cb36a26b13a3bff4c6094e3f91c3242d554` (`/home/user/flashdec_results/r8_3ba68e3_formal_trials4/model_latency_raw/datasets/qwen_b8_i512_o2.json`).
  - `qwen_b8_i8192_o4096`: `9114886d56796d3ae9e699b79585417c8b935ccf09e0322aa84c9a074d98cec3` (`/home/user/flashdec_results/r8_3ba68e3_formal_trials4/model_latency_raw/datasets/qwen_b8_i8192_o4096.json`).
- Per-case generated-token identities:
  - `qwen_b8_i512_o2`: minimum cross-backend common prefix `2` tokens/request; 1 unique full-rollout SHA-256 (descriptive only).
  - `qwen_b8_i8192_o4096`: minimum cross-backend common prefix `49` tokens/request; 2 unique full-rollout SHA-256 (descriptive only).

## Paired Results

Ratios are `FlashDec/vLLM Triton`; values below 1 favor FlashDec. Latency is fixed-batch, end-to-end `LLM.generate` time with model loading and compilation excluded. Reduction and TPS uplift are derived from the same paired-median ratio used by the gate.

| case | trials | vLLM p50 ms | FlashDec p50 ms | ratio [min,max] | latency reduction | output TPS uplift |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| qwen_b8_i512_o2 | 4 | 401.357 | 403.915 | 1.0029x [0.9890,1.0100] | -0.29% | -0.29% |
| qwen_b8_i8192_o4096 | 4 | 78118.907 | 74578.237 | 0.9542x [0.9530,0.9560] | 4.58% | 4.80% |

## Frozen Confirmatory Performance Gate

These pilot-informed thresholds were frozen before the confirmatory four-trial balanced AB/BA run.
- B8 input8192/output4096 target <= 0.970x (at least 3% end-to-end latency reduction): PASS.
- B8 input512/output2 two-token split-decode guardrail <= 1.05x: PASS.
- Every case paired-ratio spread <= 0.03: PASS.
- Every request shares at least 2 output tokens across backends (the second generated token covers the first eligible FlashDec split decode): PASS.
- Geometric-mean p50 ratio: 0.9783x.
- Overall external-model gate: **PASS**.

## Reproduction command

The WSL run used the following project entry points; `$RESULT_DIR` was `/home/user/flashdec_results/r8_3ba68e3_formal_trials4`.

```bash
export VLLM_USE_FLASHINFER_SAMPLER=0
export VLLM_WSL2_ENABLE_PIN_MEMORY=1
export VLLM_PLUGINS=flashdec
export VLLM_ENABLE_V1_MULTIPROCESSING=1
unset FLASHDEC_VLLM_NUM_SPLITS

(cd /home/user/models/Qwen2.5-3B-Instruct && \
  sha256sum --check SHA256SUMS)

python benchmarks/run_vllm_model_latency.py \
  --model /home/user/models/Qwen2.5-3B-Instruct \
  --output "$RESULT_DIR/model_latency.csv" \
  --case qwen_b8_i512_o2 --case qwen_b8_i8192_o4096 \
  --trials 4 --prime-iters 1 --warmup-iters 1 --num-iters 1 \
  --gpu-memory-utilization 0.85 --max-model-len 12288 \
  --max-num-seqs 8 --max-num-batched-tokens 2048 \
  --vllm-cache-base "$RESULT_DIR/vllm-cache" --require-clean

python benchmarks/summarize_vllm_model_latency.py \
  "$RESULT_DIR/model_latency.csv" \
  --output "$RESULT_DIR/model_latency_summary.md"
```

## Retained raw evidence

Raw CSV, logs, datasets, worker JSON, eight unique activation markers, and the commit-scoped vLLM cache remain outside Git under the input directory above. Their top-level SHA-256 identities are:

| artifact | SHA-256 |
| --- | --- |
| `model_latency.csv` | `ae57b1788abb61847e1faa4ee1a6ab57de0fba309c2cb5317d660e4913d503e2` |
| generated `model_latency_summary.md` | `f511af02757b66cc75007768c5df7e9180ae31f3ed34853d00d00038e9354520` |
| `run.log` | `8c1956118f877f4f006c4fba50f9c91c814dcc283457c84ebc3ec59723a4b7f7` |
| `summary.log` | `1c8b5e95716fbd1a84528e8d39d6420d1d0b7b3a9023403c854edba2a4509bc6` |
| `evidence_manifest.sha256` | `cf7ea96e39133ff4bf12959877b177c31547d9eb6f17273e94ab35d19946fd57` |

The generated-summary hash identifies the unmodified summarizer output. This tracked canonical copy retains that output and adds only the reproduction command and raw-evidence identities.

## Boundary

This is an offline, fixed-batch vLLM model-latency comparison. It includes Qwen transformer execution, scheduling, KV-cache access, sampling, and Python API overhead, but excludes model startup/JIT and does not claim online TTFT/TPOT behavior.
The first two greedy tokens must match from identical prompt state; at the 512-token prompt boundary, the second generated token covers the first eligible FlashDec split-decode decision. Full autoregressive rollout hashes are descriptive because one near-tied floating-point choice can change all later inputs.
