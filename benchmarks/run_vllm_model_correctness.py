#!/usr/bin/env python3
"""Generate fixed Qwen outputs for one vLLM attention backend."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from datetime import datetime
from pathlib import Path

import torch
import vllm
from vllm import LLM, SamplingParams


PROMPTS = (
    "Explain in one sentence why paged KV caches help LLM serving.",
    "Write a concise Python docstring for a single-token decode function.",
    "用一句话解释什么是分组查询注意力。",
    "列出两个验证 GPU kernel 正确性的关键步骤。",
    "A request has 128 prompt tokens and generates 32 tokens. What is the total?",
    "Summarize the purpose of an attention block table in twelve words or fewer.",
    "If latency falls from 10 ms to 8 ms, state the percentage reduction.",
    "Return a short reminder about comparing benchmarks on identical workloads.",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_value(*args: str) -> str:
    completed = subprocess.run(
        ["git", *args],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return completed.stdout.strip() if completed.returncode == 0 else "unknown"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=("TRITON_ATTN", "CUSTOM"), required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20260830)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    if args.max_tokens <= 0:
        raise ValueError("max-tokens must be positive")
    if os.environ.get("VLLM_USE_FLASHINFER_SAMPLER") != "0":
        raise RuntimeError("set VLLM_USE_FLASHINFER_SAMPLER=0 for the frozen R7 protocol")
    if os.environ.get("VLLM_WSL2_ENABLE_PIN_MEMORY") != "1":
        raise RuntimeError("set VLLM_WSL2_ENABLE_PIN_MEMORY=1 for the frozen WSL protocol")

    model_config = args.model / "config.json"
    checksums = args.model / "SHA256SUMS"
    if not model_config.is_file() or not checksums.is_file():
        raise FileNotFoundError("model config.json and SHA256SUMS are required")

    llm = LLM(
        model=str(args.model),
        tokenizer=str(args.model),
        dtype="bfloat16",
        seed=args.seed,
        gpu_memory_utilization=0.78,
        max_model_len=8192,
        max_num_seqs=8,
        max_num_batched_tokens=2048,
        kv_cache_dtype="bfloat16",
        enable_prefix_caching=False,
        enforce_eager=True,
        attention_config={"backend": args.backend},
    )
    params = SamplingParams(
        temperature=0.0,
        max_tokens=args.max_tokens,
        ignore_eos=True,
        seed=args.seed,
    )
    outputs = llm.generate(list(PROMPTS), params, use_tqdm=False)
    records = []
    for prompt, request_output in zip(PROMPTS, outputs, strict=True):
        generation = request_output.outputs[0]
        records.append(
            {
                "prompt": prompt,
                "token_ids": list(generation.token_ids),
                "text": generation.text,
            }
        )

    payload = {
        "schema_version": 1,
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "backend": args.backend,
        "git_commit": _git_value("rev-parse", "HEAD"),
        "git_worktree_clean": _git_value("status", "--porcelain") == "",
        "device": torch.cuda.get_device_name(),
        "torch_version": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "vllm_version": vllm.__version__,
        "model_path": str(args.model.resolve()),
        "model_config_sha256": _sha256(model_config),
        "model_manifest_sha256": _sha256(checksums),
        "seed": args.seed,
        "max_tokens": args.max_tokens,
        "prompts_sha256": hashlib.sha256(
            json.dumps(PROMPTS, ensure_ascii=False).encode("utf-8")
        ).hexdigest(),
        "outputs": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Wrote {len(records)} outputs to {args.output}")


if __name__ == "__main__":
    main()
