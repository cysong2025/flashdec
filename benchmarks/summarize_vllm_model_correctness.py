#!/usr/bin/env python3
"""Compare fixed Qwen outputs from vLLM Triton and FlashDec backends."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def summarize(native_path: Path, flashdec_path: Path, output_path: Path) -> str:
    native = json.loads(native_path.read_text(encoding="utf-8"))
    flashdec = json.loads(flashdec_path.read_text(encoding="utf-8"))
    if native.get("backend") != "TRITON_ATTN":
        raise ValueError("native input must use TRITON_ATTN")
    if flashdec.get("backend") != "CUSTOM":
        raise ValueError("FlashDec input must use CUSTOM")
    invariant_fields = (
        "schema_version",
        "git_commit",
        "git_worktree_clean",
        "device",
        "torch_version",
        "torch_cuda",
        "vllm_version",
        "model_path",
        "model_config_sha256",
        "model_manifest_sha256",
        "seed",
        "max_tokens",
        "prompts_sha256",
    )
    for field in invariant_fields:
        if native.get(field) != flashdec.get(field):
            raise ValueError(f"invariant mismatch: {field}")
    native_outputs = native.get("outputs", [])
    flashdec_outputs = flashdec.get("outputs", [])
    if not native_outputs or len(native_outputs) != len(flashdec_outputs):
        raise ValueError("output counts must be equal and non-zero")

    exact = 0
    for native_row, flashdec_row in zip(
        native_outputs, flashdec_outputs, strict=True
    ):
        if native_row["prompt"] != flashdec_row["prompt"]:
            raise ValueError("prompt order/content mismatch")
        if native_row["token_ids"] == flashdec_row["token_ids"]:
            exact += 1
    passed = exact == len(native_outputs)
    lines = [
        "# R7 Qwen2.5-3B Cross-backend Correctness",
        "",
        f"- Device: {native['device']}.",
        f"- PyTorch / vLLM / CUDA: {native['torch_version']} / {native['vllm_version']} / {native['torch_cuda']}.",
        f"- Model config SHA-256: `{native['model_config_sha256']}`.",
        f"- Prompt set SHA-256: `{native['prompts_sha256']}`.",
        f"- Greedy token sequences equal: {exact}/{len(native_outputs)}.",
        f"- Result: **{'PASS' if passed else 'FAIL'}**.",
        "",
    ]
    text = "\n".join(lines)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf-8")
    if not passed:
        raise ValueError("cross-backend generated token mismatch")
    return text


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("native", type=Path)
    parser.add_argument("flashdec", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(summarize(args.native, args.flashdec, args.output))


if __name__ == "__main__":
    main()
