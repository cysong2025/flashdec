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

    exact_sequences = 0
    first_token_exact = 0
    common_prefix_tokens = 0
    generated_tokens = 0
    for native_row, flashdec_row in zip(
        native_outputs, flashdec_outputs, strict=True
    ):
        if native_row["prompt"] != flashdec_row["prompt"]:
            raise ValueError("prompt order/content mismatch")
        native_tokens = native_row["token_ids"]
        flashdec_tokens = flashdec_row["token_ids"]
        if not native_tokens or not flashdec_tokens:
            raise ValueError("generated token sequences must be non-empty")
        if len(native_tokens) != len(flashdec_tokens):
            raise ValueError("generated token counts must match per prompt")
        if native_tokens == flashdec_tokens:
            exact_sequences += 1
        if native_tokens[0] == flashdec_tokens[0]:
            first_token_exact += 1
        common_prefix_tokens += next(
            (
                index
                for index, (native_token, flashdec_token) in enumerate(
                    zip(native_tokens, flashdec_tokens, strict=True)
                )
                if native_token != flashdec_token
            ),
            len(native_tokens),
        )
        generated_tokens += len(native_tokens)
    # Different mathematically valid attention reductions need not remain
    # bit-identical over an autoregressive rollout: one near-tied greedy choice
    # changes every subsequent model input.  The end-to-end integration gate is
    # therefore exact first-step top-1 agreement for every fixed prompt.  Full
    # rollout and common-prefix agreement remain visible descriptive evidence;
    # elementwise kernel accuracy is gated by the separate attention benchmark.
    passed = first_token_exact == len(native_outputs)
    lines = [
        "# R7 Qwen2.5-3B Cross-backend Correctness",
        "",
        f"- Device: {native['device']}.",
        f"- PyTorch / vLLM / CUDA: {native['torch_version']} / {native['vllm_version']} / {native['torch_cuda']}.",
        f"- Model config SHA-256: `{native['model_config_sha256']}`.",
        f"- Prompt set SHA-256: `{native['prompts_sha256']}`.",
        (
            "- First-step greedy top-1 tokens equal: "
            f"{first_token_exact}/{len(native_outputs)}."
        ),
        (
            "- Full greedy token sequences equal: "
            f"{exact_sequences}/{len(native_outputs)} (descriptive only)."
        ),
        (
            "- Shared-prefix tokens before autoregressive divergence: "
            f"{common_prefix_tokens}/{generated_tokens} (descriptive only)."
        ),
        f"- Result: **{'PASS' if passed else 'FAIL'}**.",
        "",
        "## Interpretation Boundary",
        "",
        (
            "The pass/fail gate checks the first decode decision from identical "
            "model state. Full-rollout identity is reported but is not a gate: "
            "a near-tied floating-point decision can change all later inputs."
        ),
        (
            "Elementwise FlashDec/vLLM attention-output agreement is validated "
            "separately on the frozen Qwen decode shapes."
        ),
        "",
    ]
    text = "\n".join(lines)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf-8")
    if not passed:
        raise ValueError("first-step cross-backend token mismatch")
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
