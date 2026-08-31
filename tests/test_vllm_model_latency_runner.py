import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).parents[1]
RUNNER_SCRIPT = ROOT / "benchmarks" / "run_vllm_model_latency.py"
WORKER_SCRIPT = ROOT / "benchmarks" / "run_vllm_model_latency_worker.py"


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


RUNNER = _load(RUNNER_SCRIPT, "vllm_model_latency_runner_test")
WORKER = _load(WORKER_SCRIPT, "vllm_model_latency_worker_test")


def test_token_id_dataset_is_deterministic_and_excludes_special_ids():
    case = RUNNER.Case("audit_b2_i4_o3", 2, 4, 3)

    first = RUNNER._generate_dataset(
        case,
        seed=20260830,
        vocab_size=32,
        excluded_token_ids=[0, 7, 31],
    )
    repeated = RUNNER._generate_dataset(
        case,
        seed=20260830,
        vocab_size=32,
        excluded_token_ids=[31, 0, 7],
    )
    changed = RUNNER._generate_dataset(
        case,
        seed=20260831,
        vocab_size=32,
        excluded_token_ids=[0, 7, 31],
    )

    assert first == repeated
    assert first != changed
    assert first["generation_protocol"] == RUNNER.DATASET_GENERATION_PROTOCOL
    assert len(first["prompt_token_ids"]) == 2
    assert all(len(prompt) == 4 for prompt in first["prompt_token_ids"])
    generated = {
        token for prompt in first["prompt_token_ids"] for token in prompt
    }
    assert not ({0, 7, 31} & generated)


def test_special_token_ids_include_tokenizer_added_special_tokens():
    assert RUNNER._special_token_ids(
        {"bos_token_id": 1, "eos_token_id": [2, 3]},
        {
            "added_tokens_decoder": {
                "4": {"content": "<special>", "special": True},
                "5": {"content": "ordinary", "special": False},
            }
        },
        8,
    ) == [1, 2, 3, 4]


def test_dataset_file_hash_is_canonical_and_worker_validates_it(tmp_path):
    case = RUNNER.Case("audit_b1_i3_o2", 1, 3, 2)
    dataset = RUNNER._generate_dataset(
        case,
        seed=17,
        vocab_size=16,
        excluded_token_ids=[1, 15],
    )
    path = tmp_path / "dataset.json"

    digest = RUNNER._write_dataset(path, dataset)

    assert digest == hashlib.sha256(path.read_bytes()).hexdigest()
    assert path.read_bytes() == RUNNER._canonical_json_bytes(dataset)
    assert WORKER._load_dataset(path, digest) == dataset
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        WORKER._load_dataset(path, "0" * 64)


def test_runner_invokes_repository_worker_with_dataset_identity(tmp_path):
    dataset = (tmp_path / "dataset.json").resolve()
    worker = (tmp_path / "worker.py").resolve()
    output = tmp_path / "result.json"
    digest = "a" * 64

    command = RUNNER._command(
        worker=worker,
        model=Path("/model"),
        dataset=dataset,
        dataset_sha256=digest,
        backend="CUSTOM",
        warmup_iters=2,
        num_iters=5,
        sampling_seed=23,
        gpu_memory_utilization=0.78,
        max_model_len=4096,
        max_num_seqs=8,
        max_num_batched_tokens=2048,
        output_json=output,
    )

    assert command[1] == str(worker)
    assert "bench" not in command
    assert "latency" not in command
    assert command[command.index("--dataset") + 1] == str(dataset)
    assert command[command.index("--dataset-sha256") + 1] == digest
    assert command[command.index("--sampling-seed") + 1] == "23"


def test_decode_heavy_builtins_are_explicit_and_custom_cases_are_supported():
    defaults = RUNNER._resolve_cases(None, None)
    assert [case.name for case in defaults] == [
        "qwen_b8_i128_o128",
        "qwen_b8_i2048_o128",
    ]

    long_case = RUNNER._resolve_cases(["qwen_b8_i2048_o2048"], None)
    assert long_case == [RUNNER.Case("qwen_b8_i2048_o2048", 8, 2048, 2048, False)]

    r8_target = RUNNER._resolve_cases(["qwen_b8_i8192_o4096"], None)
    assert r8_target == [RUNNER.Case("qwen_b8_i8192_o4096", 8, 8192, 4096, False)]

    custom = RUNNER._resolve_cases(None, ["qwen_b4_i3072_o512:4:3072:512"])
    assert custom == [RUNNER.Case("qwen_b4_i3072_o512", 4, 3072, 512, False)]


def test_worker_rejects_prompt_shape_mismatch_even_with_matching_hash(tmp_path):
    dataset = {
        "schema_version": WORKER.DATASET_SCHEMA_VERSION,
        "generation_protocol": WORKER.DATASET_GENERATION_PROTOCOL,
        "seed": 1,
        "case": "broken",
        "batch_size": 1,
        "input_len": 2,
        "output_len": 1,
        "vocab_size": 8,
        "excluded_token_ids": [],
        "prompt_token_ids": [[3]],
    }
    path = tmp_path / "broken.json"
    path.write_text(json.dumps(dataset), encoding="utf-8")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()

    with pytest.raises(ValueError, match="prompt length"):
        WORKER._load_dataset(path, digest)


def test_worker_percentile_uses_linear_interpolation():
    assert WORKER._percentile([1.0, 2.0, 3.0, 4.0], 50.0) == 2.5
    assert WORKER._percentile([1.0, 2.0, 3.0, 4.0], 90.0) == pytest.approx(3.7)


def test_runner_rejects_inconsistent_worker_output_hashes(tmp_path):
    case = RUNNER.Case("audit_b1_i2_o2", 1, 2, 2)
    dataset = tmp_path / "dataset.json"
    digest = "a" * 64
    result = {
        "schema_version": 1,
        "backend_arg": "CUSTOM",
        "case": case.name,
        "batch_size": case.batch_size,
        "input_len": case.input_len,
        "output_len": case.output_len,
        "dataset_path": str(dataset.resolve()),
        "dataset_sha256": digest,
        "dataset_seed": 17,
        "dataset_generation_protocol": RUNNER.DATASET_GENERATION_PROTOCOL,
        "prompt_format": "token_ids",
        "skip_tokenizer_init": True,
        "sampling_seed": 23,
        "sampling_n": 1,
        "sampling_temperature": 0.0,
        "sampling_min_tokens": case.output_len,
        "sampling_max_tokens": case.output_len,
        "sampling_ignore_eos": True,
        "sampling_detokenize": False,
        "warmup_iters": 1,
        "num_iters": 1,
        "timing_scope": RUNNER.TIMING_SCOPE,
        "vllm_engine_multiprocessing": True,
        "accuracy_prefix_len": RUNNER.ACCURACY_PREFIX_LEN,
        "latencies_s": [1.0],
        "avg_latency_s": 1.0,
        "percentiles_s": {"50": 1.0, "90": 1.0},
        "warmup_output_sha256": ["b" * 64],
        "measured_output_sha256": ["c" * 64],
        "output_token_ids_sha256": "c" * 64,
        "output_token_ids": [[1, 2]],
    }

    with pytest.raises(ValueError, match="outputs differ"):
        RUNNER._validate_worker_result(
            result,
            case=case,
            backend_arg="CUSTOM",
            dataset_path=dataset,
            dataset_sha256=digest,
            dataset_seed=17,
            sampling_seed=23,
            warmup_iters=1,
            num_iters=1,
        )


def test_cross_backend_parity_requires_first_custom_decode_decision():
    parity = RUNNER._cross_backend_parity(
        ((1, 2, 3, 4), (5, 6, 7, 8)),
        ((1, 2, 9, 4), (5, 6, 7, 8)),
    )

    assert parity == {
        "cross_backend_exact_sequences": 1,
        "cross_backend_common_prefix_tokens": 6,
        "cross_backend_min_common_prefix_tokens": 2,
        "cross_backend_generated_tokens": 8,
        "cross_backend_full_hash_equal": False,
        "cross_backend_accuracy_prefix_pass": True,
    }
