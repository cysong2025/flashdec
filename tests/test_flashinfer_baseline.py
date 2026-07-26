"""CUDA correctness coverage for the optional FlashInfer baseline."""

from importlib import import_module, util

import pytest


torch = pytest.importorskip("torch")
pytest.importorskip("triton")

from benchmarks.run_flashinfer_baseline import (
    CASES,
    EXPECTED_FLASHINFER_VERSION,
    FLASHINFER_BACKEND,
    FLASHINFER_WORKSPACE_MIB,
    _flashinfer_version,
    _make_flashinfer_wrapper,
    _make_inputs,
    _validate_outputs,
    _validate_flashinfer_environment,
)


if util.find_spec("flashinfer") is None:
    pytest.skip("could not import 'flashinfer'", allow_module_level=True)
if not torch.cuda.is_available():
    pytest.skip("FlashInfer baseline requires CUDA", allow_module_level=True)
try:
    _validate_flashinfer_environment(torch)
except RuntimeError as exc:
    pytest.fail(str(exc), pytrace=False)
try:
    flashinfer = import_module("flashinfer")
except Exception as exc:
    pytest.fail(f"installed flashinfer failed to import: {exc}", pytrace=False)


@pytest.mark.parametrize("dtype_name", ["float16", "bfloat16"])
def test_flashinfer_fa2_paths_match_flashdec_and_reference(dtype_name):
    if dtype_name == "bfloat16" and not torch.cuda.is_bf16_supported():
        pytest.skip("CUDA device does not report BF16 support")
    assert _flashinfer_version(flashinfer) == EXPECTED_FLASHINFER_VERSION

    from flashdec.kernels.paged_decode import paged_decode_attention

    dtype = getattr(torch, dtype_name)
    inputs = _make_inputs(torch, CASES["small"], dtype, seed=1701)
    core_wrapper, core_workspace = _make_flashinfer_wrapper(
        torch,
        flashinfer,
        inputs,
        dtype=dtype,
        use_tensor_cores=False,
        workspace_mib=FLASHINFER_WORKSPACE_MIB,
        backend=FLASHINFER_BACKEND,
    )
    tensor_wrapper, tensor_workspace = _make_flashinfer_wrapper(
        torch,
        flashinfer,
        inputs,
        dtype=dtype,
        use_tensor_cores=True,
        workspace_mib=FLASHINFER_WORKSPACE_MIB,
        backend=FLASHINFER_BACKEND,
    )
    kv_cache = (inputs["k_cache"], inputs["v_cache"])
    callables = {
        "flashdec_triton": lambda: paged_decode_attention(
            inputs["q"],
            inputs["k_cache"],
            inputs["v_cache"],
            inputs["block_tables"],
            inputs["seq_lens"],
            block_size=32,
            num_warps=2,
            kv_layout="token_major",
        ),
        "flashinfer_fa2_cuda_core": lambda: core_wrapper.run(
            inputs["q"], kv_cache
        ),
        "flashinfer_fa2_tensor_core": lambda: tensor_wrapper.run(
            inputs["q"], kv_cache
        ),
    }
    validation = _validate_outputs(torch, inputs, callables, dtype_name)

    assert validation["reference_sample_size"] == 1
    assert set(validation["reference_errors"]) == set(callables)
    assert set(validation["cross_errors"]) == set(callables)
    assert set(validation["reference_tolerance_ratios"]) == set(callables)
    assert set(validation["cross_tolerance_ratios"]) == set(callables)
    assert max(validation["reference_tolerance_ratios"].values()) <= 1.0 + 1e-7
    assert max(validation["cross_tolerance_ratios"].values()) <= 1.0 + 1e-7
    assert core_workspace.numel() == FLASHINFER_WORKSPACE_MIB * 1024 * 1024
    assert tensor_workspace.numel() == FLASHINFER_WORKSPACE_MIB * 1024 * 1024
