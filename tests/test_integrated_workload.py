"""CPU/reference integration coverage for the full integrated workload."""

import shutil

import pytest


torch = pytest.importorskip("torch")
from torch.utils.cpp_extension import CUDA_HOME

from flashdec.cache import PagedKVCache
from flashdec.engine import DecodeEngine
from flashdec.integrated_workload import (
    run_integrated_workload,
    standard_integrated_config,
)
from flashdec.scheduler import BlockAwareScheduler, RequestSpec, SchedulerConfig


HAS_CUDA_TOOLCHAIN = (
    torch.cuda.is_available()
    and CUDA_HOME is not None
    and shutil.which("nvcc") is not None
)
CUDA_TOOLCHAIN_REASON = "CUDA GPU, CUDA_HOME, and nvcc are required"


def _prefix(cache, num_blocks):
    shape = (
        cache.num_layers,
        num_blocks,
        cache.num_kv_heads,
        cache.block_size,
        cache.head_dim,
    )
    k = torch.arange(
        torch.tensor(shape).prod().item(),
        dtype=cache.dtype,
        device=cache.device,
    ).reshape(shape)
    return k, k + 1000


def test_multi_layer_prompt_prefill_commits_once_and_rolls_back_on_failure(monkeypatch):
    cache = PagedKVCache(
        num_layers=2,
        num_kv_heads=1,
        head_dim=4,
        block_size=2,
        max_blocks=2,
        dtype=torch.float32,
        device="cpu",
    )
    engine = DecodeEngine(cache)
    engine.submit_request(RequestSpec("request", 1, 1, 0))
    scheduler = BlockAwareScheduler(
        SchedulerConfig(max_active_requests=1, max_batch_requests=1)
    )
    snapshot = engine.scheduling_snapshot(logical_step=0)
    decision = scheduler.plan(snapshot)
    engine.apply_scheduler_decision(
        decision,
        scheduler=scheduler,
        snapshot=snapshot,
    )
    k = torch.randn((2, 1, 4))
    v = torch.randn_like(k)
    original = cache.write_token_layer

    def fail_second_layer(transaction, layer_idx, layer_k, layer_v):
        if layer_idx == 1:
            raise RuntimeError("injected prefill failure")
        return original(transaction, layer_idx, layer_k, layer_v)

    monkeypatch.setattr(cache, "write_token_layer", fail_second_layer)
    with pytest.raises(RuntimeError, match="injected prefill failure"):
        engine.prefill_request_layers("request", k, v)
    assert cache.request_state("request")["seq_len"] == 0
    assert cache.request_block_ids("request") == ()
    assert cache.metrics()["transaction_abort_count"] == 1
    assert engine.validate_invariants()

    monkeypatch.setattr(cache, "write_token_layer", original)
    assert engine.prefill_request_layers("request", k, v) == 1
    assert cache.request_state("request")["seq_len"] == 1
    assert cache.metrics()["transaction_commit_count"] == 1
    assert cache.metrics()["transaction_layer_write_count"] == 3
    assert engine.validate_invariants()


def test_integrated_trace_matches_reference_rolls_back_reuses_and_cleans_up():
    config = standard_integrated_config(num_layers=2, context_tokens=64)
    cache = PagedKVCache(
        num_layers=2,
        num_kv_heads=1,
        head_dim=4,
        block_size=32,
        max_blocks=8,
        dtype=torch.float32,
        device="cpu",
        prefix_cache_capacity_blocks=2,
    )
    engine = DecodeEngine(cache, append_backend="torch", decode_backend="reference")
    prefix_k, prefix_v = _prefix(cache, 2)
    engine.register_prefix("shared", prefix_k, prefix_v)

    result = run_integrated_workload(engine, config, num_q_heads=2, seed=1701)

    assert result.completed_request_ids == ("miss-a", "hit-a", "miss-b")
    assert result.cancelled_request_ids == ("hit-cancel",)
    assert result.rejected_request_ids == ()
    assert result.successful_steps == 9
    assert result.aborted_steps == 1
    assert result.completed_tokens == 13
    assert result.block_reuse_count > 0
    assert result.terminal_resident_prefix_blocks == 2
    assert result.final_free_blocks == cache.max_blocks
    assert result.trajectory_digest == result.reference.digest
    assert result.engine_metrics["cache"]["used_blocks"] == 0
    assert result.engine_metrics["cache"]["open_transaction_count"] == 0
    assert engine.validate_invariants()


def test_engine_prefix_cleanup_requires_resolved_lifecycles():
    cache = PagedKVCache(
        num_layers=2,
        num_kv_heads=1,
        head_dim=4,
        block_size=2,
        max_blocks=2,
        dtype=torch.float32,
        device="cpu",
        prefix_cache_capacity_blocks=1,
    )
    engine = DecodeEngine(cache)
    prefix_k, prefix_v = _prefix(cache, 1)
    engine.register_prefix("shared", prefix_k, prefix_v)
    engine.submit_request(RequestSpec("waiting", 2, 1, 0, "shared"))
    with pytest.raises(RuntimeError, match="after all requests are resolved"):
        engine.evict_prefix("shared")


@pytest.mark.skipif(not HAS_CUDA_TOOLCHAIN, reason=CUDA_TOOLCHAIN_REASON)
def test_integrated_fused_triton_trace_matches_reference_on_cuda():
    pytest.importorskip("triton")
    config = standard_integrated_config(num_layers=2, context_tokens=32)
    cache = PagedKVCache(
        num_layers=2,
        num_kv_heads=2,
        head_dim=64,
        block_size=32,
        max_blocks=6,
        dtype=torch.float16,
        device="cuda",
        prefix_cache_capacity_blocks=1,
    )
    engine = DecodeEngine(
        cache,
        append_backend="fused_cuda",
        decode_backend="triton",
        num_warps=2,
    )
    prefix_k, prefix_v = _prefix(cache, 1)
    engine.register_prefix("shared", prefix_k, prefix_v)

    result = run_integrated_workload(engine, config, num_q_heads=4, seed=1711)
    torch.cuda.synchronize()

    assert result.trajectory_digest == result.reference.digest
    assert result.aborted_steps == 1
    assert result.block_reuse_count > 0
    assert result.final_free_blocks == cache.max_blocks
    assert engine.validate_invariants()
