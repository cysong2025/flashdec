"""FlashDec: a single-GPU LLM decode runtime research prototype."""

__all__ = [
    "PagedKVCache",
    "KVTokenTransactionView",
    "AdmissionResult",
    "DecodeLayerResult",
    "DecodeEngine",
    "DecodeStepTransaction",
    "DecodeStepResult",
    "ActiveRequestMetadata",
    "BlockAwareScheduler",
    "RequestSpec",
    "SchedulerConfig",
    "SchedulerDecision",
    "SchedulingSnapshot",
    "WaitingRequestMetadata",
    "CANCEL_ON_BACKPRESSURE",
    "GREEDY_STEP_ONLY",
    "LIFETIME_FIFO_AGING",
    "RequestArrival",
    "SchedulerWorkloadConfig",
    "SchedulerWorkloadResult",
    "RopeAppendResult",
    "WorkloadConfig",
    "WorkloadResult",
    "__version__",
    "apply_rope",
    "cuda_kv_append",
    "decode",
    "dense_decode_attention_ref",
    "fused_rope_kv_append",
    "load_cuda_kv_append_extension",
    "load_fused_rope_kv_append_extension",
    "paged_decode_attention",
    "paged_decode_attention_ref",
    "rope_paged_kv_append",
    "rope_paged_kv_append_ref",
    "run_synthetic_workload",
    "run_scheduler_workload",
    "boundary_deadlock_arrivals",
]

__version__ = "0.0.0"


def __getattr__(name):
    if name in (
        "AdmissionResult",
        "DecodeEngine",
        "DecodeLayerResult",
        "DecodeStepResult",
        "DecodeStepTransaction",
    ):
        from .engine import (
            AdmissionResult,
            DecodeEngine,
            DecodeLayerResult,
            DecodeStepResult,
            DecodeStepTransaction,
        )

        return {
            "AdmissionResult": AdmissionResult,
            "DecodeEngine": DecodeEngine,
            "DecodeLayerResult": DecodeLayerResult,
            "DecodeStepResult": DecodeStepResult,
            "DecodeStepTransaction": DecodeStepTransaction,
        }[name]
    if name in ("KVTokenTransactionView", "PagedKVCache"):
        from .cache import KVTokenTransactionView, PagedKVCache

        return {
            "KVTokenTransactionView": KVTokenTransactionView,
            "PagedKVCache": PagedKVCache,
        }[name]
    if name in (
        "ActiveRequestMetadata",
        "BlockAwareScheduler",
        "RequestSpec",
        "SchedulerConfig",
        "SchedulerDecision",
        "SchedulingSnapshot",
        "WaitingRequestMetadata",
    ):
        from .scheduler import (
            ActiveRequestMetadata,
            BlockAwareScheduler,
            RequestSpec,
            SchedulerConfig,
            SchedulerDecision,
            SchedulingSnapshot,
            WaitingRequestMetadata,
        )

        return {
            "ActiveRequestMetadata": ActiveRequestMetadata,
            "BlockAwareScheduler": BlockAwareScheduler,
            "RequestSpec": RequestSpec,
            "SchedulerConfig": SchedulerConfig,
            "SchedulerDecision": SchedulerDecision,
            "SchedulingSnapshot": SchedulingSnapshot,
            "WaitingRequestMetadata": WaitingRequestMetadata,
        }[name]
    if name in ("WorkloadConfig", "WorkloadResult", "run_synthetic_workload"):
        from .workload import WorkloadConfig, WorkloadResult, run_synthetic_workload

        return {
            "WorkloadConfig": WorkloadConfig,
            "WorkloadResult": WorkloadResult,
            "run_synthetic_workload": run_synthetic_workload,
        }[name]
    if name in (
        "RequestArrival",
        "SchedulerWorkloadConfig",
        "SchedulerWorkloadResult",
        "CANCEL_ON_BACKPRESSURE",
        "GREEDY_STEP_ONLY",
        "LIFETIME_FIFO_AGING",
        "boundary_deadlock_arrivals",
        "run_scheduler_workload",
    ):
        from .scheduled_workload import (
            CANCEL_ON_BACKPRESSURE,
            GREEDY_STEP_ONLY,
            LIFETIME_FIFO_AGING,
            RequestArrival,
            SchedulerWorkloadConfig,
            SchedulerWorkloadResult,
            boundary_deadlock_arrivals,
            run_scheduler_workload,
        )

        return {
            "CANCEL_ON_BACKPRESSURE": CANCEL_ON_BACKPRESSURE,
            "GREEDY_STEP_ONLY": GREEDY_STEP_ONLY,
            "LIFETIME_FIFO_AGING": LIFETIME_FIFO_AGING,
            "RequestArrival": RequestArrival,
            "SchedulerWorkloadConfig": SchedulerWorkloadConfig,
            "SchedulerWorkloadResult": SchedulerWorkloadResult,
            "boundary_deadlock_arrivals": boundary_deadlock_arrivals,
            "run_scheduler_workload": run_scheduler_workload,
        }[name]
    if name in (
        "RopeAppendResult",
        "apply_rope",
        "rope_paged_kv_append",
        "rope_paged_kv_append_ref",
    ):
        from .rope import (
            RopeAppendResult,
            apply_rope,
            rope_paged_kv_append,
            rope_paged_kv_append_ref,
        )

        return {
            "RopeAppendResult": RopeAppendResult,
            "apply_rope": apply_rope,
            "rope_paged_kv_append": rope_paged_kv_append,
            "rope_paged_kv_append_ref": rope_paged_kv_append_ref,
        }[name]
    if name in ("cuda_kv_append", "load_cuda_kv_append_extension"):
        from ._cuda_kv_append import cuda_kv_append, load_cuda_kv_append_extension

        return {
            "cuda_kv_append": cuda_kv_append,
            "load_cuda_kv_append_extension": load_cuda_kv_append_extension,
        }[name]
    if name in ("fused_rope_kv_append", "load_fused_rope_kv_append_extension"):
        from ._fused_rope_kv_append import (
            fused_rope_kv_append,
            load_fused_rope_kv_append_extension,
        )

        return {
            "fused_rope_kv_append": fused_rope_kv_append,
            "load_fused_rope_kv_append_extension": load_fused_rope_kv_append_extension,
        }[name]
    if name == "dense_decode_attention_ref":
        from .reference import dense_decode_attention_ref

        return dense_decode_attention_ref
    if name in ("decode", "paged_decode_attention"):
        from .kernels.paged_decode import paged_decode_attention

        return paged_decode_attention
    if name == "paged_decode_attention_ref":
        from .paged_reference import paged_decode_attention_ref

        return paged_decode_attention_ref
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
