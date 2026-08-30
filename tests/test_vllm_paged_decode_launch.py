import importlib.util
import sys
import unittest
from pathlib import Path
from types import ModuleType
from unittest.mock import patch


PAGED_DECODE_PATH = (
    Path(__file__).resolve().parents[1] / "flashdec" / "kernels" / "paged_decode.py"
)


class _FakeTensor:
    def __init__(self, shape):
        self.shape = tuple(shape)
        strides = []
        stride = 1
        for size in reversed(self.shape):
            strides.append(stride)
            stride *= size
        self._strides = tuple(reversed(strides))

    def stride(self, dim=None):
        if dim is None:
            return self._strides
        return self._strides[dim]


class _KernelRecorder:
    def __init__(self):
        self.calls = []

    def __getitem__(self, grid):
        def launch(*args, **kwargs):
            self.calls.append((grid, args, kwargs))

        return launch


def _load_paged_decode_without_gpu_dependencies():
    fake_torch = ModuleType("torch")
    fake_triton = ModuleType("triton")
    fake_tl = ModuleType("triton.language")
    fake_triton.jit = lambda function: function
    fake_triton.language = fake_tl

    fake_modules = {
        "torch": fake_torch,
        "triton": fake_triton,
        "triton.language": fake_tl,
    }
    with patch.dict(sys.modules, fake_modules):
        spec = importlib.util.spec_from_file_location(
            "_flashdec_paged_decode_launch_test",
            PAGED_DECODE_PATH,
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    return module


def _run_split_launcher(
    module,
    *,
    num_q_heads=16,
    num_kv_heads=2,
    head_dim=128,
    block_size=16,
):
    split_kernel = _KernelRecorder()
    reduce_kernel = _KernelRecorder()
    module._paged_decode_gqa_split_kernel = split_kernel
    module._paged_decode_gqa_reduce_kernel = reduce_kernel

    num_reqs = 3
    num_splits = 8
    q = _FakeTensor((num_reqs, num_q_heads, head_dim))
    append_k = _FakeTensor((num_reqs, num_kv_heads, head_dim))
    append_v = _FakeTensor((num_reqs, num_kv_heads, head_dim))
    kv_cache = _FakeTensor((128, 2, block_size, num_kv_heads, head_dim))
    block_tables = _FakeTensor((num_reqs, 64))
    seq_lens = _FakeTensor((num_reqs,))
    out = _FakeTensor((num_reqs, num_q_heads, head_dim))
    slot_mapping = _FakeTensor((num_reqs,))
    split_acc = _FakeTensor((num_reqs, num_q_heads, 16, head_dim))
    split_max = _FakeTensor((num_reqs, num_q_heads, 16))
    split_sum = _FakeTensor((num_reqs, num_q_heads, 16))

    returned = module._vllm_paged_decode_attention_into(
        q,
        append_k,
        append_v,
        kv_cache,
        block_tables,
        seq_lens,
        out,
        slot_mapping,
        split_acc,
        split_max,
        split_sum,
        num_reqs=num_reqs,
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        block_size=block_size,
        sm_scale=head_dim**-0.5,
        num_splits=num_splits,
    )

    assert returned is out
    assert len(split_kernel.calls) == 1
    assert len(reduce_kernel.calls) == 1
    return split_kernel.calls[0], reduce_kernel.calls[0]


class VllmPagedDecodeLaunchTests(unittest.TestCase):
    def test_exact_qwen_vllm_split_uses_measured_group_tile(self):
        module = _load_paged_decode_without_gpu_dependencies()

        split_call, reduce_call = _run_split_launcher(module)

        split_grid, _, split_options = split_call
        self.assertEqual(split_grid, (3, 2, 8))
        self.assertEqual(split_options["GROUP_SIZE"], 8)
        self.assertEqual(split_options["GROUP_BLOCK"], 8)
        self.assertEqual(split_options["TOKEN_BLOCK"], 32)
        self.assertEqual(split_options["num_warps"], 4)
        self.assertEqual(split_options["num_stages"], 3)

        reduce_grid, _, reduce_options = reduce_call
        self.assertEqual(reduce_grid, (3, 16))
        self.assertEqual(reduce_options["GROUP_SIZE"], 1)
        self.assertEqual(reduce_options["GROUP_BLOCK"], 1)
        self.assertEqual(reduce_options["num_warps"], 1)

    def test_other_internal_vllm_shapes_keep_general_split_launch(self):
        cases = [
            (32, 4, 128, 16),
            (16, 1, 128, 16),
            (16, 4, 128, 16),
            (16, 2, 64, 16),
            (16, 2, 128, 32),
        ]
        for num_q_heads, num_kv_heads, head_dim, block_size in cases:
            with self.subTest(
                num_q_heads=num_q_heads,
                num_kv_heads=num_kv_heads,
                head_dim=head_dim,
                block_size=block_size,
            ):
                module = _load_paged_decode_without_gpu_dependencies()
                split_call, _ = _run_split_launcher(
                    module,
                    num_q_heads=num_q_heads,
                    num_kv_heads=num_kv_heads,
                    head_dim=head_dim,
                    block_size=block_size,
                )

                _, _, split_options = split_call
                self.assertEqual(split_options["GROUP_BLOCK"], 16)
                self.assertEqual(split_options["num_warps"], 4)
                self.assertNotIn("num_stages", split_options)


if __name__ == "__main__":
    unittest.main()
