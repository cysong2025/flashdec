import pytest

from flashdec.perf import dtype_nbytes, estimate_paged_decode_bytes, paged_decode_metric_metadata


def test_dtype_nbytes_accepts_names_and_torch_style_strings():
    assert dtype_nbytes("float16") == 2
    assert dtype_nbytes("torch.bfloat16") == 2
    assert dtype_nbytes("float32") == 4
    assert dtype_nbytes("int64") == 8


def test_estimate_paged_decode_bytes_matches_current_kernel_model():
    estimate = estimate_paged_decode_bytes(
        num_seqs=2,
        num_q_heads=4,
        head_dim=8,
        seq_lens=[3, 5],
        max_blocks_per_seq=2,
        block_size=4,
        dtype="float16",
    )

    assert estimate["decode_tokens"] == 2
    assert estimate["head_outputs"] == 8
    assert estimate["total_context_tokens"] == 8
    assert estimate["estimated_used_blocks"] == 3
    assert estimate["estimated_q_read_bytes"] == 128
    assert estimate["estimated_out_write_bytes"] == 128
    assert estimate["estimated_kv_read_bytes"] == 1024
    assert estimate["estimated_block_table_read_bytes"] == 64
    assert estimate["estimated_seq_lens_read_bytes"] == 32
    assert estimate["estimated_total_bytes"] == 1376


def test_paged_decode_metric_metadata_formats_derived_metrics():
    estimate = estimate_paged_decode_bytes(
        num_seqs=2,
        num_q_heads=4,
        head_dim=8,
        seq_lens=[3, 5],
        max_blocks_per_seq=2,
        block_size=4,
        dtype="float16",
    )

    metadata = paged_decode_metric_metadata(estimate, mean_ms=2.0, p50_ms=1.0)

    assert metadata["decode_tokens_per_s_mean"] == "1000.00"
    assert metadata["head_outputs_per_s_mean"] == "4000.00"
    assert metadata["effective_kv_gbps_mean"] == "0.0005"
    assert metadata["effective_total_gbps_mean"] == "0.0007"
    assert metadata["effective_kv_gbps_p50"] == "0.0010"
    assert metadata["effective_total_gbps_p50"] == "0.0014"


def test_estimate_paged_decode_bytes_rejects_bad_seq_lens():
    with pytest.raises(ValueError, match="one value per sequence"):
        estimate_paged_decode_bytes(
            num_seqs=2,
            num_q_heads=4,
            head_dim=8,
            seq_lens=[3],
            max_blocks_per_seq=2,
            block_size=4,
            dtype="float16",
        )
