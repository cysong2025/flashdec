#include <torch/extension.h>

void fused_rope_kv_append_cuda(
    torch::Tensor q_out,
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor k_cache,
    torch::Tensor v_cache,
    torch::Tensor block_ids,
    torch::Tensor block_offsets,
    torch::Tensor positions,
    int64_t rotary_dim,
    double base);

namespace {

void check_fused_rope_kv_append_inputs(
    const torch::Tensor& q_out,
    const torch::Tensor& q,
    const torch::Tensor& k,
    const torch::Tensor& v,
    const torch::Tensor& k_cache,
    const torch::Tensor& v_cache,
    const torch::Tensor& block_ids,
    const torch::Tensor& block_offsets,
    const torch::Tensor& positions,
    int64_t rotary_dim) {
  TORCH_CHECK(q_out.is_cuda(), "q_out must be a CUDA tensor");
  TORCH_CHECK(q.is_cuda(), "q must be a CUDA tensor");
  TORCH_CHECK(k.is_cuda(), "k must be a CUDA tensor");
  TORCH_CHECK(v.is_cuda(), "v must be a CUDA tensor");
  TORCH_CHECK(k_cache.is_cuda(), "k_cache must be a CUDA tensor");
  TORCH_CHECK(v_cache.is_cuda(), "v_cache must be a CUDA tensor");
  TORCH_CHECK(block_ids.is_cuda(), "block_ids must be a CUDA tensor");
  TORCH_CHECK(block_offsets.is_cuda(), "block_offsets must be a CUDA tensor");
  TORCH_CHECK(positions.is_cuda(), "positions must be a CUDA tensor");

  TORCH_CHECK(q_out.device() == q.device(), "all inputs must share a device");
  TORCH_CHECK(q.device() == k.device(), "all inputs must share a device");
  TORCH_CHECK(q.device() == v.device(), "all inputs must share a device");
  TORCH_CHECK(q.device() == k_cache.device(), "all inputs must share a device");
  TORCH_CHECK(q.device() == v_cache.device(), "all inputs must share a device");
  TORCH_CHECK(q.device() == block_ids.device(), "all inputs must share a device");
  TORCH_CHECK(q.device() == block_offsets.device(), "all inputs must share a device");
  TORCH_CHECK(q.device() == positions.device(), "all inputs must share a device");

  TORCH_CHECK(q.dim() == 3, "q must have shape [batch, q_heads, head_dim]");
  TORCH_CHECK(q_out.sizes() == q.sizes(), "q_out must match q shape");
  TORCH_CHECK(k.dim() == 3, "k must have shape [batch, kv_heads, head_dim]");
  TORCH_CHECK(v.sizes() == k.sizes(), "v must match k shape");
  TORCH_CHECK(k_cache.dim() == 4, "k_cache must have shape [blocks, kv_heads, block_size, head_dim]");
  TORCH_CHECK(v_cache.sizes() == k_cache.sizes(), "v_cache must match k_cache shape");
  TORCH_CHECK(q.size(0) == k.size(0), "q and k batch must match");
  TORCH_CHECK(q.size(2) == k.size(2), "q and k head_dim must match");
  TORCH_CHECK(q.size(1) > 0 && k.size(1) > 0, "q and k must have at least one head");
  TORCH_CHECK(k.size(1) == k_cache.size(1), "k heads must match cache");
  TORCH_CHECK(k.size(2) == k_cache.size(3), "k head_dim must match cache");
  TORCH_CHECK(block_ids.dim() == 1, "block_ids must be rank 1");
  TORCH_CHECK(block_offsets.dim() == 1, "block_offsets must be rank 1");
  TORCH_CHECK(positions.dim() == 1, "positions must be rank 1");
  TORCH_CHECK(block_ids.size(0) == q.size(0), "block_ids length must match batch");
  TORCH_CHECK(block_offsets.size(0) == q.size(0), "block_offsets length must match batch");
  TORCH_CHECK(positions.size(0) == q.size(0), "positions length must match batch");
  TORCH_CHECK(rotary_dim > 0 && rotary_dim <= q.size(2) && rotary_dim % 2 == 0,
              "rotary_dim must be positive, even, and no larger than head_dim");

  TORCH_CHECK(q_out.scalar_type() == q.scalar_type(), "q_out dtype must match q");
  TORCH_CHECK(q.scalar_type() == k.scalar_type(), "q and k dtype must match");
  TORCH_CHECK(q.scalar_type() == v.scalar_type(), "q and v dtype must match");
  TORCH_CHECK(q.scalar_type() == k_cache.scalar_type(), "q and k_cache dtype must match");
  TORCH_CHECK(q.scalar_type() == v_cache.scalar_type(), "q and v_cache dtype must match");
  TORCH_CHECK(block_ids.scalar_type() == at::kLong, "block_ids must use int64");
  TORCH_CHECK(block_offsets.scalar_type() == at::kLong, "block_offsets must use int64");
  TORCH_CHECK(positions.scalar_type() == at::kLong, "positions must use int64");

  TORCH_CHECK(q_out.is_contiguous(), "q_out must be contiguous");
  TORCH_CHECK(q.is_contiguous(), "q must be contiguous");
  TORCH_CHECK(k.is_contiguous(), "k must be contiguous");
  TORCH_CHECK(v.is_contiguous(), "v must be contiguous");
  TORCH_CHECK(k_cache.is_contiguous(), "k_cache must be contiguous");
  TORCH_CHECK(v_cache.is_contiguous(), "v_cache must be contiguous");
  TORCH_CHECK(block_ids.is_contiguous(), "block_ids must be contiguous");
  TORCH_CHECK(block_offsets.is_contiguous(), "block_offsets must be contiguous");
  TORCH_CHECK(positions.is_contiguous(), "positions must be contiguous");
}

}  // namespace

void fused_rope_kv_append(
    torch::Tensor q_out,
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor k_cache,
    torch::Tensor v_cache,
    torch::Tensor block_ids,
    torch::Tensor block_offsets,
    torch::Tensor positions,
    int64_t rotary_dim,
    double base) {
  check_fused_rope_kv_append_inputs(
      q_out,
      q,
      k,
      v,
      k_cache,
      v_cache,
      block_ids,
      block_offsets,
      positions,
      rotary_dim);
  fused_rope_kv_append_cuda(
      q_out,
      q,
      k,
      v,
      k_cache,
      v_cache,
      block_ids,
      block_offsets,
      positions,
      rotary_dim,
      base);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def(
      "fused_rope_kv_append",
      &fused_rope_kv_append,
      "Fused RoPE and one-token paged K/V append");
}
