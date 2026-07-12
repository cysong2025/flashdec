#include <torch/extension.h>

void kv_append_cuda(
    torch::Tensor k_cache,
    torch::Tensor v_cache,
    torch::Tensor block_ids,
    torch::Tensor block_offsets,
    torch::Tensor k,
    torch::Tensor v);

namespace {

void check_kv_append_inputs(
    const torch::Tensor& k_cache,
    const torch::Tensor& v_cache,
    const torch::Tensor& block_ids,
    const torch::Tensor& block_offsets,
    const torch::Tensor& k,
    const torch::Tensor& v) {
  TORCH_CHECK(k_cache.is_cuda(), "k_cache must be a CUDA tensor");
  TORCH_CHECK(v_cache.is_cuda(), "v_cache must be a CUDA tensor");
  TORCH_CHECK(block_ids.is_cuda(), "block_ids must be a CUDA tensor");
  TORCH_CHECK(block_offsets.is_cuda(), "block_offsets must be a CUDA tensor");
  TORCH_CHECK(k.is_cuda(), "k must be a CUDA tensor");
  TORCH_CHECK(v.is_cuda(), "v must be a CUDA tensor");

  TORCH_CHECK(k_cache.device() == v_cache.device(), "K/V caches must share a device");
  TORCH_CHECK(k_cache.device() == block_ids.device(), "all inputs must share a device");
  TORCH_CHECK(k_cache.device() == block_offsets.device(), "all inputs must share a device");
  TORCH_CHECK(k_cache.device() == k.device(), "all inputs must share a device");
  TORCH_CHECK(k_cache.device() == v.device(), "all inputs must share a device");

  TORCH_CHECK(k_cache.dim() == 4, "k_cache must have shape [blocks, heads, block_size, head_dim]");
  TORCH_CHECK(v_cache.sizes() == k_cache.sizes(), "v_cache must match k_cache shape");
  TORCH_CHECK(k.dim() == 3, "k must have shape [batch, heads, head_dim]");
  TORCH_CHECK(v.sizes() == k.sizes(), "v must match k shape");
  TORCH_CHECK(block_ids.dim() == 1, "block_ids must be rank 1");
  TORCH_CHECK(block_offsets.dim() == 1, "block_offsets must be rank 1");
  TORCH_CHECK(block_ids.size(0) == k.size(0), "block_ids length must match batch");
  TORCH_CHECK(block_offsets.size(0) == k.size(0), "block_offsets length must match batch");
  TORCH_CHECK(k.size(1) == k_cache.size(1), "k head count must match cache");
  TORCH_CHECK(k.size(2) == k_cache.size(3), "k head_dim must match cache");

  TORCH_CHECK(k_cache.scalar_type() == v_cache.scalar_type(), "K/V caches must share dtype");
  TORCH_CHECK(k_cache.scalar_type() == k.scalar_type(), "k dtype must match cache");
  TORCH_CHECK(k_cache.scalar_type() == v.scalar_type(), "v dtype must match cache");
  TORCH_CHECK(block_ids.scalar_type() == at::kLong, "block_ids must use int64");
  TORCH_CHECK(block_offsets.scalar_type() == at::kLong, "block_offsets must use int64");

  TORCH_CHECK(k_cache.is_contiguous(), "k_cache must be contiguous");
  TORCH_CHECK(v_cache.is_contiguous(), "v_cache must be contiguous");
  TORCH_CHECK(block_ids.is_contiguous(), "block_ids must be contiguous");
  TORCH_CHECK(block_offsets.is_contiguous(), "block_offsets must be contiguous");
  TORCH_CHECK(k.is_contiguous(), "k must be contiguous");
  TORCH_CHECK(v.is_contiguous(), "v must be contiguous");
}

}  // namespace

void kv_append(
    torch::Tensor k_cache,
    torch::Tensor v_cache,
    torch::Tensor block_ids,
    torch::Tensor block_offsets,
    torch::Tensor k,
    torch::Tensor v) {
  check_kv_append_inputs(k_cache, v_cache, block_ids, block_offsets, k, v);
  kv_append_cuda(k_cache, v_cache, block_ids, block_offsets, k, v);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("kv_append", &kv_append, "Append one K/V token per request into paged CUDA cache");
}
