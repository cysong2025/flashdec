#include <cstdint>

#include <ATen/Dispatch.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <torch/extension.h>

namespace {

template <typename scalar_t>
__global__ void kv_append_kernel(
    scalar_t* k_cache,
    scalar_t* v_cache,
    const int64_t* block_ids,
    const int64_t* block_offsets,
    const scalar_t* k,
    const scalar_t* v,
    int64_t batch_size,
    int64_t num_kv_heads,
    int64_t block_size,
    int64_t head_dim) {
  const int64_t linear_idx =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t values_per_request = num_kv_heads * head_dim;
  const int64_t total_values = batch_size * values_per_request;
  if (linear_idx >= total_values) {
    return;
  }

  const int64_t request_idx = linear_idx / values_per_request;
  const int64_t head_dim_idx = linear_idx % head_dim;
  const int64_t head_idx = (linear_idx / head_dim) % num_kv_heads;
  const int64_t physical_block = block_ids[request_idx];
  const int64_t block_offset = block_offsets[request_idx];
  const int64_t cache_idx =
      (((physical_block * num_kv_heads + head_idx) * block_size + block_offset) * head_dim) +
      head_dim_idx;

  k_cache[cache_idx] = k[linear_idx];
  v_cache[cache_idx] = v[linear_idx];
}

}  // namespace

void kv_append_cuda(
    torch::Tensor k_cache,
    torch::Tensor v_cache,
    torch::Tensor block_ids,
    torch::Tensor block_offsets,
    torch::Tensor k,
    torch::Tensor v) {
  const int64_t batch_size = k.size(0);
  const int64_t num_kv_heads = k.size(1);
  const int64_t block_size = k_cache.size(2);
  const int64_t head_dim = k.size(2);
  const int64_t total_values = batch_size * num_kv_heads * head_dim;
  if (total_values == 0) {
    return;
  }

  constexpr int threads = 256;
  const int blocks = static_cast<int>((total_values + threads - 1) / threads);
  const auto stream = at::cuda::getCurrentCUDAStream();

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::kHalf,
      at::kBFloat16,
      k.scalar_type(),
      "flashdec_kv_append_cuda",
      [&] {
        kv_append_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
            k_cache.data_ptr<scalar_t>(),
            v_cache.data_ptr<scalar_t>(),
            block_ids.data_ptr<int64_t>(),
            block_offsets.data_ptr<int64_t>(),
            k.data_ptr<scalar_t>(),
            v.data_ptr<scalar_t>(),
            batch_size,
            num_kv_heads,
            block_size,
            head_dim);
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
