#include <cstdint>

#include <ATen/Dispatch.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <torch/extension.h>

namespace {

template <typename scalar_t>
__device__ scalar_t apply_split_half_rope(
    const scalar_t* values,
    int64_t linear_idx,
    int64_t head_dim,
    int64_t rotary_dim,
    int64_t position,
    float base) {
  const int64_t dim_idx = linear_idx % head_dim;
  if (dim_idx >= rotary_dim) {
    return values[linear_idx];
  }

  const int64_t half_dim = rotary_dim / 2;
  const int64_t pair_dim = dim_idx < half_dim ? dim_idx + half_dim : dim_idx - half_dim;
  const int64_t row_start = linear_idx - dim_idx;
  const int64_t freq_idx = dim_idx < half_dim ? dim_idx : dim_idx - half_dim;
  const float inv_freq = powf(base, -2.0f * static_cast<float>(freq_idx) / static_cast<float>(rotary_dim));
  const float angle = static_cast<float>(position) * inv_freq;
  const float value = static_cast<float>(values[linear_idx]);
  const float pair_value = static_cast<float>(values[row_start + pair_dim]);
  const float cosine = cosf(angle);
  const float sine = sinf(angle);
  const float rotated = dim_idx < half_dim ? value * cosine - pair_value * sine
                                            : value * cosine + pair_value * sine;
  return static_cast<scalar_t>(rotated);
}

template <typename scalar_t>
__global__ void fused_rope_kv_append_kernel(
    scalar_t* q_out,
    const scalar_t* q,
    const scalar_t* k,
    const scalar_t* v,
    scalar_t* k_cache,
    scalar_t* v_cache,
    const int64_t* block_ids,
    const int64_t* block_offsets,
    const int64_t* positions,
    int64_t batch_size,
    int64_t num_q_heads,
    int64_t num_kv_heads,
    int64_t block_size,
    int64_t head_dim,
    int64_t rotary_dim,
    float base) {
  const int64_t linear_idx =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t q_values_per_request = num_q_heads * head_dim;
  const int64_t kv_values_per_request = num_kv_heads * head_dim;
  const int64_t total_q_values = batch_size * q_values_per_request;
  const int64_t total_kv_values = batch_size * kv_values_per_request;

  if (linear_idx < total_q_values) {
    const int64_t request_idx = linear_idx / q_values_per_request;
    q_out[linear_idx] = apply_split_half_rope(
        q, linear_idx, head_dim, rotary_dim, positions[request_idx], base);
  }

  if (linear_idx < total_kv_values) {
    const int64_t request_idx = linear_idx / kv_values_per_request;
    const int64_t dim_idx = linear_idx % head_dim;
    const int64_t head_idx = (linear_idx / head_dim) % num_kv_heads;
    const int64_t physical_block = block_ids[request_idx];
    const int64_t block_offset = block_offsets[request_idx];
    const int64_t cache_idx =
        (((physical_block * num_kv_heads + head_idx) * block_size + block_offset) * head_dim) +
        dim_idx;
    k_cache[cache_idx] = apply_split_half_rope(
        k, linear_idx, head_dim, rotary_dim, positions[request_idx], base);
    v_cache[cache_idx] = v[linear_idx];
  }
}

}  // namespace

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
    double base) {
  const int64_t batch_size = q.size(0);
  const int64_t num_q_heads = q.size(1);
  const int64_t num_kv_heads = k.size(1);
  const int64_t block_size = k_cache.size(2);
  const int64_t head_dim = q.size(2);
  const int64_t total_q_values = batch_size * num_q_heads * head_dim;
  const int64_t total_kv_values = batch_size * num_kv_heads * head_dim;
  const int64_t total_values = total_q_values > total_kv_values ? total_q_values : total_kv_values;
  if (total_values == 0) {
    return;
  }

  constexpr int threads = 256;
  const int blocks = static_cast<int>((total_values + threads - 1) / threads);
  const auto stream = at::cuda::getCurrentCUDAStream();

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::kHalf,
      at::kBFloat16,
      q.scalar_type(),
      "flashdec_fused_rope_kv_append_cuda",
      [&] {
        fused_rope_kv_append_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
            q_out.data_ptr<scalar_t>(),
            q.data_ptr<scalar_t>(),
            k.data_ptr<scalar_t>(),
            v.data_ptr<scalar_t>(),
            k_cache.data_ptr<scalar_t>(),
            v_cache.data_ptr<scalar_t>(),
            block_ids.data_ptr<int64_t>(),
            block_offsets.data_ptr<int64_t>(),
            positions.data_ptr<int64_t>(),
            batch_size,
            num_q_heads,
            num_kv_heads,
            block_size,
            head_dim,
            rotary_dim,
            static_cast<float>(base));
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
