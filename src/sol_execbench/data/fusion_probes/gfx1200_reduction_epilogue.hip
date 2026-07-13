// SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
// SPDX-License-Identifier: Apache-2.0

// Canonical FP32 reduction_epilogue.v1 kernels used by the external probe
// driver.  The driver owns allocation, deterministic initialization, HIP-event
// timing, reference comparison, and occupancy queries; keeping kernels here
// makes the source identity included in fusion_validation.v1 reproducible.

#include <hip/hip_runtime.h>

namespace {
constexpr int kBlockSize = 256;

__device__ float block_sum(float value, float* shared) {
  const int lane = threadIdx.x;
  shared[lane] = value;
  __syncthreads();
  for (int offset = blockDim.x / 2; offset != 0; offset >>= 1) {
    if (lane < offset) shared[lane] += shared[lane + offset];
    __syncthreads();
  }
  return shared[0];
}
}  // namespace

// RMSNorm capacity slice: FP32 mean(x_squared, dim=-1) followed by + epsilon.
extern "C" __global__ __launch_bounds__(kBlockSize)
void sol_fusion_rms_mean_epsilon(const float* input, float* output,
                                 int rows, int hidden, float epsilon) {
  __shared__ float partial[kBlockSize];
  const int row = blockIdx.x;
  if (row >= rows) return;
  float value = 0.0f;
  for (int column = threadIdx.x; column < hidden; column += blockDim.x) {
    value += input[row * hidden + column];
  }
  const float sum = block_sum(value, partial);
  if (threadIdx.x == 0) output[row] = sum / static_cast<float>(hidden) + epsilon;
}

extern "C" __global__ __launch_bounds__(kBlockSize)
void sol_unfused_rms_mean(const float* input, float* output,
                          int rows, int hidden) {
  __shared__ float partial[kBlockSize];
  const int row = blockIdx.x;
  if (row >= rows) return;
  float value = 0.0f;
  for (int column = threadIdx.x; column < hidden; column += blockDim.x) {
    value += input[row * hidden + column];
  }
  const float sum = block_sum(value, partial);
  if (threadIdx.x == 0) output[row] = sum / static_cast<float>(hidden);
}

extern "C" __global__
void sol_unfused_rms_epsilon(float* values, int count, float epsilon) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < count) values[index] += epsilon;
}

// Tanh backward capacity slice: full FP32 sum(product) followed by a scalar.
// Each workgroup contributes a partial sum.  Atomic accumulation permits a
// workload-sized launch; the probe compares with the workload's 5% tolerance.
extern "C" __global__ __launch_bounds__(kBlockSize)
void sol_fusion_tanh_backward_sum_scalar(const float* input, float* output,
                                         long count, float scalar) {
  __shared__ float partial[kBlockSize];
  float value = 0.0f;
  for (long index = blockIdx.x * blockDim.x + threadIdx.x;
       index < count;
       index += blockDim.x * gridDim.x) {
    value += input[index];
  }
  const float sum = block_sum(value, partial);
  if (threadIdx.x == 0) atomicAdd(output, sum * scalar);
}

extern "C" __global__ __launch_bounds__(kBlockSize)
void sol_unfused_tanh_backward_sum(const float* input, float* output,
                                   long count) {
  __shared__ float partial[kBlockSize];
  float value = 0.0f;
  for (long index = blockIdx.x * blockDim.x + threadIdx.x;
       index < count;
       index += blockDim.x * gridDim.x) {
    value += input[index];
  }
  const float sum = block_sum(value, partial);
  if (threadIdx.x == 0) atomicAdd(output, sum);
}

extern "C" __global__
void sol_unfused_tanh_backward_scalar(float* output, float scalar) {
  if (blockIdx.x == 0 && threadIdx.x == 0) output[0] *= scalar;
}
