#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdio>
#include <vector>

#ifndef SOL_WMMA_WAVES_PER_WGP
#define SOL_WMMA_WAVES_PER_WGP 64
#endif

using packed_fp8x8 = int __attribute__((ext_vector_type(2)));
using float8v = float __attribute__((ext_vector_type(8)));

constexpr int matrix_m = 16;
constexpr int matrix_n = 16;
constexpr int matrix_k = 16;
constexpr int repetitions = 1024;
constexpr int lane_count = 32;
constexpr int values_per_lane = 8;
constexpr float validation_tolerance = 1.0e-3f;

__global__ void matrix_probe_kernel(float* out) {
  // OCP E4M3 1.0f is 0x38. Each source operand contains eight packed values.
  packed_fp8x8 a = {0x38383838, 0x38383838};
  packed_fp8x8 b = {0x38383838, 0x38383838};
  float8v accumulator = {};
  for (int repeat = 0; repeat < repetitions; ++repeat) {
    accumulator = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(
        a, b, accumulator);
  }
  const int lane = threadIdx.x;
  const size_t block_offset =
      static_cast<size_t>(blockIdx.x) * matrix_m * matrix_n;
  for (int index = 0; index < values_per_lane; ++index) {
    out[block_offset + lane * values_per_lane + index] = accumulator[index];
  }
}

int main() {
  hipDeviceProp_t properties{};
  if (hipGetDeviceProperties(&properties, 0) != hipSuccess ||
      properties.multiProcessorCount <= 0)
    return 2;
  // gfx12 reports WGPs through multiProcessorCount (one WGP contains two CUs).
  // Queue enough one-wave blocks per reported WGP to hide the loop-carried
  // matrix latency and saturate all CU schedulers.
  const int waves_per_reported_wgp = SOL_WMMA_WAVES_PER_WGP;
  if (waves_per_reported_wgp <= 0)
    return 2;
  const int grid_blocks =
      properties.multiProcessorCount * waves_per_reported_wgp;
  const size_t output_count =
      static_cast<size_t>(grid_blocks) * matrix_m * matrix_n;
  std::vector<float> output(output_count);
  float* device_output = nullptr;
  if (hipMalloc(&device_output, output_count * sizeof(float)) != hipSuccess)
    return 2;
  hipEvent_t start = nullptr, stop = nullptr;
  if (hipEventCreate(&start) != hipSuccess ||
      hipEventCreate(&stop) != hipSuccess)
    return 2;

  hipLaunchKernelGGL(matrix_probe_kernel, dim3(grid_blocks), dim3(lane_count), 0,
                     0, device_output);
  if (hipDeviceSynchronize() != hipSuccess)
    return 2;
  if (hipMemcpy(output.data(), device_output, output_count * sizeof(float),
                hipMemcpyDeviceToHost) != hipSuccess)
    return 2;
  const float cpu_reference = static_cast<float>(matrix_k * repetitions);
  for (float value : output) {
    if (std::fabs(value - cpu_reference) > validation_tolerance)
      return 4;
  }

  double samples[7] = {};
  for (int sample = 0; sample < 7; ++sample) {
    if (hipEventRecord(start) != hipSuccess)
      return 2;
    hipLaunchKernelGGL(matrix_probe_kernel, dim3(grid_blocks),
                       dim3(lane_count), 0, 0, device_output);
    if (hipEventRecord(stop) != hipSuccess ||
        hipEventSynchronize(stop) != hipSuccess)
      return 2;
    float elapsed_ms = 0.0f;
    if (hipEventElapsedTime(&elapsed_ms, start, stop) != hipSuccess ||
        elapsed_ms <= 0.0f)
      return 3;
    const double elapsed_seconds = static_cast<double>(elapsed_ms) / 1.0e3;
    samples[sample] =
        static_cast<double>(grid_blocks) * 2.0 * matrix_m * matrix_n * matrix_k *
        repetitions / elapsed_seconds / 1.0e12;
  }

  for (double sample : samples)
    std::printf("RESULT %.9f\n", sample);
  hipEventDestroy(start);
  hipEventDestroy(stop);
  hipFree(device_output);
  return 0;
}
