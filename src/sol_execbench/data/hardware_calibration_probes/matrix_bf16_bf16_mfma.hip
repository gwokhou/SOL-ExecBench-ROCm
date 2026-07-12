#include <hip/hip_runtime.h>
#include <hip/hip_bfloat16.h>
#include <cmath>
#include <cstdio>
#include <vector>

// HIP provides the __hip_bfloat16 type used by this matrix probe.
using bf16x4 = unsigned short __attribute__((ext_vector_type(4)));
using float4v = float __attribute__((ext_vector_type(4)));

constexpr int matrix_m = 16;
constexpr int matrix_n = 16;
constexpr int matrix_k = 16;
constexpr int repetitions = 1024;
constexpr int lane_count = 64;
constexpr int values_per_lane = 4;
constexpr float validation_tolerance = 1.0e-3f;

__global__ void matrix_probe_kernel(float* out) {
  bf16x4 a = {0x3f80, 0x3f80, 0x3f80, 0x3f80};
  bf16x4 b = {0x3f80, 0x3f80, 0x3f80, 0x3f80};
  float4v accumulator = {};
  for (int repeat = 0; repeat < repetitions; ++repeat) {
    accumulator = __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(a, b, accumulator, 0, 0, 0);
  }
  const int lane = threadIdx.x;
  const size_t block_offset = static_cast<size_t>(blockIdx.x) * matrix_m * matrix_n;
  for (int index = 0; index < values_per_lane; ++index) {
    out[block_offset + lane * values_per_lane + index] = accumulator[index];
  }
}

int main() {
  hipDeviceProp_t properties{};
  if (hipGetDeviceProperties(&properties, 0) != hipSuccess || properties.multiProcessorCount <= 0) return 2;
  // One wave cannot measure device throughput.  Launch several independent
  // waves per CU so the scheduler has enough work to hide matrix-instruction
  // latency and the FLOP accounting covers every launched block.
  const int blocks_per_cu = 8;
  const int grid_blocks = properties.multiProcessorCount * blocks_per_cu;
  const size_t output_count = static_cast<size_t>(grid_blocks) * matrix_m * matrix_n;
  std::vector<float> output(output_count);
  float* device_output = nullptr;
  if (hipMalloc(&device_output, output_count * sizeof(float)) != hipSuccess) return 2;
  hipEvent_t start = nullptr, stop = nullptr;
  if (hipEventCreate(&start) != hipSuccess || hipEventCreate(&stop) != hipSuccess) return 2;

  hipLaunchKernelGGL(matrix_probe_kernel, dim3(grid_blocks), dim3(lane_count), 0, 0, device_output);
  if (hipDeviceSynchronize() != hipSuccess) return 2;
  if (hipMemcpy(output.data(), device_output, output_count * sizeof(float), hipMemcpyDeviceToHost) != hipSuccess) return 2;
  const float cpu_reference = static_cast<float>(matrix_k * repetitions);
  for (float value : output) {
    if (std::fabs(value - cpu_reference) > validation_tolerance) return 4;
  }

  double samples[7] = {};
  for (int sample = 0; sample < 7; ++sample) {
    if (hipEventRecord(start) != hipSuccess) return 2;
    hipLaunchKernelGGL(matrix_probe_kernel, dim3(grid_blocks), dim3(lane_count), 0, 0, device_output);
    if (hipEventRecord(stop) != hipSuccess || hipEventSynchronize(stop) != hipSuccess) return 2;
    float elapsed_ms = 0.0f;
    if (hipEventElapsedTime(&elapsed_ms, start, stop) != hipSuccess || elapsed_ms <= 0.0f) return 3;
    const double elapsed_seconds = static_cast<double>(elapsed_ms) / 1.0e3;
    samples[sample] = static_cast<double>(grid_blocks) * 2.0 * matrix_m * matrix_n * matrix_k * repetitions / elapsed_seconds / 1.0e12;
  }

  for (double sample : samples) std::printf("RESULT %.9f\n", sample);
  hipEventDestroy(start); hipEventDestroy(stop); hipFree(device_output);
  return 0;
}
