#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

namespace {

constexpr int kSamples = 7;
constexpr int kWarmups = 3;
constexpr int kThreads = 256;

bool hip_ok(hipError_t status) {
  if (status == hipSuccess) return true;
  std::fprintf(stderr, "HIP error: %s\n", hipGetErrorString(status));
  return false;
}

template <typename Launch>
double elapsed_ms(Launch launch) {
  hipEvent_t start = nullptr;
  hipEvent_t stop = nullptr;
  const auto cleanup = [&] {
    bool success = true;
    if (start != nullptr) success = hip_ok(hipEventDestroy(start)) && success;
    if (stop != nullptr) success = hip_ok(hipEventDestroy(stop)) && success;
    return success;
  };
  if (!hip_ok(hipEventCreate(&start)) || !hip_ok(hipEventCreate(&stop))) {
    (void)cleanup();
    return -1.0;
  }
  for (int index = 0; index < kWarmups; ++index) launch();
  if (!hip_ok(hipDeviceSynchronize()) || !hip_ok(hipEventRecord(start))) {
    (void)cleanup();
    return -1.0;
  }
  launch();
  if (!hip_ok(hipEventRecord(stop)) || !hip_ok(hipEventSynchronize(stop))) {
    (void)cleanup();
    return -1.0;
  }
  float milliseconds = 0.0f;
  if (!hip_ok(hipEventElapsedTime(&milliseconds, start, stop)) || !cleanup()) return -1.0;
  return static_cast<double>(milliseconds);
}

bool metric(const char* name, const char* variant, double value, const char* unit) {
  if (!std::isfinite(value) || value <= 0.0) return false;
  std::printf("METRIC %s %s %.12g %s\n", name, variant, value, unit);
  return true;
}

__global__ void empty_kernel() {}

int run_dispatch() {
  constexpr int launches = 512;
  for (int sample = 0; sample < kSamples; ++sample) {
    const double milliseconds = elapsed_ms([] {
      for (int index = 0; index < launches; ++index) {
        hipLaunchKernelGGL(empty_kernel, dim3(1), dim3(1), 0, 0);
      }
    });
    if (!metric("dispatch_floor_ms", "device", milliseconds / launches, "ms")) return 2;
  }
  return 0;
}

__global__ void copy_kernel(const float* input, float* output, size_t count, size_t stride) {
  const size_t logical = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (logical >= count) return;
  const size_t index = (logical * stride) % count;
  output[index] = input[index];
}

int measure_copy(size_t bytes, size_t stride, const char* variant) {
  const size_t count = bytes / sizeof(float);
  float* input = nullptr;
  float* output = nullptr;
  if (!hip_ok(hipMalloc(&input, bytes)) || !hip_ok(hipMalloc(&output, bytes))) return 2;
  if (!hip_ok(hipMemset(input, 1, bytes))) return 2;
  for (int sample = 0; sample < kSamples; ++sample) {
    const double milliseconds = elapsed_ms([&] {
      hipLaunchKernelGGL(copy_kernel, dim3((count + kThreads - 1) / kThreads),
                         dim3(kThreads), 0, 0, input, output, count, stride);
    });
    if (!metric("memory_byte_per_ms", variant, 2.0 * bytes / milliseconds, "byte/ms")) {
      return 2;
    }
  }
  const bool input_freed = hip_ok(hipFree(input));
  const bool output_freed = hip_ok(hipFree(output));
  return input_freed && output_freed ? 0 : 2;
}

int run_memory() {
  const size_t sizes[] = {
      32ULL << 10, 256ULL << 10, 2ULL << 20, 16ULL << 20, 64ULL << 20, 256ULL << 20};
  const char* names[] = {"32KiB", "256KiB", "2MiB", "16MiB", "64MiB", "256MiB"};
  for (int index = 0; index < 6; ++index) {
    const int result = measure_copy(sizes[index], 1, names[index]);
    if (result != 0) return result;
  }
  return 0;
}

int run_access() {
  constexpr size_t bytes = 128ULL << 20;
  for (const auto& entry :
       std::vector<std::pair<size_t, const char*>>{{1, "contiguous"}, {32, "transpose"},
                                                   {127, "stride127"}}) {
    const int result = measure_copy(bytes, entry.first, entry.second);
    if (result != 0) return result;
  }
  return 0;
}

__global__ void lds_kernel(float* output, int bank_stride, int repetitions) {
  __shared__ float values[32 * 32];
  const int lane = threadIdx.x & 31;
  float value = static_cast<float>(lane + 1);
  for (int repeat = 0; repeat < repetitions; ++repeat) {
    const int index = (lane * bank_stride) & 1023;
    values[index] = value;
    __syncthreads();
    value += values[index];
    __syncthreads();
  }
  if (lane == 0) output[blockIdx.x] = value;
}

int run_lds() {
  hipDeviceProp_t properties{};
  if (!hip_ok(hipGetDeviceProperties(&properties, 0))) return 2;
  const int blocks = std::max(properties.multiProcessorCount * 8, 1);
  float* output = nullptr;
  if (!hip_ok(hipMalloc(&output, static_cast<size_t>(blocks) * sizeof(float)))) return 2;
  constexpr int repetitions = 1024;
  for (const auto& entry :
       std::vector<std::pair<int, const char*>>{{1, "normal"}, {32, "bank_conflict"}}) {
    for (int sample = 0; sample < kSamples; ++sample) {
      const double milliseconds = elapsed_ms([&] {
        hipLaunchKernelGGL(lds_kernel, dim3(blocks), dim3(32), 0, 0, output,
                           entry.first, repetitions);
      });
      const double bytes = static_cast<double>(blocks) * 32 * repetitions * 2 * sizeof(float);
      if (!metric("lds_byte_per_ms", entry.second, bytes / milliseconds, "byte/ms")) return 2;
    }
  }
  return hip_ok(hipFree(output)) ? 0 : 2;
}

__global__ void reduction_kernel(const float* input, float* output, int width) {
  extern __shared__ float values[];
  const int lane = threadIdx.x;
  values[lane] = input[static_cast<size_t>(blockIdx.x) * width + lane];
  __syncthreads();
  for (int stride = width / 2; stride > 0; stride /= 2) {
    if (lane < stride) values[lane] += values[lane + stride];
    __syncthreads();
  }
  if (lane == 0) output[blockIdx.x] = values[0];
}

int run_reduction() {
  constexpr int blocks = 4096;
  constexpr int maximum_width = 1024;
  float* input = nullptr;
  float* output = nullptr;
  if (!hip_ok(hipMalloc(&input, static_cast<size_t>(blocks) * maximum_width * sizeof(float))) ||
      !hip_ok(hipMalloc(&output, static_cast<size_t>(blocks) * sizeof(float)))) return 2;
  if (!hip_ok(
          hipMemset(input, 0, static_cast<size_t>(blocks) * maximum_width * sizeof(float)))) {
    return 2;
  }
  for (int width : {32, 64, 128, 256, 512, 1024}) {
    char variant[32];
    std::snprintf(variant, sizeof(variant), "width%d", width);
    for (int sample = 0; sample < kSamples; ++sample) {
      const double milliseconds = elapsed_ms([&] {
        hipLaunchKernelGGL(reduction_kernel, dim3(blocks), dim3(width),
                           static_cast<size_t>(width) * sizeof(float), 0, input, output, width);
      });
      if (!metric("reduction_op_per_ms", variant,
                  static_cast<double>(blocks) * (width - 1) / milliseconds, "op/ms") ||
          !metric("barrier_penalty_ms", variant,
                  milliseconds /
                      (blocks * (width / 32) * std::log2(static_cast<double>(width))),
                  "ms/barrier")) {
        return 2;
      }
    }
  }
  const bool input_freed = hip_ok(hipFree(input));
  const bool output_freed = hip_ok(hipFree(output));
  return input_freed && output_freed ? 0 : 2;
}

using half8 = _Float16 __attribute__((ext_vector_type(8)));
using float8 = float __attribute__((ext_vector_type(8)));

__global__ void wmma_kernel(float* output, int repetitions) {
  half8 a = {1, 1, 1, 1, 1, 1, 1, 1};
  half8 b = {1, 1, 1, 1, 1, 1, 1, 1};
  float8 accumulator = {};
  for (int repeat = 0; repeat < repetitions; ++repeat) {
    accumulator =
        __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a, b, accumulator);
  }
  output[static_cast<size_t>(blockIdx.x) * 32 + threadIdx.x] = accumulator[0];
}

int run_wmma() {
  hipDeviceProp_t properties{};
  if (!hip_ok(hipGetDeviceProperties(&properties, 0))) return 2;
  constexpr int repetitions = 512;
  for (int waves : {1, 3, 7, 16, 32, 64}) {
    const int blocks = std::max(properties.multiProcessorCount * waves, 1);
    float* output = nullptr;
    if (!hip_ok(hipMalloc(&output, static_cast<size_t>(blocks) * 32 * sizeof(float)))) return 2;
    char variant[32];
    std::snprintf(variant, sizeof(variant), "active_waves%d", waves);
    for (int sample = 0; sample < kSamples; ++sample) {
      const double milliseconds = elapsed_ms([&] {
        hipLaunchKernelGGL(wmma_kernel, dim3(blocks), dim3(32), 0, 0, output, repetitions);
      });
      const double flops = static_cast<double>(blocks) * repetitions * 2 * 16 * 16 * 16;
      if (!metric("wmma_flop_per_ms", variant, flops / milliseconds, "flop/ms")) return 2;
    }
    if (!hip_ok(hipFree(output))) return 2;
  }
  return 0;
}

__global__ void valu_kernel(float* output, int repetitions) {
  float a = static_cast<float>(threadIdx.x + 1);
  float b = 1.0001f;
  float c = 0.0001f;
  for (int repeat = 0; repeat < repetitions; ++repeat) {
    a = fmaf(a, b, c);
  }
  output[static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x] = a;
}

int run_valu() {
  hipDeviceProp_t properties{};
  if (!hip_ok(hipGetDeviceProperties(&properties, 0))) return 2;
  constexpr int repetitions = 8192;
  const int blocks = std::max(properties.multiProcessorCount * 32, 1);
  float* output = nullptr;
  if (!hip_ok(hipMalloc(&output, static_cast<size_t>(blocks) * kThreads * sizeof(float)))) return 2;
  for (int sample = 0; sample < kSamples; ++sample) {
    const double milliseconds = elapsed_ms([&] {
      hipLaunchKernelGGL(valu_kernel, dim3(blocks), dim3(kThreads), 0, 0, output, repetitions);
    });
    const double flops = static_cast<double>(blocks) * kThreads * repetitions * 2;
    if (!metric("valu_flop_per_ms", "fp32", flops / milliseconds, "flop/ms")) return 2;
  }
  return hip_ok(hipFree(output)) ? 0 : 2;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 2) {
    std::fprintf(stderr, "usage: diagnostic_microarchitecture MODE\n");
    return 1;
  }
  if (std::strcmp(argv[1], "dispatch") == 0) return run_dispatch();
  if (std::strcmp(argv[1], "memory") == 0) return run_memory();
  if (std::strcmp(argv[1], "access") == 0) return run_access();
  if (std::strcmp(argv[1], "lds") == 0) return run_lds();
  if (std::strcmp(argv[1], "reduction") == 0) return run_reduction();
  if (std::strcmp(argv[1], "wmma") == 0) return run_wmma();
  if (std::strcmp(argv[1], "valu") == 0) return run_valu();
  std::fprintf(stderr, "unknown mode: %s\n", argv[1]);
  return 1;
}
