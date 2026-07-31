#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include <hip/hip_fp16.h>

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

__global__ void transpose_kernel(const float* input, float* output, size_t rows,
                                 size_t columns) {
  const size_t input_index =
      static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const size_t count = rows * columns;
  if (input_index >= count) return;
  const size_t row = input_index / columns;
  const size_t column = input_index % columns;
  output[column * rows + row] = input[input_index];
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
  const int contiguous = measure_copy(bytes, 1, "contiguous");
  if (contiguous != 0) return contiguous;
  constexpr size_t rows = 4096;
  constexpr size_t columns = 8192;
  const size_t count = rows * columns;
  float* input = nullptr;
  float* output = nullptr;
  if (!hip_ok(hipMalloc(&input, bytes)) || !hip_ok(hipMalloc(&output, bytes))) return 2;
  if (!hip_ok(hipMemset(input, 1, bytes))) return 2;
  for (int sample = 0; sample < kSamples; ++sample) {
    const double milliseconds = elapsed_ms([&] {
      hipLaunchKernelGGL(
          transpose_kernel, dim3((count + kThreads - 1) / kThreads),
          dim3(kThreads), 0, 0, input, output, rows, columns);
    });
    if (!metric("memory_byte_per_ms", "transpose", 2.0 * bytes / milliseconds,
                "byte/ms")) {
      return 2;
    }
  }
  const bool input_freed = hip_ok(hipFree(input));
  const bool output_freed = hip_ok(hipFree(output));
  return input_freed && output_freed ? 0 : 2;
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
  double normal_ms[kSamples] = {};
  for (const auto& entry :
       std::vector<std::pair<int, const char*>>{{1, "normal"}, {32, "bank_conflict"}}) {
    for (int sample = 0; sample < kSamples; ++sample) {
      const double milliseconds = elapsed_ms([&] {
        hipLaunchKernelGGL(lds_kernel, dim3(blocks), dim3(32), 0, 0, output,
                           entry.first, repetitions);
      });
      if (entry.first == 1) normal_ms[sample] = milliseconds;
      const double bytes = static_cast<double>(blocks) * 32 * repetitions * 2 * sizeof(float);
      if (!metric("lds_byte_per_ms", entry.second, bytes / milliseconds, "byte/ms")) return 2;
      if (entry.first == 32 &&
          !metric("lds_bank_conflict_penalty_ms", entry.second,
                  std::max(milliseconds - normal_ms[sample], 1.0e-12) /
                      (static_cast<double>(blocks) * repetitions),
                  "ms/event")) {
        return 2;
      }
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
                  static_cast<double>(blocks) * (width - 1) / milliseconds, "item/ms") ||
          !metric("barrier_penalty_ms", variant,
                  milliseconds /
                      (blocks * (width / 32) * std::log2(static_cast<double>(width))),
                  "ms/event")) {
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
      char edge_variant[64];
      char irregular_variant[64];
      std::snprintf(edge_variant, sizeof(edge_variant), "edge_tile15_%s", variant);
      std::snprintf(irregular_variant, sizeof(irregular_variant),
                    "irregular_tile17_%s", variant);
      if (!metric("wmma_flop_per_ms", edge_variant,
                  flops * (15.0 / 16.0) * (15.0 / 16.0) / milliseconds,
                  "flop/ms") ||
          !metric("wmma_flop_per_ms", irregular_variant,
                  flops * (17.0 / 32.0) * (17.0 / 32.0) / milliseconds,
                  "flop/ms")) {
        return 2;
      }
    }
    if (!hip_ok(hipFree(output))) return 2;
  }
  return 0;
}

__global__ void valu_kernel(float* output, int repetitions, int operation, bool use_bf16) {
  float a = static_cast<float>(threadIdx.x + 1);
  float b = 1.0001f;
  float c = 0.0001f;
  for (int repeat = 0; repeat < repetitions; ++repeat) {
    if (operation == 0) {
      a = fmaf(a, b, c);
    } else if (operation == 1) {
      a = expf(fminf(a * 0.00001f, 1.0f)) + c;
    } else {
      const float exponential = expf(-fminf(a * 0.00001f, 1.0f));
      a = 1.0f / (1.0f + exponential) + c;
    }
    if (use_bf16) a = static_cast<float>(__hip_bfloat16(a));
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
  const char* classes[] = {"simple", "transcendental", "composite"};
  for (int operation = 0; operation < 3; ++operation) {
    for (int dtype = 0; dtype < 2; ++dtype) {
      char variant[48];
      std::snprintf(variant, sizeof(variant), "%s_%s", classes[operation],
                    dtype == 0 ? "fp32" : "bf16");
      for (int sample = 0; sample < kSamples; ++sample) {
        const double milliseconds = elapsed_ms([&] {
          hipLaunchKernelGGL(valu_kernel, dim3(blocks), dim3(kThreads), 0, 0, output,
                             repetitions, operation, dtype == 1);
        });
        const double items = static_cast<double>(blocks) * kThreads * repetitions;
        if (!metric("valu_item_per_ms", variant, items / milliseconds, "item/ms")) return 2;
      }
    }
  }
  return hip_ok(hipFree(output)) ? 0 : 2;
}

template <typename T>
__global__ void indexed_read_kernel(const T* source, const int* indices,
                                    T* output, size_t count) {
  const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < count) output[index] = source[indices[index]];
}

template <typename T>
double measure_indexed_read(const T* source, const int* indices, T* output,
                            size_t count) {
  return elapsed_ms([&] {
    hipLaunchKernelGGL((indexed_read_kernel<T>),
                       dim3((count + kThreads - 1) / kThreads), dim3(kThreads),
                       0, 0, source, indices, output, count);
  });
}

template <typename T>
int emit_indexed_read_metrics(const T* source, int* device_indices, T* output,
                              size_t source_count, size_t count, int pattern,
                              const char* variant, bool emit_address_rate) {
  std::vector<int> indices(count);
  unsigned int state = 17;
  for (size_t index = 0; index < count; ++index) {
    state = state * 1664525U + 1013904223U;
    if (pattern == 0) indices[index] = static_cast<int>(state % source_count);
    if (pattern == 1) {
      indices[index] =
          static_cast<int>(((index / 2) * 17 + index % 2) % source_count);
    }
    if (pattern == 2) indices[index] = static_cast<int>(index % source_count);
  }
  if (!hip_ok(hipMemcpy(device_indices, indices.data(), count * sizeof(int),
                        hipMemcpyHostToDevice))) {
    return 2;
  }
  for (int sample = 0; sample < kSamples; ++sample) {
    const double milliseconds =
        measure_indexed_read(source, device_indices, output, count);
    if (!metric("indexed_read_item_per_ms", variant, count / milliseconds,
                "item/ms")) {
      return 2;
    }
    if (emit_address_rate && pattern == 2 &&
        !metric("indexed_address_op_per_ms", "device", count / milliseconds,
                "item/ms")) {
      return 2;
    }
  }
  return 0;
}

template <typename T>
int run_indexed_read_type(int* indices, size_t source_bytes,
                          const size_t* working_set_bytes,
                          const char* element_interval,
                          bool emit_address_rate) {
  constexpr size_t count = 1ULL << 20;
  T* source = nullptr;
  T* output = nullptr;
  if (!hip_ok(hipMalloc(&source, source_bytes)) ||
      !hip_ok(hipMalloc(&output, count * sizeof(T))) ||
      !hip_ok(hipMemset(source, 1, source_bytes))) {
    return 2;
  }
  const char* locality[] = {"0:0.33", "0.34:0.66", "0.67:1"};
  const char* working_sets[] = {
      "0:2097152", "2097152.000001:67108864",
      "67108864.000001:134217728"};
  for (int tier = 0; tier < 3; ++tier) {
    const size_t source_count = working_set_bytes[tier] / sizeof(T);
    for (int pattern = 0; pattern < 3; ++pattern) {
      char variant[192];
      std::snprintf(
          variant, sizeof(variant),
          "index_locality=%s,working_set_bytes=%s,element_bytes=%s",
          locality[pattern], working_sets[tier], element_interval);
      if (emit_indexed_read_metrics(
              source, indices, output, source_count, count, pattern, variant,
              emit_address_rate && tier == 0) != 0) {
        return 2;
      }
    }
  }
  const bool freed = hip_ok(hipFree(source)) && hip_ok(hipFree(output));
  return freed ? 0 : 2;
}

int run_indexed_read() {
  constexpr size_t source_bytes = 128ULL << 20;
  constexpr size_t working_set_bytes[] = {
      256ULL << 10, 16ULL << 20, 128ULL << 20};
  int* indices = nullptr;
  if (!hip_ok(hipMalloc(&indices, (1ULL << 20) * sizeof(int)))) return 2;
  const int fp32_status = run_indexed_read_type<float>(
      indices, source_bytes, working_set_bytes, "4:4", true);
  const int fp16_status = run_indexed_read_type<__half>(
      indices, source_bytes, working_set_bytes, "2:2", false);
  const bool freed = hip_ok(hipFree(indices));
  return fp32_status == 0 && fp16_status == 0 && freed ? 0 : 2;
}

__global__ void atomic_update_kernel(float* output, const int* indices, size_t count) {
  const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < count) atomicAdd(&output[indices[index]], 1.0f);
}

int emit_atomic_metrics(float* output, int* device_indices, size_t count,
                        size_t output_count, const char* variant) {
  std::vector<int> indices(count);
  const size_t multiplicity = count / output_count;
  for (size_t index = 0; index < count; ++index) {
    indices[index] = static_cast<int>(index / multiplicity);
  }
  if (!hip_ok(hipMemcpy(device_indices, indices.data(), count * sizeof(int),
                        hipMemcpyHostToDevice))) {
    return 2;
  }
  for (int sample = 0; sample < kSamples; ++sample) {
    if (!hip_ok(hipMemset(output, 0, count * sizeof(float)))) return 2;
    const double milliseconds = elapsed_ms([&] {
      hipLaunchKernelGGL(atomic_update_kernel, dim3((count + kThreads - 1) / kThreads),
                         dim3(kThreads), 0, 0, output, device_indices, count);
    });
    if (!metric("atomic_update_item_per_ms", variant, count / milliseconds,
                "item/ms")) {
      return 2;
    }
  }
  return 0;
}

int run_atomic_update() {
  constexpr size_t count = 1ULL << 20;
  float* output = nullptr;
  int* indices = nullptr;
  if (!hip_ok(hipMalloc(&output, count * sizeof(float))) ||
      !hip_ok(hipMalloc(&indices, count * sizeof(int)))) {
    return 2;
  }
  const size_t output_counts[] = {count, count / 4, count / 32};
  const char* variants[] = {
      "collision_fraction=0:0.25,max_multiplicity=1:2,element_bytes=4:4",
      "collision_fraction=0.250001:0.85,max_multiplicity=3:8,element_bytes=4:4",
      "collision_fraction=0.850001:1,max_multiplicity=9:1.0995e12,element_bytes=4:4"};
  for (int pattern = 0; pattern < 3; ++pattern) {
    if (emit_atomic_metrics(output, indices, count, output_counts[pattern],
                            variants[pattern]) != 0) {
      return 2;
    }
  }
  return hip_ok(hipFree(output)) && hip_ok(hipFree(indices)) ? 0 : 2;
}

__global__ void fp32_matrix_kernel(float* output, int repetitions, int stride) {
  const size_t logical =
      static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  float value = static_cast<float>(logical + 1);
  for (int repeat = 0; repeat < repetitions; ++repeat) {
    value = fmaf(value, 1.000001f, 0.000001f);
  }
  output[(logical * static_cast<size_t>(stride)) %
         (static_cast<size_t>(gridDim.x) * blockDim.x)] = value;
}

int run_fp32_matrix() {
  hipDeviceProp_t properties{};
  if (!hip_ok(hipGetDeviceProperties(&properties, 0))) return 2;
  const int blocks = std::max(properties.multiProcessorCount * 32, 1);
  constexpr int repetitions = 8192;
  const size_t threads = static_cast<size_t>(blocks) * kThreads;
  float* output = nullptr;
  if (!hip_ok(hipMalloc(&output, threads * sizeof(float)))) return 2;
  for (int stride : {1, 17}) {
    const char* variant = stride == 1 ? "dense" : "strided";
    for (int sample = 0; sample < kSamples; ++sample) {
      const double milliseconds = elapsed_ms([&] {
        hipLaunchKernelGGL(fp32_matrix_kernel, dim3(blocks), dim3(kThreads), 0, 0,
                           output, repetitions, stride);
      });
      const double flops = 2.0 * threads * repetitions;
      if (!metric("fp32_matrix_flop_per_ms", variant, flops / milliseconds,
                  "flop/ms")) {
        return 2;
      }
    }
  }
  return hip_ok(hipFree(output)) ? 0 : 2;
}

int run_residency() {
  hipDeviceProp_t properties{};
  if (!hip_ok(hipGetDeviceProperties(&properties, 0))) return 2;
  const int low_blocks = std::max(properties.multiProcessorCount, 1);
  const int high_blocks = std::max(properties.multiProcessorCount * 16, 1);
  constexpr int repetitions = 4096;
  const size_t high_threads = static_cast<size_t>(high_blocks) * kThreads;
  float* output = nullptr;
  if (!hip_ok(hipMalloc(&output, high_threads * sizeof(float)))) return 2;
  const char* low =
      "active_waves=0:8,element_bytes=2:4";
  const char* high =
      "active_waves=8.000001:64,element_bytes=2:4";
  for (int sample = 0; sample < kSamples; ++sample) {
    const double low_ms = elapsed_ms([&] {
      hipLaunchKernelGGL(fp32_matrix_kernel, dim3(low_blocks), dim3(kThreads), 0,
                         0, output, repetitions, 1);
    });
    const double high_ms = elapsed_ms([&] {
      hipLaunchKernelGGL(fp32_matrix_kernel, dim3(high_blocks), dim3(kThreads),
                         0, 0, output, repetitions, 1);
    });
    const double low_rate =
        2.0 * low_blocks * kThreads * repetitions / low_ms;
    const double high_rate =
        2.0 * high_blocks * kThreads * repetitions / high_ms;
    const double reference_rate = std::max(low_rate, high_rate);
    if (!metric("residency_ratio", low, low_rate / reference_rate, "ratio") ||
        !metric("residency_ratio", high, high_rate / reference_rate, "ratio")) {
      return 2;
    }
  }
  return hip_ok(hipFree(output)) ? 0 : 2;
}

__global__ void mixed_resource_kernel(const float* input, float* output,
                                      size_t count, int repetitions) {
  const size_t index =
      static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= count) return;
  float value = input[index];
  for (int repeat = 0; repeat < repetitions; ++repeat) {
    value = fmaf(value, 1.000001f, 0.000001f);
  }
  output[index] = value;
}

void launch_overlap_kernel(int repetitions, hipStream_t stream,
                           const float* input,
                           float* output, size_t count) {
  const dim3 grid((count + kThreads - 1) / kThreads);
  if (repetitions < 0) {
    hipLaunchKernelGGL(fp32_matrix_kernel, grid, dim3(kThreads), 0, stream,
                       output, 4096, 1);
    return;
  }
  hipLaunchKernelGGL(mixed_resource_kernel, grid, dim3(kThreads), 0, stream,
                     input, output, count, repetitions);
}

double measure_overlap_single(int repetitions, hipStream_t stream,
                              const float* input, float* output, size_t count) {
  return elapsed_ms([&] {
    launch_overlap_kernel(repetitions, stream, input, output, count);
    (void)hipStreamSynchronize(stream);
  });
}

double measure_overlap_pair(int repetitions, hipStream_t first,
                            hipStream_t second, const float* left,
                            const float* right, float* left_out,
                            float* right_out, size_t count) {
  return elapsed_ms([&] {
    launch_overlap_kernel(repetitions, first, left, left_out, count);
    launch_overlap_kernel(repetitions, second, right, right_out, count);
    (void)hipStreamSynchronize(first);
    (void)hipStreamSynchronize(second);
  });
}

int run_overlap() {
  constexpr size_t bytes = 32ULL << 20;
  constexpr size_t count = bytes / sizeof(float);
  float *left = nullptr, *right = nullptr, *left_out = nullptr,
        *right_out = nullptr;
  hipStream_t first = nullptr, second = nullptr;
  if (!hip_ok(hipMalloc(&left, bytes)) || !hip_ok(hipMalloc(&right, bytes)) ||
      !hip_ok(hipMalloc(&left_out, bytes)) ||
      !hip_ok(hipMalloc(&right_out, bytes)) ||
      !hip_ok(hipStreamCreate(&first)) ||
      !hip_ok(hipStreamCreate(&second))) {
    return 2;
  }
  const int repetitions[] = {0, 1, 2, 4, 8, 16, 32, 64, 128, 256, -1};
  const char* variants[] = {
      "arithmetic_repetitions=0,concurrent_dispatches=2:2",
      "arithmetic_repetitions=1,concurrent_dispatches=2:2",
      "arithmetic_repetitions=2,concurrent_dispatches=2:2",
      "arithmetic_repetitions=4,concurrent_dispatches=2:2",
      "arithmetic_repetitions=8,concurrent_dispatches=2:2",
      "arithmetic_repetitions=16,concurrent_dispatches=2:2",
      "arithmetic_repetitions=32,concurrent_dispatches=2:2",
      "arithmetic_repetitions=64,concurrent_dispatches=2:2",
      "arithmetic_repetitions=128,concurrent_dispatches=2:2",
      "arithmetic_repetitions=256,concurrent_dispatches=2:2",
      "arithmetic_repetitions=pure_compute,concurrent_dispatches=2:2"};
  for (int mode = 0; mode < 11; ++mode) {
    for (int sample = 0; sample < kSamples; ++sample) {
      const double single = measure_overlap_single(
          repetitions[mode], first, left, left_out, count);
      const double concurrent = measure_overlap_pair(
          repetitions[mode], first, second, left, right, left_out, right_out,
          count);
      const double efficiency =
          std::min(std::max(single / concurrent, 0.05), 1.0);
      if (!metric("overlap_ratio", variants[mode], efficiency, "ratio")) {
        return 2;
      }
    }
  }
  const bool freed =
      hip_ok(hipStreamDestroy(first)) && hip_ok(hipStreamDestroy(second)) &&
      hip_ok(hipFree(left)) && hip_ok(hipFree(right)) &&
      hip_ok(hipFree(left_out)) && hip_ok(hipFree(right_out));
  return freed ? 0 : 2;
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
  if (std::strcmp(argv[1], "indexed_read") == 0) return run_indexed_read();
  if (std::strcmp(argv[1], "atomic_update") == 0) return run_atomic_update();
  if (std::strcmp(argv[1], "fp32_matrix") == 0) return run_fp32_matrix();
  if (std::strcmp(argv[1], "residency") == 0) return run_residency();
  if (std::strcmp(argv[1], "overlap") == 0) return run_overlap();
  std::fprintf(stderr, "unknown mode: %s\n", argv[1]);
  return 1;
}
