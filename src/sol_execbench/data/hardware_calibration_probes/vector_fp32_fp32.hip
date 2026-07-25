#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

#ifndef SOL_FP32_ACCUMULATOR_COUNT
#define SOL_FP32_ACCUMULATOR_COUNT 16
#endif

constexpr int arithmetic_repetitions = 4096;
constexpr int accumulator_count = SOL_FP32_ACCUMULATOR_COUNT;
static_assert(accumulator_count == 2 || accumulator_count == 4 ||
                  accumulator_count == 8 || accumulator_count == 16,
              "SOL_FP32_ACCUMULATOR_COUNT must be 2, 4, 8, or 16");

__device__ __forceinline__ void dual_fmac(float& value_x, float& value_y,
                                          float factor_x, float factor_y) {
  // The published FP32 vector peak requires two independent FMA operations in
  // one RDNA4 VOPD bundle.  Keep the bundle explicit: LLVM does not reliably
  // form VOPD from two source-level fmaf recurrences.
  asm volatile(
      "v_dual_fmac_f32 %0, %2, %2 :: "
      "v_dual_fmac_f32 %1, %3, %3"
      : "+v"(value_x), "+v"(value_y)
      : "v"(factor_x), "v"(factor_y));
}

__global__ void probe_kernel(const float* right, float* out, size_t count) {
  const size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < count) {
    float value_0 = 1.0f;
    float value_1 = 1.25f;
    float value_2 = 1.5f;
    float value_3 = 1.75f;
    float value_4 = 2.0f;
    float value_5 = 2.25f;
    float value_6 = 2.5f;
    float value_7 = 2.75f;
    float value_8 = 3.0f;
    float value_9 = 3.25f;
    float value_10 = 3.5f;
    float value_11 = 3.75f;
    float value_12 = 4.0f;
    float value_13 = 4.25f;
    float value_14 = 4.5f;
    float value_15 = 4.75f;
    const float factor_x = right[index];
    const float factor_y = 2.0f - right[index];
#pragma unroll 4
    for (int repeat = 0; repeat < arithmetic_repetitions; ++repeat) {
      dual_fmac(value_0, value_1, factor_x, factor_y);
#if SOL_FP32_ACCUMULATOR_COUNT >= 4
      dual_fmac(value_2, value_3, factor_x, factor_y);
#endif
#if SOL_FP32_ACCUMULATOR_COUNT >= 8
      dual_fmac(value_4, value_5, factor_x, factor_y);
      dual_fmac(value_6, value_7, factor_x, factor_y);
#endif
#if SOL_FP32_ACCUMULATOR_COUNT >= 16
      dual_fmac(value_8, value_9, factor_x, factor_y);
      dual_fmac(value_10, value_11, factor_x, factor_y);
      dual_fmac(value_12, value_13, factor_x, factor_y);
      dual_fmac(value_14, value_15, factor_x, factor_y);
#endif
    }
    const size_t offset = index * accumulator_count;
    out[offset] = value_0;
    out[offset + 1] = value_1;
#if SOL_FP32_ACCUMULATOR_COUNT >= 4
    out[offset + 2] = value_2;
    out[offset + 3] = value_3;
#endif
#if SOL_FP32_ACCUMULATOR_COUNT >= 8
    out[offset + 4] = value_4;
    out[offset + 5] = value_5;
    out[offset + 6] = value_6;
    out[offset + 7] = value_7;
#endif
#if SOL_FP32_ACCUMULATOR_COUNT >= 16
    out[offset + 8] = value_8;
    out[offset + 9] = value_9;
    out[offset + 10] = value_10;
    out[offset + 11] = value_11;
    out[offset + 12] = value_12;
    out[offset + 13] = value_13;
    out[offset + 14] = value_14;
    out[offset + 15] = value_15;
#endif
  }
}

int main() {
  constexpr size_t count = 1 << 20;
  constexpr size_t output_count = count * accumulator_count;
  std::vector<float> right(count), out(output_count);
  for (size_t index = 0; index < count; ++index) {
    right[index] = 1.0000001f;
  }
  float *d_right = nullptr, *d_out = nullptr;
  if (hipMalloc(&d_right, count * sizeof(float)) ||
      hipMalloc(&d_out, output_count * sizeof(float)))
    return 2;
  if (hipMemcpy(d_right, right.data(), count * sizeof(float),
                hipMemcpyHostToDevice) != hipSuccess)
    return 2;
  hipEvent_t start, stop;
  if (hipEventCreate(&start) != hipSuccess ||
      hipEventCreate(&stop) != hipSuccess)
    return 2;
  for (int warmup = 0; warmup < 3; ++warmup) {
    hipLaunchKernelGGL(probe_kernel, dim3((count + 255) / 256), dim3(256), 0,
                       0, d_right, d_out, count);
  }
  if (hipDeviceSynchronize() != hipSuccess)
    return 2;
  for (int repeat = 0; repeat < 7; ++repeat) {
    if (hipEventRecord(start) != hipSuccess)
      return 2;
    hipLaunchKernelGGL(probe_kernel, dim3((count + 255) / 256), dim3(256), 0,
                       0, d_right, d_out, count);
    if (hipEventRecord(stop) != hipSuccess ||
        hipEventSynchronize(stop) != hipSuccess)
      return 2;
    float elapsed_ms = 0.0f;
    if (hipEventElapsedTime(&elapsed_ms, start, stop) != hipSuccess ||
        elapsed_ms <= 0.0f)
      return 3;
    std::printf(
        "RESULT %.9f\n",
        (static_cast<double>(count) * 2.0 * accumulator_count *
         arithmetic_repetitions / elapsed_ms / 1.0e9));
  }
  if (hipMemcpy(out.data(), d_out, output_count * sizeof(float),
                hipMemcpyDeviceToHost) != hipSuccess)
    return 2;
  float expected[16] = {
      1.0f,  1.25f, 1.5f,  1.75f, 2.0f,  2.25f, 2.5f,  2.75f,
      3.0f,  3.25f, 3.5f,  3.75f, 4.0f,  4.25f, 4.5f,  4.75f,
  };
  for (int repeat = 0; repeat < arithmetic_repetitions; ++repeat) {
    for (int accumulator = 0; accumulator < accumulator_count;
         ++accumulator) {
      const float factor =
          accumulator % 2 == 0 ? 1.0000001f : 0.9999999f;
      expected[accumulator] = fmaf(factor, factor, expected[accumulator]);
    }
  }
  for (size_t index = 0; index < count; ++index) {
    for (int accumulator = 0; accumulator < accumulator_count;
         ++accumulator) {
      if (std::fabs(out[index * accumulator_count + accumulator] -
                    expected[accumulator]) > 1.0e-5f)
        return 4;
    }
  }
  static_cast<void>(hipFree(d_right));
  static_cast<void>(hipFree(d_out));
  static_cast<void>(hipEventDestroy(start));
  static_cast<void>(hipEventDestroy(stop));
  return 0;
}
