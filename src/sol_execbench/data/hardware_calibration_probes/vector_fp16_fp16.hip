#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

constexpr int arithmetic_repetitions = 4096;

// Packed fp16 VALU probe. Each thread drives a __half2 (2 fp16 lanes) through a
// dependent FMA recurrence via __hfma2, which lowers to v_pk_fma_f16 (true
// RDNA4 packed dual-issue, 2:1 over fp32). Each packed FMA = 2 lanes * 2 flops
// (mul+add) = 4 flops. `right` is a runtime memory load so the compiler cannot
// constant-fold the loop-carried FMA away.
__global__ void probe_kernel(const __half2* left, const __half2* right, __half2* out, size_t count2) {
  const size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < count2) {
    __half2 value = left[index];
    const __half2 r = right[index];
    const __half2 addend = __float2half2_rn(0.0000001f);
    #pragma unroll 16
    for (int repeat = 0; repeat < arithmetic_repetitions; ++repeat) {
      value = __hfma2(value, r, addend);
    }
    out[index] = value;
  }
}

int main() {
  constexpr size_t count = 1 << 20;         // fp16 elements
  constexpr size_t count2 = count / 2;      // __half2 lanes / threads
  std::vector<__half2> left(count2), right(count2), out(count2);
  for (size_t index = 0; index < count2; ++index) {
    left[index] = __float2half2_rn(1.0f);
    right[index] = __float2half2_rn(1.0000001f);
  }
  __half2 *d_left = nullptr, *d_right = nullptr, *d_out = nullptr;
  if (hipMalloc(&d_left, count2 * sizeof(__half2)) || hipMalloc(&d_right, count2 * sizeof(__half2)) || hipMalloc(&d_out, count2 * sizeof(__half2))) return 2;
  if (hipMemcpy(d_left, left.data(), count2 * sizeof(__half2), hipMemcpyHostToDevice) != hipSuccess || hipMemcpy(d_right, right.data(), count2 * sizeof(__half2), hipMemcpyHostToDevice) != hipSuccess) return 2;
  hipEvent_t start, stop; hipEventCreate(&start); hipEventCreate(&stop);
  for (int warmup = 0; warmup < 3; ++warmup) {
    hipLaunchKernelGGL(probe_kernel, dim3((count2 + 255) / 256), dim3(256), 0, 0, d_left, d_right, d_out, count2);
  }
  if (hipDeviceSynchronize() != hipSuccess) return 2;
  for (int repeat = 0; repeat < 7; ++repeat) {
    hipEventRecord(start); hipLaunchKernelGGL(probe_kernel, dim3((count2 + 255) / 256), dim3(256), 0, 0, d_left, d_right, d_out, count2); hipEventRecord(stop); hipEventSynchronize(stop);
    float elapsed_ms = 0.0f; hipEventElapsedTime(&elapsed_ms, start, stop);
    if (elapsed_ms <= 0.0f) return 3;
    std::printf("RESULT %.9f\n", (static_cast<double>(count2) * 4.0 * arithmetic_repetitions / elapsed_ms / 1.0e9));
  }
  if (hipMemcpy(out.data(), d_out, count2 * sizeof(__half2), hipMemcpyDeviceToHost) != hipSuccess) return 2;
  for (size_t index = 0; index < count2; ++index) {
    const float2 v = __half22float2(out[index]);
    if (!std::isfinite(v.x) || !std::isfinite(v.y) || std::fabs(v.x - 1.0f) > 0.1f || std::fabs(v.y - 1.0f) > 0.1f) return 4;
  }
  hipFree(d_left); hipFree(d_right); hipFree(d_out); hipEventDestroy(start); hipEventDestroy(stop); return 0;
}
