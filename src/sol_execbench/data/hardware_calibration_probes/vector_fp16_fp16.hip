#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

constexpr int arithmetic_repetitions = 4096;

__global__ void probe_kernel(const __half* left, const __half* right, __half* out, size_t count) {
  const size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < count) { __half value = left[index];
    #pragma unroll 16
    for (int repeat = 0; repeat < arithmetic_repetitions; ++repeat) {
      value = __float2half(fmaf(__half2float(value), __half2float(right[index]), 0.0000001f));
    }
    out[index] = value; }
}

int main() {
  constexpr size_t count = 1 << 20;
  std::vector<__half> left(count), right(count), out(count);
  for (size_t index = 0; index < count; ++index) { left[index] = __float2half(1.0f); right[index] = __float2half(1.0f); }
  __half *d_left = nullptr, *d_right = nullptr, *d_out = nullptr;
  if (hipMalloc(&d_left, count * sizeof(__half)) || hipMalloc(&d_right, count * sizeof(__half)) || hipMalloc(&d_out, count * sizeof(__half))) return 2;
  if (hipMemcpy(d_left, left.data(), count * sizeof(__half), hipMemcpyHostToDevice) != hipSuccess || hipMemcpy(d_right, right.data(), count * sizeof(__half), hipMemcpyHostToDevice) != hipSuccess) return 2;
  hipEvent_t start, stop; hipEventCreate(&start); hipEventCreate(&stop);
  for (int warmup = 0; warmup < 3; ++warmup) {
    hipLaunchKernelGGL(probe_kernel, dim3((count + 255) / 256), dim3(256), 0, 0, d_left, d_right, d_out, count);
  }
  if (hipDeviceSynchronize() != hipSuccess) return 2;
  for (int repeat = 0; repeat < 7; ++repeat) {
    hipEventRecord(start); hipLaunchKernelGGL(probe_kernel, dim3((count + 255) / 256), dim3(256), 0, 0, d_left, d_right, d_out, count); hipEventRecord(stop); hipEventSynchronize(stop);
    float elapsed_ms = 0.0f; hipEventElapsedTime(&elapsed_ms, start, stop);
    if (elapsed_ms <= 0.0f) return 3;
    std::printf("RESULT %.9f\n", (static_cast<double>(count) * 2.0 * arithmetic_repetitions / elapsed_ms / 1.0e9));
  }
  if (hipMemcpy(out.data(), d_out, count * sizeof(__half), hipMemcpyDeviceToHost) != hipSuccess) return 2;
  for (size_t index = 0; index < count; ++index) if (!std::isfinite(__half2float(out[index])) || std::fabs(__half2float(out[index]) - 1.0f) > 0.01f) return 4;
  hipFree(d_left); hipFree(d_right); hipFree(d_out); hipEventDestroy(start); hipEventDestroy(stop); return 0;
}
