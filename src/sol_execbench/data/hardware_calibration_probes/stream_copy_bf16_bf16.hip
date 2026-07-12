#include <hip/hip_runtime.h>
#include <hip/hip_bfloat16.h>
#include <hip/hip_fp16.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

constexpr int arithmetic_repetitions = 4096;

__global__ void probe_kernel(const hip_bfloat16* left, const hip_bfloat16* right, hip_bfloat16* out, size_t count) {
  const size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < count) { out[index] = left[index]; }
}

int main() {
  size_t free_bytes = 0, total_bytes = 0;
  if (hipMemGetInfo(&free_bytes, &total_bytes) != hipSuccess) return 2;
  const size_t maximum_count = static_cast<size_t>(1) << 27;
  const size_t count = std::min(maximum_count, free_bytes / (sizeof(hip_bfloat16) * 12));
  if (count < (static_cast<size_t>(1) << 24)) return 5;
  std::vector<hip_bfloat16> left(count), right(count), out(count);
  for (size_t index = 0; index < count; ++index) { left[index] = hip_bfloat16(1.0f); right[index] = hip_bfloat16(0.0f); }
  hip_bfloat16 *d_left = nullptr, *d_right = nullptr, *d_out = nullptr;
  if (hipMalloc(&d_left, count * sizeof(hip_bfloat16)) || hipMalloc(&d_right, count * sizeof(hip_bfloat16)) || hipMalloc(&d_out, count * sizeof(hip_bfloat16))) return 2;
  if (hipMemcpy(d_left, left.data(), count * sizeof(hip_bfloat16), hipMemcpyHostToDevice) != hipSuccess || hipMemcpy(d_right, right.data(), count * sizeof(hip_bfloat16), hipMemcpyHostToDevice) != hipSuccess) return 2;
  hipEvent_t start, stop; hipEventCreate(&start); hipEventCreate(&stop);
  for (int warmup = 0; warmup < 3; ++warmup) {
    hipLaunchKernelGGL(probe_kernel, dim3((count + 255) / 256), dim3(256), 0, 0, d_left, d_right, d_out, count);
  }
  if (hipDeviceSynchronize() != hipSuccess) return 2;
  for (int repeat = 0; repeat < 7; ++repeat) {
    hipEventRecord(start); hipLaunchKernelGGL(probe_kernel, dim3((count + 255) / 256), dim3(256), 0, 0, d_left, d_right, d_out, count); hipEventRecord(stop); hipEventSynchronize(stop);
    float elapsed_ms = 0.0f; hipEventElapsedTime(&elapsed_ms, start, stop);
    if (elapsed_ms <= 0.0f) return 3;
    std::printf("RESULT %.9f\n", (static_cast<double>(count) * 2.0 * sizeof(hip_bfloat16) / elapsed_ms / 1.0e6));
  }
  if (hipMemcpy(out.data(), d_out, count * sizeof(hip_bfloat16), hipMemcpyDeviceToHost) != hipSuccess) return 2;
  for (size_t index = 0; index < count; ++index) if (!std::isfinite(static_cast<float>(out[index])) || std::fabs(static_cast<float>(out[index]) - 1.0f) > 0.01f) return 4;
  hipFree(d_left); hipFree(d_right); hipFree(d_out); hipEventDestroy(start); hipEventDestroy(stop); return 0;
}
