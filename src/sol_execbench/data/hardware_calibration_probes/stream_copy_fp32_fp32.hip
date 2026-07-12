#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

constexpr int arithmetic_repetitions = 4096;

__global__ void probe_kernel(const float* left, const float* right, float* out, size_t count) {
  const size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < count) { out[index] = left[index]; }
}

int main() {
  size_t free_bytes = 0, total_bytes = 0;
  if (hipMemGetInfo(&free_bytes, &total_bytes) != hipSuccess) return 2;
  const size_t maximum_count = static_cast<size_t>(1) << 27;
  const size_t count = std::min(maximum_count, free_bytes / (sizeof(float) * 12));
  if (count < (static_cast<size_t>(1) << 24)) return 5;
  std::vector<float> left(count), right(count), out(count);
  for (size_t index = 0; index < count; ++index) { left[index] = static_cast<float>(1.0f); right[index] = static_cast<float>(0.0f); }
  float *d_left = nullptr, *d_right = nullptr, *d_out = nullptr;
  if (hipMalloc(&d_left, count * sizeof(float)) || hipMalloc(&d_right, count * sizeof(float)) || hipMalloc(&d_out, count * sizeof(float))) return 2;
  hipMemcpy(d_left, left.data(), count * sizeof(float), hipMemcpyHostToDevice);
  hipMemcpy(d_right, right.data(), count * sizeof(float), hipMemcpyHostToDevice);
  hipEvent_t start, stop; hipEventCreate(&start); hipEventCreate(&stop);
  for (int warmup = 0; warmup < 3; ++warmup) {
    hipLaunchKernelGGL(probe_kernel, dim3((count + 255) / 256), dim3(256), 0, 0, d_left, d_right, d_out, count);
  }
  hipDeviceSynchronize();
  for (int repeat = 0; repeat < 7; ++repeat) {
    hipEventRecord(start); hipLaunchKernelGGL(probe_kernel, dim3((count + 255) / 256), dim3(256), 0, 0, d_left, d_right, d_out, count); hipEventRecord(stop); hipEventSynchronize(stop);
    float elapsed_ms = 0.0f; hipEventElapsedTime(&elapsed_ms, start, stop);
    if (elapsed_ms <= 0.0f) return 3;
    std::printf("RESULT %.9f\n", (static_cast<double>(count) * 2.0 * sizeof(float) / elapsed_ms / 1.0e6));
  }
  hipMemcpy(out.data(), d_out, count * sizeof(float), hipMemcpyDeviceToHost);
  
  for (size_t index = 0; index < count; ++index) if (std::fabs(static_cast<float>(out[index]) - 1.0f) > 1.0e-5f) return 4;
  hipFree(d_left); hipFree(d_right); hipFree(d_out); hipEventDestroy(start); hipEventDestroy(stop); return 0;
}
