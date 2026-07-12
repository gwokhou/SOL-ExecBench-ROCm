#include <hip/hip_runtime.h>
#include <cmath>
#include <cstdio>
#include <vector>

constexpr size_t count = static_cast<size_t>(1) << 24;
constexpr int repetitions = 64;

__global__ void transcendental_probe_kernel(const float* input, float* output) {
  const size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= count) return;
  float value = input[index];
  #pragma unroll 8
  for (int repeat = 0; repeat < repetitions; ++repeat) value = tanhf(value);
  output[index] = value;
}

int main() {
  std::vector<float> input(count), output(count);
  for (size_t index = 0; index < count; ++index) input[index] = 0.25f + static_cast<float>(index % 31) * 0.01f;
  float *d_input = nullptr, *d_output = nullptr;
  if (hipMalloc(&d_input, count * sizeof(float)) != hipSuccess || hipMalloc(&d_output, count * sizeof(float)) != hipSuccess) return 2;
  if (hipMemcpy(d_input, input.data(), count * sizeof(float), hipMemcpyHostToDevice) != hipSuccess) return 2;
  hipEvent_t start, stop; hipEventCreate(&start); hipEventCreate(&stop);
  for (int warmup = 0; warmup < 3; ++warmup) hipLaunchKernelGGL(transcendental_probe_kernel, dim3((count + 255) / 256), dim3(256), 0, 0, d_input, d_output);
  if (hipDeviceSynchronize() != hipSuccess) return 2;
  for (int sample = 0; sample < 7; ++sample) {
    hipEventRecord(start); hipLaunchKernelGGL(transcendental_probe_kernel, dim3((count + 255) / 256), dim3(256), 0, 0, d_input, d_output); hipEventRecord(stop); hipEventSynchronize(stop);
    float elapsed_ms = 0.0f; hipEventElapsedTime(&elapsed_ms, start, stop);
    if (elapsed_ms <= 0.0f) return 3;
    std::printf("RESULT %.9f\n", static_cast<double>(count) * repetitions / elapsed_ms / 1.0e9);
  }
  if (hipMemcpy(output.data(), d_output, count * sizeof(float), hipMemcpyDeviceToHost) != hipSuccess) return 2;
  for (float value : output) if (!std::isfinite(value) || value <= 0.0f || value >= 1.0f) return 4;
  hipFree(d_input); hipFree(d_output); hipEventDestroy(start); hipEventDestroy(stop); return 0;
}
