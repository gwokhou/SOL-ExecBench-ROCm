#include <hip/hip_runtime.h>
#include <cmath>
#include <cstdio>
#include <vector>

constexpr size_t count = static_cast<size_t>(1) << 24;
constexpr int threads = 256;

__global__ void reduction_probe_kernel(const float* input, float* output) {
  __shared__ float partial[threads];
  const size_t index = static_cast<size_t>(blockIdx.x) * threads + threadIdx.x;
  partial[threadIdx.x] = input[index];
  __syncthreads();
  for (int stride = threads / 2; stride > 0; stride /= 2) {
    if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
    __syncthreads();
  }
  if (threadIdx.x == 0) output[blockIdx.x] = partial[0];
}

int main() {
  constexpr size_t blocks = count / threads;
  std::vector<float> input(count, 1.0f), output(blocks);
  float *d_input = nullptr, *d_output = nullptr;
  if (hipMalloc(&d_input, count * sizeof(float)) != hipSuccess || hipMalloc(&d_output, blocks * sizeof(float)) != hipSuccess) return 2;
  if (hipMemcpy(d_input, input.data(), count * sizeof(float), hipMemcpyHostToDevice) != hipSuccess) return 2;
  hipEvent_t start, stop; hipEventCreate(&start); hipEventCreate(&stop);
  for (int warmup = 0; warmup < 3; ++warmup) hipLaunchKernelGGL(reduction_probe_kernel, dim3(blocks), dim3(threads), 0, 0, d_input, d_output);
  if (hipDeviceSynchronize() != hipSuccess) return 2;
  for (int sample = 0; sample < 7; ++sample) {
    hipEventRecord(start); hipLaunchKernelGGL(reduction_probe_kernel, dim3(blocks), dim3(threads), 0, 0, d_input, d_output); hipEventRecord(stop); hipEventSynchronize(stop);
    float elapsed_ms = 0.0f; hipEventElapsedTime(&elapsed_ms, start, stop);
    if (elapsed_ms <= 0.0f) return 3;
    std::printf("RESULT %.9f\n", static_cast<double>(count - blocks) / elapsed_ms / 1.0e9);
  }
  if (hipMemcpy(output.data(), d_output, blocks * sizeof(float), hipMemcpyDeviceToHost) != hipSuccess) return 2;
  for (float value : output) if (std::fabs(value - static_cast<float>(threads)) > 1.0e-5f) return 4;
  hipFree(d_input); hipFree(d_output); hipEventDestroy(start); hipEventDestroy(stop); return 0;
}
