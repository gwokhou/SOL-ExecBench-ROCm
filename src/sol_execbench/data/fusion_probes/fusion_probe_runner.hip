// SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
// SPDX-License-Identifier: Apache-2.0

// The Python collector compiles this executable once and invokes it in three
// independent processes for every shape.  Its line-oriented output deliberately
// avoids a JSON dependency in the HIP build environment.

#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "gfx1200_reduction_epilogue.hip"

namespace {
constexpr int kRunnerBlockSize = 256;
constexpr int kTanhBlocks = 1024;

#define HIP_CHECK(call)                                                        \
  do {                                                                         \
    const hipError_t error = (call);                                           \
    if (error != hipSuccess) {                                                 \
      std::fprintf(stderr, "%s failed: %s\n", #call, hipGetErrorString(error)); \
      return 2;                                                                \
    }                                                                          \
  } while (0)

__global__ void initialise(float* values, long count, int modulus) {
  const long index = static_cast<long>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < count) values[index] = static_cast<float>((index % modulus) - modulus / 2) * 0.03125f;
}

template <typename Kernel>
int occupancy(Kernel kernel) {
  int blocks = 0;
  hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, kernel, kRunnerBlockSize, 0);
  return blocks;
}

int lds_limit_bytes() {
  int device = 0;
  int limit = 0;
  if (hipGetDevice(&device) != hipSuccess ||
      hipDeviceGetAttribute(&limit, hipDeviceAttributeMaxSharedMemoryPerBlock,
                            device) != hipSuccess) {
    return 0;
  }
  return limit;
}

int rms(int rows) {
  const long count = static_cast<long>(rows) * 4096;
  float *input = nullptr, *fused = nullptr, *unfused = nullptr;
  HIP_CHECK(hipMalloc(&input, count * sizeof(float)));
  HIP_CHECK(hipMalloc(&fused, rows * sizeof(float)));
  HIP_CHECK(hipMalloc(&unfused, rows * sizeof(float)));
  initialise<<<(count + 255) / 256, 256>>>(input, count, 17);
  HIP_CHECK(hipDeviceSynchronize());
  const dim3 grid(rows), block(kRunnerBlockSize);
  auto fused_launch = [&] { sol_fusion_rms_mean_epsilon<<<grid, block>>>(input, fused, rows, 4096, 1.0e-5f); };
  auto unfused_launch = [&] {
    sol_unfused_rms_mean<<<grid, block>>>(input, unfused, rows, 4096);
    sol_unfused_rms_epsilon<<<(rows + 255) / 256, 256>>>(unfused, rows, 1.0e-5f);
  };
  for (int index = 0; index < 10; ++index) { fused_launch(); unfused_launch(); }
  HIP_CHECK(hipDeviceSynchronize());
  fused_launch();
  unfused_launch();
  HIP_CHECK(hipDeviceSynchronize());
  std::vector<float> left(rows), right(rows);
  HIP_CHECK(hipMemcpy(left.data(), fused, rows * sizeof(float), hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(right.data(), unfused, rows * sizeof(float), hipMemcpyDeviceToHost));
  bool correct = true;
  for (int index = 0; index < rows; ++index) correct = correct && std::fabs(left[index] - right[index]) <= 1.0e-4f;
  std::printf("CORRECT %d\n", correct ? 1 : 0);
  std::printf("LDS_LIMIT %d\n", lds_limit_bytes());
  std::printf("OCCUPANCY sol_fusion_rms_mean_epsilon %d\n", occupancy(sol_fusion_rms_mean_epsilon));
  std::printf("OCCUPANCY sol_unfused_rms_mean %d\n", occupancy(sol_unfused_rms_mean));
  std::printf("OCCUPANCY sol_unfused_rms_epsilon %d\n", occupancy(sol_unfused_rms_epsilon));
  for (int pair = 0; pair < 30; ++pair) {
    float fused_ms = 0.0f, unfused_ms = 0.0f;
    hipEvent_t start = nullptr, stop = nullptr;
    hipEventCreate(&start); hipEventCreate(&stop);
    auto measure_fused = [&] { hipEventRecord(start); fused_launch(); hipEventRecord(stop); hipEventSynchronize(stop); hipEventElapsedTime(&fused_ms, start, stop); };
    auto measure_unfused = [&] { hipEventRecord(start); unfused_launch(); hipEventRecord(stop); hipEventSynchronize(stop); hipEventElapsedTime(&unfused_ms, start, stop); };
    if (pair % 2 == 0) { measure_fused(); measure_unfused(); } else { measure_unfused(); measure_fused(); }
    hipEventDestroy(start); hipEventDestroy(stop);
    std::printf("PAIR %.9g %.9g\n", fused_ms, unfused_ms);
  }
  hipFree(input); hipFree(fused); hipFree(unfused);
  return 0;
}

int tanh_backward(int rows) {
  const long count = static_cast<long>(rows) * 4096;
  float *input = nullptr, *fused = nullptr, *unfused = nullptr;
  HIP_CHECK(hipMalloc(&input, count * sizeof(float))); HIP_CHECK(hipMalloc(&fused, sizeof(float))); HIP_CHECK(hipMalloc(&unfused, sizeof(float)));
  initialise<<<(count + 255) / 256, 256>>>(input, count, 17);
  HIP_CHECK(hipDeviceSynchronize());
  auto fused_launch = [&] { sol_fusion_tanh_backward_sum_scalar<<<kTanhBlocks, kRunnerBlockSize>>>(input, fused, count, 0.75f); };
  auto unfused_launch = [&] { sol_unfused_tanh_backward_sum<<<kTanhBlocks, kRunnerBlockSize>>>(input, unfused, count); sol_unfused_tanh_backward_scalar<<<1, 1>>>(unfused, 0.75f); };
  for (int index = 0; index < 10; ++index) { hipMemset(fused, 0, sizeof(float)); fused_launch(); hipMemset(unfused, 0, sizeof(float)); unfused_launch(); }
  HIP_CHECK(hipDeviceSynchronize()); hipMemset(fused, 0, sizeof(float)); fused_launch(); hipMemset(unfused, 0, sizeof(float)); unfused_launch(); HIP_CHECK(hipDeviceSynchronize());
  float first = 0.0f, second = 0.0f; HIP_CHECK(hipMemcpy(&first, fused, sizeof(float), hipMemcpyDeviceToHost)); HIP_CHECK(hipMemcpy(&second, unfused, sizeof(float), hipMemcpyDeviceToHost));
  const bool correct = std::fabs(first - second) <= 0.05f * fmaxf(1.0f, std::fabs(second));
  std::printf("CORRECT %d\n", correct ? 1 : 0); std::printf("LDS_LIMIT %d\n", lds_limit_bytes());
  std::printf("OCCUPANCY sol_fusion_tanh_backward_sum_scalar %d\n", occupancy(sol_fusion_tanh_backward_sum_scalar));
  std::printf("OCCUPANCY sol_unfused_tanh_backward_sum %d\n", occupancy(sol_unfused_tanh_backward_sum));
  std::printf("OCCUPANCY sol_unfused_tanh_backward_scalar %d\n", occupancy(sol_unfused_tanh_backward_scalar));
  for (int pair = 0; pair < 30; ++pair) {
    float fused_ms = 0.0f, unfused_ms = 0.0f; hipEvent_t start = nullptr, stop = nullptr; hipEventCreate(&start); hipEventCreate(&stop);
    auto measure_fused = [&] { hipMemset(fused, 0, sizeof(float)); hipEventRecord(start); fused_launch(); hipEventRecord(stop); hipEventSynchronize(stop); hipEventElapsedTime(&fused_ms, start, stop); };
    auto measure_unfused = [&] { hipMemset(unfused, 0, sizeof(float)); hipEventRecord(start); unfused_launch(); hipEventRecord(stop); hipEventSynchronize(stop); hipEventElapsedTime(&unfused_ms, start, stop); };
    if (pair % 2 == 0) { measure_fused(); measure_unfused(); } else { measure_unfused(); measure_fused(); }
    hipEventDestroy(start); hipEventDestroy(stop); std::printf("PAIR %.9g %.9g\n", fused_ms, unfused_ms);
  }
  hipFree(input); hipFree(fused); hipFree(unfused); return 0;
}
}  // namespace

int main(int argc, char** argv) {
  if (argc != 4) { std::fprintf(stderr, "usage: fusion_probe_runner <device> <rms|tanh> <rows>\n"); return 2; }
  HIP_CHECK(hipSetDevice(std::atoi(argv[1])));
  const int rows = std::atoi(argv[3]);
  if (rows <= 0) return 2;
  return std::strcmp(argv[2], "rms") == 0 ? rms(rows) : std::strcmp(argv[2], "tanh") == 0 ? tanh_backward(rows) : 2;
}
