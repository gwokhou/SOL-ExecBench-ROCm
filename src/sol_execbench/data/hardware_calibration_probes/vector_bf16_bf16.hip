#include <hip/hip_runtime.h>
#include <hip/hip_bfloat16.h>
#include <hip/amd_detail/amd_hip_bf16.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

constexpr int arithmetic_repetitions = 4096;

// Packed-bf16 intrinsic probe. Each thread drives a __hip_bfloat162 (2 bf16
// lanes) through a dependent __hfma2 recurrence. gfx1200 does not declare
// v_pk_fma_bf16, so ROCm 7.2 lowers this intrinsic to scalar fp32 FMA plus bf16
// conversion. The calibration audit verifies that absence twice: against the
// machine-readable ISA and against the emitted code object. Each source-level
// packed FMA still represents 2 lanes * 2 flops (mul+add) = 4 requested flops.
// `right` is a runtime load so the compiler cannot fold the recurrence away.
__global__ void probe_kernel(const __hip_bfloat162* left, const __hip_bfloat162* right, __hip_bfloat162* out, size_t count2) {
  const size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < count2) {
    __hip_bfloat162 value = left[index];
    const __hip_bfloat162 r = right[index];
    const __hip_bfloat162 addend = __bfloat162bfloat162(__float2bfloat16(0.0000001f));
    #pragma unroll 16
    for (int repeat = 0; repeat < arithmetic_repetitions; ++repeat) {
      value = __hfma2(value, r, addend);
    }
    out[index] = value;
  }
}

int main() {
  constexpr size_t count = 1 << 20;         // bf16 elements
  constexpr size_t count2 = count / 2;      // __hip_bfloat162 lanes / threads
  std::vector<__hip_bfloat162> left(count2), right(count2), out(count2);
  for (size_t index = 0; index < count2; ++index) {
    left[index] = __bfloat162bfloat162(__float2bfloat16(1.0f));
    right[index] = __bfloat162bfloat162(__float2bfloat16(1.0000001f));
  }
  __hip_bfloat162 *d_left = nullptr, *d_right = nullptr, *d_out = nullptr;
  if (hipMalloc(&d_left, count2 * sizeof(__hip_bfloat162)) || hipMalloc(&d_right, count2 * sizeof(__hip_bfloat162)) || hipMalloc(&d_out, count2 * sizeof(__hip_bfloat162))) return 2;
  if (hipMemcpy(d_left, left.data(), count2 * sizeof(__hip_bfloat162), hipMemcpyHostToDevice) != hipSuccess || hipMemcpy(d_right, right.data(), count2 * sizeof(__hip_bfloat162), hipMemcpyHostToDevice) != hipSuccess) return 2;
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
  if (hipMemcpy(out.data(), d_out, count2 * sizeof(__hip_bfloat162), hipMemcpyDeviceToHost) != hipSuccess) return 2;
  for (size_t index = 0; index < count2; ++index) {
    const float2 v = __bfloat1622float2(out[index]);
    if (!std::isfinite(v.x) || !std::isfinite(v.y) || std::fabs(v.x - 1.0f) > 0.1f || std::fabs(v.y - 1.0f) > 0.1f) return 4;
  }
  hipFree(d_left); hipFree(d_right); hipFree(d_out); hipEventDestroy(start); hipEventDestroy(stop); return 0;
}
