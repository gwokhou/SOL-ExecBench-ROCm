// SPDX-License-Identifier: Apache-2.0
// Trusted diagnostic fixture: named dispatches give an independent edge oracle.
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <initializer_list>

static void checked(hipError_t result) {
  if (result != hipSuccess) {
    std::fprintf(stderr, "%s\n", hipGetErrorString(result));
    std::exit(2);
  }
}

extern "C" __global__ void produce_first(int *out) {
  auto start = clock64();
  while (clock64() - start < 2000000) {}
  out[0] = 11;
}
extern "C" __global__ void consume_first(int *out) { out[1] = out[0] + 1; }
extern "C" __global__ void consume_fanout(int *out) { out[2] = out[0] + 2; }
extern "C" __global__ void produce_second(int *out) {
  auto start = clock64();
  while (clock64() - start < 2000000) {}
  out[0] = 21;
}
extern "C" __global__ void consume_second(int *out) { out[3] = out[0] + 1; }
extern "C" __global__ void independent(int *out) { out[4] = 99; }

int main(int argc, char **argv) {
  if (argc != 2) return 2;
  const bool reuse = std::strcmp(argv[1], "reuse") == 0;
  const bool recreate = std::strcmp(argv[1], "recreate") == 0;
  const bool no_wait = std::strcmp(argv[1], "independent") == 0;
  const bool unrecorded = std::strcmp(argv[1], "unrecorded") == 0;
  if (!(reuse || recreate || no_wait || unrecorded)) return 2;
  checked(hipSetDevice(0));
  hipDeviceProp_t props;
  checked(hipGetDeviceProperties(&props, 0));
  if (std::strncmp(props.gcnArchName, "gfx1200", 7) != 0) return 2;
  int *out;
  checked(hipMalloc(&out, 5 * sizeof(int)));
  checked(hipMemset(out, 0, 5 * sizeof(int)));
  hipStream_t source, sink, fanout, separate;
  for (auto stream : {&source, &sink, &fanout, &separate})
    checked(hipStreamCreateWithFlags(stream, hipStreamNonBlocking));
  hipEvent_t event;
  checked(hipEventCreateWithFlags(&event, hipEventDisableTiming));
  checked(hipDeviceSynchronize());
  if (reuse || recreate) {
    hipLaunchKernelGGL(produce_first, dim3(1), dim3(1), 0, source, out);
    checked(hipGetLastError());
    checked(hipEventRecord(event, source));
    hipLaunchKernelGGL(independent, dim3(1), dim3(1), 0, separate, out);
    checked(hipGetLastError());
    checked(hipStreamWaitEvent(sink, event, 0));
    hipLaunchKernelGGL(consume_first, dim3(1), dim3(1), 0, sink, out);
    checked(hipGetLastError());
    checked(hipStreamWaitEvent(fanout, event, 0));
    hipLaunchKernelGGL(consume_fanout, dim3(1), dim3(1), 0, fanout, out);
    checked(hipGetLastError());
    checked(hipDeviceSynchronize());
    if (recreate) {
      checked(hipEventDestroy(event));
      checked(hipEventCreateWithFlags(&event, hipEventDisableTiming));
    }
    hipLaunchKernelGGL(produce_second, dim3(1), dim3(1), 0, source, out);
    checked(hipGetLastError());
    checked(hipEventRecord(event, source));
    checked(hipStreamWaitEvent(sink, event, 0));
    hipLaunchKernelGGL(consume_second, dim3(1), dim3(1), 0, sink, out);
    checked(hipGetLastError());
  } else {
    if (unrecorded) checked(hipStreamWaitEvent(separate, event, 0));
    hipLaunchKernelGGL(independent, dim3(1), dim3(1), 0, separate, out);
    checked(hipGetLastError());
  }
  checked(hipDeviceSynchronize());
  int host[5];
  checked(hipMemcpy(host, out, sizeof(host), hipMemcpyDeviceToHost));
  const bool correct = host[4] == 99 &&
      (!(reuse || recreate) || (host[0] == 21 && host[1] == 12 && host[2] == 13 && host[3] == 22));
  checked(hipEventDestroy(event));
  for (auto stream : {source, sink, fanout, separate}) checked(hipStreamDestroy(stream));
  checked(hipFree(out));
  std::printf("{\"case\":\"%s\",\"correct\":%s}\n", argv[1], correct ? "true" : "false");
  return correct ? 0 : 1;
}
