// SPDX-License-Identifier: Apache-2.0
// Trusted diagnostic fixture: named dispatches give an independent edge oracle.
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <initializer_list>
#include "hip_trace_scope_journal.h"

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
  if (argc != 6) return 2;
  const bool reuse = std::strcmp(argv[1], "reuse") == 0;
  const bool recreate = std::strcmp(argv[1], "recreate") == 0;
  const bool no_wait = std::strcmp(argv[1], "independent") == 0;
  const bool unrecorded = std::strcmp(argv[1], "unrecorded") == 0;
  if (!(reuse || recreate || no_wait || unrecorded)) return 2;
  checked(hipSetDevice(0));
  hipDeviceProp_t props;
  checked(hipGetDeviceProperties(&props, 0));
  if (std::strncmp(props.gcnArchName, "gfx1200", 7) != 0) return 2;
  open_journal(argv);
  int *out, *host;
  checked(hipHostMalloc(&host, 5 * sizeof(int), hipHostMallocMapped));
  checked(hipHostGetDevicePointer(reinterpret_cast<void **>(&out), host, 0));
  for (int i = 0; i < 5; ++i) host[i] = 0;
  hipStream_t source, sink, fanout, separate;
  for (auto stream : {&source, &sink, &fanout, &separate})
    checked(hipStreamCreateWithFlags(stream, hipStreamNonBlocking));
  hipEvent_t event;
  checked(hipEventCreateWithFlags(&event, hipEventDisableTiming));
  logged(hipDeviceSynchronize(), "hipDeviceSynchronize");
  if (reuse || recreate) {
    hipLaunchKernelGGL(produce_first, dim3(1), dim3(1), 0, source, out);
    logged(hipGetLastError(), "hipLaunchKernel", source, nullptr, "produce_first");
    logged(hipEventRecord(event, source), "hipEventRecord", source, event);
    hipLaunchKernelGGL(independent, dim3(1), dim3(1), 0, separate, out);
    logged(hipGetLastError(), "hipLaunchKernel", separate, nullptr, "independent");
    logged(hipStreamWaitEvent(sink, event, 0), "hipStreamWaitEvent", sink, event);
    hipLaunchKernelGGL(consume_first, dim3(1), dim3(1), 0, sink, out);
    logged(hipGetLastError(), "hipLaunchKernel", sink, nullptr, "consume_first");
    logged(hipStreamWaitEvent(fanout, event, 0), "hipStreamWaitEvent", fanout, event);
    hipLaunchKernelGGL(consume_fanout, dim3(1), dim3(1), 0, fanout, out);
    logged(hipGetLastError(), "hipLaunchKernel", fanout, nullptr, "consume_fanout");
    logged(hipDeviceSynchronize(), "hipDeviceSynchronize");
    logged(hipEventSynchronize(event), "hipEventSynchronize", nullptr, event);
    if (recreate) {
      logged(hipEventDestroy(event), "hipEventDestroy", nullptr, event);
      checked(hipEventCreateWithFlags(&event, hipEventDisableTiming));
    }
    hipLaunchKernelGGL(produce_second, dim3(1), dim3(1), 0, source, out);
    logged(hipGetLastError(), "hipLaunchKernel", source, nullptr, "produce_second");
    logged(hipEventRecord(event, source), "hipEventRecord", source, event);
    logged(hipStreamWaitEvent(sink, event, 0), "hipStreamWaitEvent", sink, event);
    hipLaunchKernelGGL(consume_second, dim3(1), dim3(1), 0, sink, out);
    logged(hipGetLastError(), "hipLaunchKernel", sink, nullptr, "consume_second");
  } else {
    if (unrecorded) logged(hipStreamWaitEvent(separate, event, 0), "hipStreamWaitEvent", separate, event);
    hipLaunchKernelGGL(independent, dim3(1), dim3(1), 0, separate, out);
    logged(hipGetLastError(), "hipLaunchKernel", separate, nullptr, "independent");
  }
  logged(hipDeviceSynchronize(), "hipDeviceSynchronize");
  const bool correct = host[4] == 99 &&
      (!(reuse || recreate) || (host[0] == 21 && host[1] == 12 && host[2] == 13 && host[3] == 22));
  logged(hipEventDestroy(event), "hipEventDestroy", nullptr, event);
  for (auto stream : {source, sink, fanout, separate}) checked(hipStreamDestroy(stream));
  checked(hipHostFree(host));
  close_journal(correct);
  std::printf("{\"case\":\"%s\",\"correct\":%s}\n", argv[1], correct ? "true" : "false");
  return correct ? 0 : 1;
}
