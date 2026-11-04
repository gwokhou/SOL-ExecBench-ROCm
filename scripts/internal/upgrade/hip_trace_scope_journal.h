// SPDX-License-Identifier: Apache-2.0
// Independent application-side journal; never reads profiler output.
#pragma once
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cctype>
#include <unistd.h>
#include <sys/syscall.h>

static FILE *journal;
static bool first_row = true;

static void open_journal(char **argv) {
  hipUUID uuid;
  if (hipDeviceGetUuid(&uuid, 0) != hipSuccess) std::exit(2);
  char hex[33];
  for (int i = 0; i < 16; ++i)
    if (!std::isxdigit(static_cast<unsigned char>(uuid.bytes[i]))) std::exit(2);
  std::snprintf(hex, sizeof(hex), "GPU-%.*s", 16, uuid.bytes);
  journal = std::fopen(argv[2], "wx");
  if (!journal) std::exit(2);
  std::fprintf(journal,
    "{\"schema_version\":\"%s\",\"run_id\":\"%s\",\"workload_id\":\"%s\","
    "\"candidate_sha256\":\"%s\",\"process_id\":%d,\"thread_id\":%ld,"
    "\"device_uuid\":\"%s\",\"calls\":[", argv[5], argv[3], argv[1], argv[4],
    getpid(), syscall(SYS_gettid), hex);
}

static void logged(hipError_t result, const char *name,
                   hipStream_t stream = nullptr, hipEvent_t event = nullptr,
                   const char *kernel = "") {
  std::fprintf(journal,
    "%s{\"name\":\"%s\",\"stream_handle\":%llu,\"event_handle\":%llu,"
    "\"kernel_name\":\"%s\",\"result\":%d}", first_row ? "" : ",", name,
    static_cast<unsigned long long>(reinterpret_cast<uintptr_t>(stream)),
    static_cast<unsigned long long>(reinterpret_cast<uintptr_t>(event)), kernel,
    static_cast<int>(result));
  first_row = false;
  if (std::fflush(journal) != 0 || result != hipSuccess) std::exit(2);
}

static void close_journal(bool correct) {
  std::fprintf(journal, "],\"completed\":true,\"correct\":%s}\n", correct ? "true" : "false");
  if (std::fclose(journal) != 0) std::exit(2);
}
