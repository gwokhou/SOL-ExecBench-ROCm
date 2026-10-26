// SPDX-License-Identifier: Apache-2.0
// Opt-in diagnostic collector. Never load this tool in a timing experiment.
#include <rocprofiler-sdk/rocprofiler.h>
#include <rocprofiler-sdk/registration.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <unistd.h>

namespace {
constexpr size_t buffer_bytes = 65536;
size_t max_rows = 0;
rocprofiler_context_id_t context{};
rocprofiler_buffer_id_t buffer{};
FILE* output = nullptr;
std::mutex mutex;
uint64_t sequence = 0, scope = 0, owner = 0, drops = 0;
bool opened = false, closed = false, sealed = false;

void check(rocprofiler_status_t status) {
    if(status != ROCPROFILER_STATUS_SUCCESS) {
        std::fprintf(stderr, "scope collector SDK error: %d\n", int(status));
        std::abort();
    }
}

// Caller holds mutex. A missing footer makes any interrupted write invalid.
void prefix(const char* kind) {
    if(++sequence > max_rows) std::abort();
    std::fprintf(output, "%s{\"seq\":%lu,\"kind\":\"%s\"",
                 sequence == 1 ? "" : ",\n", sequence, kind);
}

void finish_row() {
    if(std::fprintf(output, "}") < 0 || std::fflush(output)) std::abort();
}

uint64_t submission_scope(uint64_t thread) {
    if(scope && thread != owner) std::abort();
    return scope;
}

void dispatch_callback(rocprofiler_callback_tracing_record_t record,
                       rocprofiler_user_data_t* user, void*) {
    if(record.operation != ROCPROFILER_KERNEL_DISPATCH_ENQUEUE) return;
    std::lock_guard<std::mutex> lock(mutex);
    if(record.phase == ROCPROFILER_CALLBACK_PHASE_ENTER) {
        user->value = submission_scope(record.thread_id);
        return;
    }
    if(record.phase != ROCPROFILER_CALLBACK_PHASE_EXIT) return;
    auto* data = static_cast<rocprofiler_callback_tracing_kernel_dispatch_data_t*>(record.payload);
    const auto& d = data->dispatch_info;
    prefix("enqueue");
    std::fprintf(output, ",\"scope\":%lu,\"tid\":%lu,\"corr\":%lu,\"dispatch\":%lu,\"agent\":%lu,\"queue\":%lu,\"kernel\":%lu",
                 user->value, record.thread_id, record.correlation_id.internal,
                 d.dispatch_id, d.agent_id.handle, d.queue_id.handle, d.kernel_id);
    finish_row();
}

void api_callback(rocprofiler_callback_tracing_record_t record,
                  rocprofiler_user_data_t*, void*) {
    if(record.phase != ROCPROFILER_CALLBACK_PHASE_EXIT) return;
    const char* name = nullptr;
    check(rocprofiler_query_callback_tracing_kind_operation_name(record.kind, record.operation, &name, nullptr));
    if(!name || std::strchr(name, '"')) std::abort();
    auto* data = static_cast<rocprofiler_callback_tracing_hip_api_data_t*>(record.payload);
    std::lock_guard<std::mutex> lock(mutex);
    prefix("api");
    std::fprintf(output, ",\"scope\":%lu,\"tid\":%lu,\"corr\":%lu,\"name\":\"%s\",\"result\":%lu",
                 submission_scope(record.thread_id), record.thread_id,
                 record.correlation_id.internal, name, data->retval.uint64_t_retval);
    finish_row();
}

void event_callback(rocprofiler_callback_tracing_record_t record,
                    rocprofiler_user_data_t* user, void*) {
    std::lock_guard<std::mutex> lock(mutex);
    if(record.phase == ROCPROFILER_CALLBACK_PHASE_ENTER) {
        user->value = submission_scope(record.thread_id);
        return;
    }
    if(record.phase != ROCPROFILER_CALLBACK_PHASE_EXIT) return;
    auto* d = static_cast<rocprofiler_callback_tracing_hip_event_data_t*>(record.payload);
    prefix("event_enqueue");
    std::fprintf(output, ",\"scope\":%lu,\"tid\":%lu,\"corr\":%lu,\"operation\":%u,\"event\":%lu,\"agent\":%lu,\"queue\":%lu,\"source_queue\":%lu",
                 user->value, record.thread_id, record.correlation_id.internal,
                 record.operation, d->hip_event_handle, d->agent_id.handle,
                 d->queue_id.handle, d->source_queue_id.handle);
    finish_row();
}

void buffered(rocprofiler_context_id_t, rocprofiler_buffer_id_t,
              rocprofiler_record_header_t** headers, size_t count, void*, uint64_t lost) {
    std::lock_guard<std::mutex> lock(mutex);
    drops += lost;
    for(size_t i = 0; i < count; ++i) {
        auto* h = headers[i];
        if(h->category != ROCPROFILER_BUFFER_CATEGORY_TRACING) std::abort();
        if(h->kind == ROCPROFILER_BUFFER_TRACING_KERNEL_DISPATCH) {
            auto* r = static_cast<rocprofiler_buffer_tracing_kernel_dispatch_record_t*>(h->payload);
            const auto& d = r->dispatch_info;
            prefix("complete");
            std::fprintf(output, ",\"tid\":%lu,\"corr\":%lu,\"dispatch\":%lu,\"agent\":%lu,\"queue\":%lu,\"kernel\":%lu,\"start\":%lu,\"end\":%lu",
                         r->thread_id, r->correlation_id.internal, d.dispatch_id,
                         d.agent_id.handle, d.queue_id.handle, d.kernel_id,
                         r->start_timestamp, r->end_timestamp);
        } else if(h->kind == ROCPROFILER_BUFFER_TRACING_HIP_EVENT) {
            auto* r = static_cast<rocprofiler_buffer_tracing_hip_event_record_t*>(h->payload);
            prefix("event_complete");
            std::fprintf(output, ",\"tid\":%lu,\"corr\":%lu,\"operation\":%u,\"event\":%lu,\"agent\":%lu,\"queue\":%lu,\"source_queue\":%lu,\"start\":%lu,\"end\":%lu",
                         r->thread_id, r->correlation_id.internal, r->operation,
                         r->hip_event_handle, r->agent_id.handle, r->queue_id.handle,
                         r->source_queue_id.handle, r->start_timestamp, r->end_timestamp);
        } else std::abort();
        finish_row();
    }
}

int initialize(rocprofiler_client_finalize_t, void*) {
    const char* path = std::getenv("SOL_SCOPE_LEDGER");
    const char* run = std::getenv("SOL_SCOPE_RUN");
    const char* schema = std::getenv("SOL_SCOPE_SCHEMA");
    const char* limit = std::getenv("SOL_SCOPE_MAX_ROWS");
    if(!limit) std::abort();
    max_rows = std::strtoul(limit, nullptr, 10);
    if(!max_rows) std::abort();
    if(!path || !run || !schema || std::strlen(run) != 32) std::abort();
    for(const char* p = run; *p; ++p)
        if(!std::strchr("0123456789abcdef", *p)) std::abort();
    if(std::strchr(schema, '"') || std::strchr(schema, '\\')) std::abort();
    output = std::fopen(path, "wx");
    if(!output) std::abort();
    std::fprintf(output, "{\"schema_version\":\"%s\",\"run_id\":\"%s\",\"pid\":%d,\"records\":[", schema, run, getpid());
    check(rocprofiler_create_context(&context));
    check(rocprofiler_create_buffer(context, buffer_bytes, buffer_bytes / 2,
          ROCPROFILER_BUFFER_POLICY_LOSSLESS, buffered, nullptr, &buffer));
    check(rocprofiler_configure_callback_tracing_service(context,
          ROCPROFILER_CALLBACK_TRACING_KERNEL_DISPATCH, nullptr, 0, dispatch_callback, nullptr));
    check(rocprofiler_configure_callback_tracing_service(context,
          ROCPROFILER_CALLBACK_TRACING_HIP_RUNTIME_API, nullptr, 0, api_callback, nullptr));
    check(rocprofiler_configure_callback_tracing_service(context,
          ROCPROFILER_CALLBACK_TRACING_HIP_EVENT, nullptr, 0, event_callback, nullptr));
    check(rocprofiler_configure_buffer_tracing_service(context,
          ROCPROFILER_BUFFER_TRACING_KERNEL_DISPATCH, nullptr, 0, buffer));
    check(rocprofiler_configure_buffer_tracing_service(context,
          ROCPROFILER_BUFFER_TRACING_HIP_EVENT, nullptr, 0, buffer));
    check(rocprofiler_start_context(context));
    return 0;
}

void seal() {
    rocprofiler_timestamp_t now{};
    rocprofiler_thread_id_t thread{};
    check(rocprofiler_get_timestamp(&now));
    check(rocprofiler_get_thread_id(&thread));
    check(rocprofiler_stop_context(context));
    check(rocprofiler_flush_buffer(buffer));
    std::lock_guard<std::mutex> lock(mutex);
    if(!opened || !closed || scope || sealed || owner != thread) std::abort();
    prefix("seal");
    std::fprintf(output, ",\"tid\":%lu,\"timestamp\":%lu", thread, now);
    finish_row();
    std::fprintf(output, "],\"row_count\":%lu,\"drop_count\":%lu,\"lossless\":true,\"finalized\":true,\"stop_ok\":true,\"flush_ok\":true}\n", sequence, drops);
    if(std::fclose(output)) std::abort();
    output = nullptr;
    sealed = true;
}

void finalize(void*) {
    // SDK destroys client contexts before invoking this callback. The executor
    // must stop and flush explicitly while its context still exists.
    if(!sealed) std::abort();
}
}

// Called by the trusted workload executor, after/before successful device sync.
extern "C" void sol_scope_boundary(int begin) {
    rocprofiler_thread_id_t thread{};
    rocprofiler_timestamp_t now{};
    check(rocprofiler_get_thread_id(&thread));
    check(rocprofiler_get_timestamp(&now));
    std::lock_guard<std::mutex> lock(mutex);
    if(!output || (begin && opened) || (!begin && (!opened || closed || owner != thread))) std::abort();
    if(begin) { opened = true; owner = thread; scope = 1; }
    else { closed = true; scope = 0; }
    prefix(begin ? "begin" : "end");
    std::fprintf(output, ",\"tid\":%lu,\"timestamp\":%lu", thread, now);
    finish_row();
}

extern "C" void sol_scope_seal() { seal(); }

extern "C" rocprofiler_tool_configure_result_t* rocprofiler_configure(
    uint32_t, const char*, uint32_t, rocprofiler_client_id_t* id) {
    id->name = "SOLScopeAudit";
    static rocprofiler_tool_configure_result_t config{sizeof(config), initialize, finalize, nullptr};
    return &config;
}
