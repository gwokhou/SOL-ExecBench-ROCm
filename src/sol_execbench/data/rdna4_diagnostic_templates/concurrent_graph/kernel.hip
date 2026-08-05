#include <torch/extension.h>
#include <c10/hip/HIPStream.h>
#include <hip/hip_runtime.h>

void require_hip(hipError_t status, const char* operation) {
    TORCH_CHECK(status == hipSuccess, operation, " failed: ", hipGetErrorString(status));
}

__global__ void concurrent_root_f32_kernel(
    const float* input,
    float* root,
    int64_t elements) {
    const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < elements) root[index] = input[index] * 0.5f;
}

__global__ void concurrent_sigmoid_f32_kernel(
    const float* root,
    float* left,
    int64_t elements) {
    const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < elements) left[index] = 1.0f / (1.0f + expf(-root[index]));
}

__global__ void concurrent_relu_f32_kernel(
    const float* root,
    float* right,
    int64_t elements) {
    const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < elements) right[index] = fmaxf(root[index], 0.0f);
}

__global__ void concurrent_join_f32_kernel(
    const float* left,
    const float* right,
    float* output,
    int64_t elements) {
    const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < elements) output[index] = left[index] + right[index];
}

torch::Tensor run(const torch::Tensor& input) {
    TORCH_CHECK(input.is_cuda() && input.is_contiguous(), "input must be a contiguous HIP tensor");
    TORCH_CHECK(input.scalar_type() == torch::kFloat32, "input must be float32");
    auto root = torch::empty_like(input);
    auto left = torch::empty_like(input);
    auto right = torch::empty_like(input);
    auto output = torch::empty_like(input);
    hipStream_t left_stream = nullptr;
    hipStream_t right_stream = nullptr;
    hipEvent_t root_ready = nullptr;
    hipEvent_t left_ready = nullptr;
    hipEvent_t right_ready = nullptr;
    require_hip(hipStreamCreateWithFlags(&left_stream, hipStreamNonBlocking), "left stream creation");
    require_hip(hipStreamCreateWithFlags(&right_stream, hipStreamNonBlocking), "right stream creation");
    require_hip(hipEventCreateWithFlags(&root_ready, hipEventDisableTiming), "root event creation");
    require_hip(hipEventCreateWithFlags(&left_ready, hipEventDisableTiming), "left event creation");
    require_hip(hipEventCreateWithFlags(&right_ready, hipEventDisableTiming), "right event creation");
    const hipStream_t caller_stream = c10::hip::getCurrentHIPStream();
    const int threads = 256;
    const int64_t elements = input.numel();
    const int blocks = (elements + threads - 1) / threads;
    concurrent_root_f32_kernel<<<blocks, threads, 0, caller_stream>>>(input.data_ptr<float>(), root.data_ptr<float>(), elements);
    require_hip(hipEventRecord(root_ready, caller_stream), "root event record");
    require_hip(hipStreamWaitEvent(left_stream, root_ready, 0), "left root wait");
    require_hip(hipStreamWaitEvent(right_stream, root_ready, 0), "right root wait");
    concurrent_sigmoid_f32_kernel<<<blocks, threads, 0, left_stream>>>(root.data_ptr<float>(), left.data_ptr<float>(), elements);
    concurrent_relu_f32_kernel<<<blocks, threads, 0, right_stream>>>(root.data_ptr<float>(), right.data_ptr<float>(), elements);
    require_hip(hipEventRecord(left_ready, left_stream), "left event record");
    require_hip(hipEventRecord(right_ready, right_stream), "right event record");
    require_hip(hipStreamWaitEvent(caller_stream, left_ready, 0), "left join wait");
    require_hip(hipStreamWaitEvent(caller_stream, right_ready, 0), "right join wait");
    concurrent_join_f32_kernel<<<blocks, threads, 0, caller_stream>>>(left.data_ptr<float>(), right.data_ptr<float>(), output.data_ptr<float>(), elements);
    require_hip(hipGetLastError(), "controlled DAG launch");
    require_hip(hipEventDestroy(root_ready), "root event destruction");
    require_hip(hipEventDestroy(left_ready), "left event destruction");
    require_hip(hipEventDestroy(right_ready), "right event destruction");
    require_hip(hipStreamDestroy(left_stream), "left stream destruction");
    require_hip(hipStreamDestroy(right_stream), "right stream destruction");
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.def("run", &run, "gfx1200 controlled concurrent FP32 DAG");
}
