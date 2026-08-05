#include <torch/extension.h>
#include <c10/hip/HIPStream.h>
#include <hip/hip_runtime.h>

__global__ void index_add_f32_kernel(
    float* output,
    const int64_t* indices,
    const float* updates,
    int64_t update_rows,
    int64_t columns) {
    const int64_t linear = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t elements = update_rows * columns;
    if (linear >= elements) return;
    const int64_t update_row = linear / columns;
    const int64_t column = linear - update_row * columns;
    const int64_t output_row = indices[update_row];
    atomicAdd(output + output_row * columns + column, updates[linear]);
}

torch::Tensor run(
    const torch::Tensor& base,
    const torch::Tensor& indices,
    const torch::Tensor& updates) {
    TORCH_CHECK(base.is_cuda() && indices.is_cuda() && updates.is_cuda(), "inputs must be HIP tensors");
    TORCH_CHECK(base.scalar_type() == torch::kFloat32 && updates.scalar_type() == torch::kFloat32, "payloads must be float32");
    TORCH_CHECK(indices.scalar_type() == torch::kInt64, "indices must be int64");
    TORCH_CHECK(base.dim() == 2 && updates.dim() == 2 && indices.dim() == 1, "invalid input ranks");
    TORCH_CHECK(base.is_contiguous() && indices.is_contiguous() && updates.is_contiguous(), "inputs must be contiguous");
    TORCH_CHECK(indices.numel() == updates.size(0) && base.size(1) == updates.size(1), "input shapes are incompatible");
    auto output = base.clone();
    const int threads = 256;
    const int64_t elements = updates.numel();
    index_add_f32_kernel<<<(elements + threads - 1) / threads, threads, 0, c10::hip::getCurrentHIPStream()>>>(
        output.data_ptr<float>(), indices.data_ptr<int64_t>(), updates.data_ptr<float>(), updates.size(0), updates.size(1));
    TORCH_CHECK(hipGetLastError() == hipSuccess, "index-add launch failed");
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.def("run", &run, "gfx1200 FP32 atomic index-add");
}
