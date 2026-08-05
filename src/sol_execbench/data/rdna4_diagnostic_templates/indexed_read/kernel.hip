#include <torch/extension.h>
#include <c10/hip/HIPStream.h>
#include <hip/hip_runtime.h>

__global__ void index_select_f32_kernel(
    const float* source,
    const int64_t* indices,
    float* output,
    int64_t rows,
    int64_t columns) {
    const int64_t linear = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t elements = rows * columns;
    if (linear >= elements) return;
    const int64_t output_row = linear / columns;
    const int64_t column = linear - output_row * columns;
    const int64_t source_row = indices[output_row];
    output[linear] = source[source_row * columns + column];
}

torch::Tensor run(const torch::Tensor& source, const torch::Tensor& indices) {
    TORCH_CHECK(source.is_cuda() && indices.is_cuda(), "inputs must be HIP tensors");
    TORCH_CHECK(source.scalar_type() == torch::kFloat32 && source.dim() == 2, "source must be 2D float32");
    TORCH_CHECK(indices.scalar_type() == torch::kInt64 && indices.dim() == 1, "indices must be 1D int64");
    TORCH_CHECK(source.is_contiguous() && indices.is_contiguous(), "inputs must be contiguous");
    auto output = torch::empty({indices.numel(), source.size(1)}, source.options());
    const int threads = 256;
    const int64_t elements = output.numel();
    index_select_f32_kernel<<<(elements + threads - 1) / threads, threads, 0, c10::hip::getCurrentHIPStream()>>>(
        source.data_ptr<float>(), indices.data_ptr<int64_t>(), output.data_ptr<float>(), indices.numel(), source.size(1));
    TORCH_CHECK(hipGetLastError() == hipSuccess, "index-select launch failed");
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.def("run", &run, "gfx1200 FP32 index-select");
}
