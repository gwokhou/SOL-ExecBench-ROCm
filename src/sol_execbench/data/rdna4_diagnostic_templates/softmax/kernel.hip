#include <torch/extension.h>
#include <c10/hip/HIPStream.h>
#include <hip/hip_runtime.h>

constexpr int kMaximumColumns = 1024;

__global__ void softmax_f32_kernel(const float* input, float* output, int64_t columns, int block_dim) {
    __shared__ float values[kMaximumColumns];
    const int64_t row = blockIdx.x;
    const int column = threadIdx.x;
    const float element = (column < columns) ? input[row * columns + column] : 0.0f;
    values[column] = (column < columns) ? element : -INFINITY;
    __syncthreads();
    for (int stride = block_dim >> 1; stride > 0; stride >>= 1) {
        if (column < stride) values[column] = fmaxf(values[column], values[column + stride]);
        __syncthreads();
    }
    const float row_maximum = values[0];
    __syncthreads();
    const float exponent = (column < columns) ? expf(element - row_maximum) : 0.0f;
    values[column] = exponent;
    __syncthreads();
    for (int stride = block_dim >> 1; stride > 0; stride >>= 1) {
        if (column < stride) values[column] += values[column + stride];
        __syncthreads();
    }
    if (column < columns) output[row * columns + column] = exponent / values[0];
}

torch::Tensor run(const torch::Tensor& input) {
    TORCH_CHECK(input.is_cuda() && input.is_contiguous(), "input must be a contiguous HIP tensor");
    TORCH_CHECK(input.scalar_type() == torch::kFloat32 && input.dim() == 2, "input must be 2D float32");
    const int64_t columns = input.size(1);
    TORCH_CHECK(columns > 0 && columns <= kMaximumColumns, "softmax width must be in [1, 1024]");
    int block_dim = 1;
    while (block_dim < columns) block_dim <<= 1;
    auto output = torch::empty_like(input);
    softmax_f32_kernel<<<input.size(0), block_dim, 0, c10::hip::getCurrentHIPStream()>>>(input.data_ptr<float>(), output.data_ptr<float>(), columns, block_dim);
    TORCH_CHECK(hipGetLastError() == hipSuccess, "softmax launch failed");
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) { module.def("run", &run, "gfx1200 block softmax"); }
