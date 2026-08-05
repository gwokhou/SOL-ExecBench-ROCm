#include <torch/extension.h>
#include <c10/hip/HIPStream.h>
#include <hip/hip_runtime.h>

__global__ void last_axis_sum_kernel(const float* input, float* output, int64_t rows, int64_t columns) {
    __shared__ float values[256];
    const int64_t row = blockIdx.x;
    float sum = 0.0f;
    for (int64_t column = threadIdx.x; column < columns; column += blockDim.x) sum += input[row * columns + column];
    values[threadIdx.x] = sum;
    __syncthreads();
    for (int stride = 128; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) values[threadIdx.x] += values[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0) output[row] = values[0];
}

torch::Tensor run(const torch::Tensor& input) {
    TORCH_CHECK(input.is_cuda() && input.is_contiguous(), "input must be contiguous HIP tensor");
    TORCH_CHECK(input.scalar_type() == torch::kFloat32 && input.dim() == 2, "input must be 2D float32");
    auto output = torch::empty({input.size(0)}, input.options());
    last_axis_sum_kernel<<<input.size(0), 256, 0, c10::hip::getCurrentHIPStream()>>>(input.data_ptr<float>(), output.data_ptr<float>(), input.size(0), input.size(1));
    TORCH_CHECK(hipGetLastError() == hipSuccess, "reduction launch failed");
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) { module.def("run", &run, "gfx1200 block reduction"); }
