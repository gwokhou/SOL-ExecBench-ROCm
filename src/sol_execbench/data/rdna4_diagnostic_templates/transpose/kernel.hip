#include <torch/extension.h>
#include <c10/hip/HIPStream.h>
#include <hip/hip_runtime.h>

__global__ void tiled_transpose_kernel(const float* input, float* output, int64_t rows, int64_t columns) {
    __shared__ float tile[32][33];
    int64_t x = static_cast<int64_t>(blockIdx.x) * 32 + threadIdx.x;
    int64_t y = static_cast<int64_t>(blockIdx.y) * 32 + threadIdx.y;
    #pragma unroll
    for (int offset = 0; offset < 32; offset += 8) {
        if (x < columns && y + offset < rows) tile[threadIdx.y + offset][threadIdx.x] = input[(y + offset) * columns + x];
    }
    __syncthreads();
    x = static_cast<int64_t>(blockIdx.y) * 32 + threadIdx.x;
    y = static_cast<int64_t>(blockIdx.x) * 32 + threadIdx.y;
    #pragma unroll
    for (int offset = 0; offset < 32; offset += 8) {
        if (x < rows && y + offset < columns) output[(y + offset) * rows + x] = tile[threadIdx.x][threadIdx.y + offset];
    }
}

torch::Tensor run(const torch::Tensor& input) {
    TORCH_CHECK(input.is_cuda() && input.is_contiguous(), "input must be contiguous HIP tensor");
    TORCH_CHECK(input.scalar_type() == torch::kFloat32 && input.dim() == 2, "input must be 2D float32");
    auto output = torch::empty({input.size(1), input.size(0)}, input.options());
    dim3 block(32, 8);
    dim3 grid((input.size(1) + 31) / 32, (input.size(0) + 31) / 32);
    tiled_transpose_kernel<<<grid, block, 0, c10::hip::getCurrentHIPStream()>>>(input.data_ptr<float>(), output.data_ptr<float>(), input.size(0), input.size(1));
    TORCH_CHECK(hipGetLastError() == hipSuccess, "transpose launch failed");
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) { module.def("run", &run, "gfx1200 tiled transpose"); }
