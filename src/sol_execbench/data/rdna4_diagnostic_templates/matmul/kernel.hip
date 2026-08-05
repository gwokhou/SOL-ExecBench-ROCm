#include <torch/extension.h>
#include <c10/hip/HIPStream.h>
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

__global__ void naive_matmul_kernel(const half* a, const half* b, half* output, int64_t m, int64_t n, int64_t k) {
    const int64_t column = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t row = static_cast<int64_t>(blockIdx.y) * blockDim.y + threadIdx.y;
    if (row >= m || column >= n) return;
    float sum = 0.0f;
    for (int64_t inner = 0; inner < k; ++inner) sum += __half2float(a[row * k + inner]) * __half2float(b[inner * n + column]);
    output[row * n + column] = __float2half(sum);
}

torch::Tensor run(const torch::Tensor& a, const torch::Tensor& b) {
    TORCH_CHECK(a.is_cuda() && b.is_cuda() && a.is_contiguous() && b.is_contiguous(), "inputs must be contiguous HIP tensors");
    TORCH_CHECK(a.scalar_type() == torch::kFloat16 && b.scalar_type() == torch::kFloat16, "inputs must be float16");
    TORCH_CHECK(a.dim() == 2 && b.dim() == 2 && a.size(1) == b.size(0), "invalid matmul shapes");
    auto output = torch::empty({a.size(0), b.size(1)}, a.options());
    dim3 block(16, 16);
    dim3 grid((b.size(1) + 15) / 16, (a.size(0) + 15) / 16);
    naive_matmul_kernel<<<grid, block, 0, c10::hip::getCurrentHIPStream()>>>(reinterpret_cast<const half*>(a.data_ptr<at::Half>()), reinterpret_cast<const half*>(b.data_ptr<at::Half>()), reinterpret_cast<half*>(output.data_ptr<at::Half>()), a.size(0), b.size(1), a.size(1));
    TORCH_CHECK(hipGetLastError() == hipSuccess, "matmul launch failed");
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) { module.def("run", &run, "gfx1200 naive FP16 matmul"); }
