#include <ATen/cuda/CUDAContext.h>
#include <hip/hip_runtime.h>
#include <rocwmma/rocwmma.hpp>
#include <torch/extension.h>

#include <string>

namespace {

constexpr uint32_t kBlockM = 16;
constexpr uint32_t kBlockN = 16;
constexpr uint32_t kBlockK = 16;

void check_tensor(const torch::Tensor& tensor, const std::string& name) {
    TORCH_CHECK(tensor.is_cuda(), name, " must be a HIP tensor");
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

void check_hip(hipError_t status, const char* call) {
    TORCH_CHECK(status == hipSuccess, call, " failed with status ", hipGetErrorString(status));
}

__global__ void rocwmma_gemm_kernel(
    const rocwmma::float16_t* __restrict__ a,
    const rocwmma::float16_t* __restrict__ b,
    float* __restrict__ c,
    uint32_t m
) {
    using FragA = rocwmma::fragment<
        rocwmma::matrix_a,
        kBlockM,
        kBlockN,
        kBlockK,
        rocwmma::float16_t,
        rocwmma::row_major>;
    using FragB = rocwmma::fragment<
        rocwmma::matrix_b,
        kBlockM,
        kBlockN,
        kBlockK,
        rocwmma::float16_t,
        rocwmma::row_major>;
    using FragC = rocwmma::fragment<
        rocwmma::accumulator,
        kBlockM,
        kBlockN,
        kBlockK,
        float>;

    const uint32_t row = blockIdx.y * kBlockM;
    const uint32_t col = blockIdx.x * kBlockN;
    if (row >= m || col >= kBlockN) {
        return;
    }

    FragA frag_a;
    FragB frag_b;
    FragC frag_c;
    rocwmma::fill_fragment(frag_c, 0.0f);
    rocwmma::load_matrix_sync(frag_a, a + row * kBlockK, kBlockK);
    rocwmma::load_matrix_sync(frag_b, b + col, kBlockN);
    rocwmma::mma_sync(frag_c, frag_a, frag_b, frag_c);
    rocwmma::store_matrix_sync(c + row * kBlockN + col, frag_c, kBlockN, rocwmma::mem_row_major);
}

}  // namespace

torch::Tensor run(const torch::Tensor& a, const torch::Tensor& b) {
    check_tensor(a, "A");
    check_tensor(b, "B");
    TORCH_CHECK(a.dtype() == torch::kFloat16, "A must have float16 dtype");
    TORCH_CHECK(b.dtype() == torch::kFloat16, "B must have float16 dtype");
    TORCH_CHECK(a.dim() == 2, "A must be 2D");
    TORCH_CHECK(b.dim() == 2, "B must be 2D");
    TORCH_CHECK(a.size(0) % kBlockM == 0, "A.size(0) must be a multiple of 16");
    TORCH_CHECK(a.size(1) == kBlockK, "A.size(1) must be 16");
    TORCH_CHECK(b.size(0) == kBlockK, "B.size(0) must be 16");
    TORCH_CHECK(b.size(1) == kBlockN, "B.size(1) must be 16");

    auto c = torch::empty({a.size(0), b.size(1)}, a.options().dtype(torch::kFloat32));
    const dim3 block(32);
    const dim3 grid(1, static_cast<uint32_t>(a.size(0)) / kBlockM);
    hipLaunchKernelGGL(
        rocwmma_gemm_kernel,
        grid,
        block,
        0,
        at::cuda::getCurrentCUDAStream(),
        reinterpret_cast<const rocwmma::float16_t*>(a.data_ptr<at::Half>()),
        reinterpret_cast<const rocwmma::float16_t*>(b.data_ptr<at::Half>()),
        c.data_ptr<float>(),
        static_cast<uint32_t>(a.size(0))
    );
    check_hip(hipGetLastError(), "rocwmma_gemm_kernel");
    return c;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("run", &run, "rocWMMA FP16-to-FP32 16x16x16 GEMM");
}
