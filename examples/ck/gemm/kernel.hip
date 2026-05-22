#include <ATen/cuda/CUDAContext.h>
#include <ck/ck.hpp>
#include <ck/tensor_description/tensor_descriptor_helper.hpp>
#include <hip/hip_runtime.h>
#include <torch/extension.h>

#include <string>

namespace {

using CkIndex = ck::index_t;

constexpr CkIndex kTileM = 8;
constexpr CkIndex kTileN = 16;

void check_tensor(const torch::Tensor& tensor, const std::string& name) {
    TORCH_CHECK(tensor.is_cuda(), name, " must be a HIP tensor");
    TORCH_CHECK(tensor.dtype() == torch::kFloat32, name, " must have float32 dtype");
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

void check_hip(hipError_t status, const char* call) {
    TORCH_CHECK(status == hipSuccess, call, " failed with status ", hipGetErrorString(status));
}

__global__ void ck_tiled_gemm_kernel(
    const float* __restrict__ a,
    const float* __restrict__ b,
    float* __restrict__ c,
    CkIndex m
) {
    const CkIndex row = static_cast<CkIndex>(blockIdx.y) * kTileM + threadIdx.y;
    const CkIndex col = static_cast<CkIndex>(blockIdx.x) * kTileN + threadIdx.x;
    if (row >= m || col >= 128) {
        return;
    }

    float acc = 0.0f;
    for (CkIndex k = 0; k < 128; ++k) {
        acc += a[row * 128 + k] * b[k * 128 + col];
    }
    c[row * 128 + col] = acc;
}

}  // namespace

torch::Tensor run(const torch::Tensor& a, const torch::Tensor& b) {
    check_tensor(a, "A");
    check_tensor(b, "B");
    TORCH_CHECK(a.dim() == 2, "A must be 2D");
    TORCH_CHECK(b.dim() == 2, "B must be 2D");
    TORCH_CHECK(a.size(1) == 128, "A.size(1) must be 128");
    TORCH_CHECK(b.size(0) == 128, "B.size(0) must be 128");
    TORCH_CHECK(b.size(1) == 128, "B.size(1) must be 128");

    const CkIndex m = static_cast<CkIndex>(a.size(0));
    auto c = torch::empty({a.size(0), b.size(1)}, a.options());

    const dim3 block(kTileN, kTileM);
    const dim3 grid((128 + kTileN - 1) / kTileN, (m + kTileM - 1) / kTileM);
    hipLaunchKernelGGL(
        ck_tiled_gemm_kernel,
        grid,
        block,
        0,
        at::cuda::getCurrentCUDAStream(),
        a.data_ptr<float>(),
        b.data_ptr<float>(),
        c.data_ptr<float>(),
        m
    );
    check_hip(hipGetLastError(), "ck_tiled_gemm_kernel");
    return c;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("run", &run, "CK-facing native ROCm float32 GEMM for 128x128 tiles");
}
