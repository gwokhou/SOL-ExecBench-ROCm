#include <c10/hip/HIPStream.h>
#include <hipblas/hipblas.h>
#include <torch/extension.h>

#include <string>

namespace {

void check_tensor(const torch::Tensor& tensor, const std::string& name) {
    TORCH_CHECK(tensor.is_cuda(), name, " must be a HIP tensor");
    TORCH_CHECK(tensor.dtype() == torch::kFloat32, name, " must have float32 dtype");
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

void check_hipblas(hipblasStatus_t status, const char* call) {
    TORCH_CHECK(status == HIPBLAS_STATUS_SUCCESS, call, " failed with status ", status);
}

}  // namespace

torch::Tensor run(const torch::Tensor& a, const torch::Tensor& b) {
    check_tensor(a, "a");
    check_tensor(b, "b");
    TORCH_CHECK(a.dim() == 2, "a must be 2D");
    TORCH_CHECK(b.dim() == 2, "b must be 2D");
    TORCH_CHECK(a.size(1) == b.size(0), "a.size(1) must equal b.size(0)");

    const int64_t m = a.size(0);
    const int64_t k = a.size(1);
    const int64_t n = b.size(1);
    auto out = torch::empty({m, n}, a.options());

    hipblasHandle_t handle = nullptr;
    check_hipblas(hipblasCreate(&handle), "hipblasCreate");
    check_hipblas(
        hipblasSetStream(handle, c10::hip::getCurrentHIPStream()),
        "hipblasSetStream"
    );

    const float alpha = 1.0f;
    const float beta = 0.0f;
    // hipBLAS assumes column-major operands. Swapping A/B computes row-major A @ B.
    check_hipblas(
        hipblasSgemm(
            handle,
            HIPBLAS_OP_N,
            HIPBLAS_OP_N,
            static_cast<int>(n),
            static_cast<int>(m),
            static_cast<int>(k),
            &alpha,
            b.data_ptr<float>(),
            static_cast<int>(n),
            a.data_ptr<float>(),
            static_cast<int>(k),
            &beta,
            out.data_ptr<float>(),
            static_cast<int>(n)
        ),
        "hipblasSgemm"
    );
    check_hipblas(hipblasDestroy(handle), "hipblasDestroy");
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("run", &run, "hipBLAS-backed float32 GEMM");
}
