#include <c10/hip/HIPStream.h>
#include <miopen/miopen.h>
#include <torch/extension.h>

#include <string>

namespace {

void check_tensor(const torch::Tensor& tensor, const std::string& name) {
    TORCH_CHECK(tensor.is_cuda(), name, " must be a HIP tensor");
    TORCH_CHECK(tensor.dtype() == torch::kFloat32, name, " must have float32 dtype");
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

void check_miopen(miopenStatus_t status, const char* call) {
    TORCH_CHECK(status == miopenStatusSuccess, call, " failed with status ", status);
}

}  // namespace

torch::Tensor run(const torch::Tensor& input) {
    check_tensor(input, "input");
    TORCH_CHECK(input.dim() == 2, "input must be 2D");
    TORCH_CHECK(input.size(1) == 4096, "input hidden dimension must be 4096");

    const int64_t batch = input.size(0);
    auto output = torch::empty_like(input);

    miopenHandle_t handle = nullptr;
    miopenTensorDescriptor_t input_desc = nullptr;
    miopenTensorDescriptor_t output_desc = nullptr;

    check_miopen(miopenCreate(&handle), "miopenCreate");
    check_miopen(
        miopenSetStream(handle, c10::hip::getCurrentHIPStream()),
        "miopenSetStream"
    );
    check_miopen(miopenCreateTensorDescriptor(&input_desc), "miopenCreateTensorDescriptor");
    check_miopen(miopenCreateTensorDescriptor(&output_desc), "miopenCreateTensorDescriptor");
    check_miopen(
        miopenSet4dTensorDescriptor(
            input_desc,
            miopenFloat,
            static_cast<int>(batch),
            4096,
            1,
            1
        ),
        "miopenSet4dTensorDescriptor(input)"
    );
    check_miopen(
        miopenSet4dTensorDescriptor(
            output_desc,
            miopenFloat,
            static_cast<int>(batch),
            4096,
            1,
            1
        ),
        "miopenSet4dTensorDescriptor(output)"
    );

    const float alpha = 1.0f;
    const float beta = 0.0f;
    check_miopen(
        miopenSoftmaxForward_V2(
            handle,
            &alpha,
            input_desc,
            input.data_ptr<float>(),
            &beta,
            output_desc,
            output.data_ptr<float>(),
            MIOPEN_SOFTMAX_ACCURATE,
            MIOPEN_SOFTMAX_MODE_INSTANCE
        ),
        "miopenSoftmaxForward_V2"
    );

    check_miopen(miopenDestroyTensorDescriptor(output_desc), "miopenDestroyTensorDescriptor(output)");
    check_miopen(miopenDestroyTensorDescriptor(input_desc), "miopenDestroyTensorDescriptor(input)");
    check_miopen(miopenDestroy(handle), "miopenDestroy");
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("run", &run, "MIOpen-backed float32 softmax over hidden_size=4096");
}
