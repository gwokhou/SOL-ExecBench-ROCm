#include <torch/extension.h>
#include <c10/hip/HIPStream.h>
#include <hip/hip_runtime.h>

__global__ void scaled_sigmoid_kernel(const float* input, float scale, float output_scale, float* output, int64_t elements) {
    const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < elements) output[index] = output_scale / (1.0f + expf(-scale * input[index]));
}

torch::Tensor run(const torch::Tensor& input, double scale, double output_scale) {
    TORCH_CHECK(input.is_cuda() && input.is_contiguous(), "input must be a contiguous HIP tensor");
    TORCH_CHECK(input.scalar_type() == torch::kFloat32, "input must be float32");
    auto output = torch::empty_like(input);
    const int threads = 256;
    const int64_t elements = input.numel();
    scaled_sigmoid_kernel<<<(elements + threads - 1) / threads, threads, 0, c10::hip::getCurrentHIPStream()>>>(input.data_ptr<float>(), static_cast<float>(scale), static_cast<float>(output_scale), output.data_ptr<float>(), elements);
    TORCH_CHECK(hipGetLastError() == hipSuccess, "scaled sigmoid launch failed");
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) { module.def("run", &run, "gfx1200 scaled sigmoid"); }
