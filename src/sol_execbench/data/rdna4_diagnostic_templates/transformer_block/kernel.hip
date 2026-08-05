#include <torch/extension.h>
#include <c10/hip/HIPStream.h>
#include <hip/hip_runtime.h>

constexpr int kHeads = 8;
constexpr int kChannels = 768;
constexpr int kMaximumSequence = 1024;

__global__ void transformer_block_f32_kernel(
    const float* input,
    const float* weight,
    float* output,
    int sequence) {
    const int column = threadIdx.x;
    for (int row = 0; row < sequence; ++row) {
        float value = 0.0f;
        for (int inner = 0; inner < kChannels; ++inner) {
            value += input[static_cast<int64_t>(row) * kChannels + inner]
                * weight[static_cast<int64_t>(inner) * kChannels + column];
        }
        output[static_cast<int64_t>(row) * kChannels + column] = value;
    }
    __syncthreads();
    float maximum = output[column];
    for (int row = 1; row < sequence; ++row) {
        maximum = fmaxf(maximum, output[static_cast<int64_t>(row) * kChannels + column]);
    }
    float total = 0.0f;
    for (int row = 0; row < sequence; ++row) {
        float* value = output + static_cast<int64_t>(row) * kChannels + column;
        *value = expf(*value - maximum);
        total += *value;
    }
    const float softmax_inverse = 1.0f / total;
    for (int row = 0; row < sequence; ++row) {
        output[static_cast<int64_t>(row) * kChannels + column] *= softmax_inverse;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        for (int row = 0; row < sequence; ++row) {
            float* row_output = output + static_cast<int64_t>(row) * kChannels;
            float mean = 0.0f;
            for (int index = 0; index < kChannels; ++index) mean += row_output[index];
            mean /= static_cast<float>(kChannels);
            float variance = 0.0f;
            for (int index = 0; index < kChannels; ++index) {
                const float centered = row_output[index] - mean;
                variance += centered * centered;
            }
            variance /= static_cast<float>(kChannels);
            const float norm_inverse = rsqrtf(variance + 1.0e-5f);
            for (int index = 0; index < kChannels; ++index) {
                row_output[index] = (row_output[index] - mean) * norm_inverse;
            }
        }
    }
}

torch::Tensor run(const torch::Tensor& input, const torch::Tensor& weight) {
    TORCH_CHECK(input.is_cuda() && weight.is_cuda(), "inputs must be HIP tensors");
    TORCH_CHECK(input.is_contiguous() && weight.is_contiguous(), "inputs must be contiguous");
    TORCH_CHECK(input.scalar_type() == torch::kFloat32 && weight.scalar_type() == torch::kFloat32, "inputs must be float32");
    TORCH_CHECK(input.dim() == 2 && weight.dim() == 2, "inputs must be matrices");
    TORCH_CHECK(input.size(1) == kChannels && weight.size(0) == kChannels && weight.size(1) == kChannels, "MiniGPT smoke requires C=768");
    TORCH_CHECK(input.size(0) > 0 && input.size(0) <= kMaximumSequence, "sequence length must be in [1, 1024]");
    const int sequence = input.size(0);
    auto output = torch::empty_like(input);
    const hipStream_t stream = c10::hip::getCurrentHIPStream();
    transformer_block_f32_kernel<<<1, kChannels, 0, stream>>>(input.data_ptr<float>(), weight.data_ptr<float>(), output.data_ptr<float>(), sequence);
    TORCH_CHECK(hipGetLastError() == hipSuccess, "MiniGPT block launch failed");
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.def("run", &run, "gfx1200 FP32 MiniGPT transformer block");
}
