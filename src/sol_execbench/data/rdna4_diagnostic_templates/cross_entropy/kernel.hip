#include <torch/extension.h>
#include <c10/hip/HIPStream.h>
#include <hip/hip_runtime.h>

constexpr int kMaximumClasses = 1024;

__global__ void cross_entropy_f32_kernel(const float* predictions, const int64_t* targets, float* loss, int64_t rows, int64_t classes) {
    __shared__ float maximums[kMaximumClasses];
    __shared__ float sums[kMaximumClasses];
    const int64_t row = blockIdx.x;
    const int column = threadIdx.x;
    float maximum = -INFINITY;
    for (int64_t index = column; index < classes; index += blockDim.x) {
        maximum = fmaxf(maximum, predictions[row * classes + index]);
    }
    maximums[column] = maximum;
    __syncthreads();
    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (column < stride) maximums[column] = fmaxf(maximums[column], maximums[column + stride]);
        __syncthreads();
    }
    const float row_maximum = maximums[0];
    float sum = 0.0f;
    for (int64_t index = column; index < classes; index += blockDim.x) {
        sum += expf(predictions[row * classes + index] - row_maximum);
    }
    sums[column] = sum;
    __syncthreads();
    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (column < stride) sums[column] += sums[column + stride];
        __syncthreads();
    }
    if (column == 0) {
        const float row_loss = logf(sums[0]) + row_maximum - predictions[row * classes + targets[row]];
        atomicAdd(loss, row_loss);
    }
}

torch::Tensor run(const torch::Tensor& predictions, const torch::Tensor& targets) {
    TORCH_CHECK(predictions.is_cuda() && targets.is_cuda(), "inputs must be HIP tensors");
    TORCH_CHECK(predictions.is_contiguous() && targets.is_contiguous(), "inputs must be contiguous");
    TORCH_CHECK(predictions.scalar_type() == torch::kFloat32 && targets.scalar_type() == torch::kInt64, "invalid input dtypes");
    TORCH_CHECK(predictions.dim() == 2 && targets.dim() == 1 && predictions.size(0) == targets.size(0), "invalid input shapes");
    const int64_t rows = predictions.size(0);
    const int64_t classes = predictions.size(1);
    TORCH_CHECK(rows > 0 && classes > 0 && classes <= kMaximumClasses, "cross entropy requires rows > 0 and classes in [1, 1024]");
    int block_dim = 1;
    while (block_dim < classes) block_dim <<= 1;
    auto loss = torch::zeros({}, predictions.options());
    cross_entropy_f32_kernel<<<rows, block_dim, 0, c10::hip::getCurrentHIPStream()>>>(predictions.data_ptr<float>(), targets.data_ptr<int64_t>(), loss.data_ptr<float>(), rows, classes);
    TORCH_CHECK(hipGetLastError() == hipSuccess, "cross entropy launch failed");
    loss.mul_(1.0 / static_cast<float>(rows));
    return loss;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) { module.def("run", &run, "gfx1200 cross entropy"); }
