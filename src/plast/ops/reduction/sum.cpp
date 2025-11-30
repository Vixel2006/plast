#include "plast/ops/reduction/sum.h"
#include "plast/core/device_management.h"
#include "plast/core/types.h"
#include "plast/kernels/cpu/reduction_kernels.h"
// #include "plast/kernels/cuda/reduction_kernels.h" // CUDA kernels not yet defined

#include <cstring>
#include <numeric>
#include <stdexcept>

namespace plast
{
namespace ops
{

tensor::Tensor SumOperation::execute_cpu(const std::vector<const tensor::Tensor*>& inputs) const
{
    const tensor::Tensor& input = *inputs[0];
    core::DType dtype = input.dtype();

    // Determine output shape using the operation's infer_output_shape method
    std::vector<size_t> output_shape_vec = infer_output_shape({input.shape()});

    // Allocate output tensor
    tensor::Tensor output(output_shape_vec, dtype, core::DeviceType::CPU);

    // Dispatch to type-specific C CPU kernel
    switch (dtype)
    {
    case core::DType::FLOAT32:
        if (full_reduction_)
        {
            plast_cpu_sum_full_reduction_float(input.data_as<const float>(), output.data_as<float>(),
                                         input.shape().data(), input.shape().size());
        }
        else
        {
            plast_cpu_sum_reduction_dim_float(input.data_as<const float>(), output.data_as<float>(),
                                        input.shape().data(), input.shape().size(),
                                        output.shape().data(), output.shape().size(), dim_);
        }
        break;
    case core::DType::INT32:
        if (full_reduction_)
        {
            plast_cpu_sum_full_reduction_int32(input.data_as<const int32_t>(), output.data_as<int32_t>(),
                                         input.shape().data(), input.shape().size());
        }
        else
        {
            plast_cpu_sum_reduction_dim_int32(input.data_as<const int32_t>(), output.data_as<int32_t>(),
                                        input.shape().data(), input.shape().size(),
                                        output.shape().data(), output.shape().size(), dim_);
        }
        break;
    // Add more types as needed
    default:
        throw std::runtime_error("Unsupported DType for Sum operation on CPU.");
    }

    return output;
}

tensor::Tensor SumOperation::execute_cuda(const std::vector<const tensor::Tensor*>& inputs) const
{
#ifdef PLAST_CUDA_ENABLED
    // TODO: Implement CUDA kernels for Sum operation
    throw std::runtime_error("CUDA Sum operation not yet implemented.");
#else
    throw std::runtime_error("CUDA is not enabled. Cannot execute Sum operation on CUDA device.");
#endif
}

} // namespace ops
} // namespace plast
