#include "plast/ops/reduction/sum.h"
#include "plast/core/device_management.h"
#include "plast/core/shape_utils_cpp.h" // Added for strided operations
#include "plast/core/types.h"
#include "plast/kernels/cpu/reduction_kernels.h"
#include "plast/kernels/cuda/reduction_kernels.h"
#ifdef PLAST_CUDA_ENABLED
#include "plast/kernels/cuda/reduction_kernels.h" // New include for CUDA kernels
#endif

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

    bool input_contiguous = input.is_contiguous();

    if (!input_contiguous)
    {
        throw std::runtime_error(
            "Sum operation on CPU does not yet support non-contiguous inputs.");
    }

    // Dispatch to type-specific C CPU kernel
    switch (dtype)
    {
    case core::DType::FLOAT32:
        if (full_reduction_)
        {
            plast_cpu_sum_full_reduction_float(input.data_as<const float>(),
                                               output.data_as<float>(), input.shape().data(),
                                               input.shape().size());
        }
        else
        {
            plast_cpu_sum_reduction_dim_float(
                input.data_as<const float>(), output.data_as<float>(), input.shape().data(),
                input.shape().size(), output.shape().data(), output.shape().size(), dim_);
        }
        break;
    case core::DType::INT32:
        if (full_reduction_)
        {
            plast_cpu_sum_full_reduction_int32(input.data_as<const int32_t>(),
                                               output.data_as<int32_t>(), input.shape().data(),
                                               input.shape().size());
        }
        else
        {
            plast_cpu_sum_reduction_dim_int32(
                input.data_as<const int32_t>(), output.data_as<int32_t>(), input.shape().data(),
                input.shape().size(), output.shape().data(), output.shape().size(), dim_);
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
    const tensor::Tensor& input = *inputs[0];
    core::DType dtype = input.dtype();

    // Determine output shape using the operation's infer_output_shape method
    std::vector<size_t> output_shape_vec = infer_output_shape({input.shape()});

    // Allocate output tensor on CUDA device
    tensor::Tensor output(output_shape_vec, dtype, core::DeviceType::CUDA);

    bool input_contiguous = input.is_contiguous();

    if (!input_contiguous)
    {
        throw std::runtime_error(
            "Sum operation on CUDA does not yet support non-contiguous inputs.");
    }

    // Dispatch to type-specific CUDA kernel
    switch (dtype)
    {
    case core::DType::FLOAT32:
        if (full_reduction_)
        {
            plast_cuda_sum_full_reduction_float(input.data_as<const float>(),
                                                output.data_as<float>(), input.shape().data(),
                                                input.shape().size());
        }
        else
        {
            throw std::runtime_error(
                "CUDA kernel for contiguous dim sum reduction (float) not implemented.");
        }
        break;
    case core::DType::INT32:
        if (full_reduction_)
        {
            throw std::runtime_error(
                "CUDA kernel for contiguous full sum reduction (int32) not implemented.");
        }
        else
        {
            throw std::runtime_error(
                "CUDA kernel for contiguous dim sum reduction (int32) not implemented.");
        }
        break;
    // Add more types as needed
    default:
        throw std::runtime_error("Unsupported DType for Sum operation on CUDA.");
    }

    return output;
#else
    throw std::runtime_error("CUDA is not enabled. Cannot execute Sum operation on CUDA device.");
#endif
}

} // namespace ops
} // namespace plast
