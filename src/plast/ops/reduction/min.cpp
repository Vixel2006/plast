#include "plast/ops/reduction/min.h"
#include "plast/core/device_management.h"
#include "plast/core/shape_utils_cpp.h" // Added for strided operations
#include "plast/core/types.h"
#include "plast/kernels/cpu/reduction_kernels.h"

#include <cstring>
#include <numeric>
#include <stdexcept>

namespace plast
{
namespace ops
{

tensor::Tensor MinOperation::execute_cpu(const std::vector<const tensor::Tensor*>& inputs) const
{
    const tensor::Tensor& input = *inputs[0];
    core::DType dtype = input.dtype();

    // Determine output shape using the operation's infer_output_shape method
    std::vector<size_t> output_shape_vec = infer_output_shape({input.shape()});

    // Allocate output tensor
    tensor::Tensor output(output_shape_vec, dtype, core::DeviceType::CPU);

    bool input_contiguous = input.is_contiguous();

    size_t* input_strides = nullptr;
    size_t* output_shape_arr = nullptr;

    if (!input_contiguous)
    {
        std::vector<size_t> input_strides_vec =
            core::get_effective_broadcast_strides(input.shape(), input.strides(), input.shape());
        input_strides = new size_t[input_strides_vec.size()];
        for (size_t i = 0; i < input_strides_vec.size(); ++i)
        {
            input_strides[i] = input_strides_vec[i];
        }

        output_shape_arr = new size_t[output.ndim()];
        for (size_t i = 0; i < output.ndim(); ++i)
        {
            output_shape_arr[i] = output.shape()[i];
        }
    }

    // Dispatch to type-specific C CPU kernel
    switch (dtype)
    {
    case core::DType::FLOAT32:
        if (full_reduction_)
        {
            if (input_contiguous)
            {
                plast_cpu_min_full_reduction_float(input.data_as<const float>(),
                                                   output.data_as<float>(), input.shape().data(),
                                                   input.shape().size());
            }
            else
            {
                plast_cpu_min_full_reduction_strided_float(
                    input.data_as<const float>(), output.data_as<float>(), input.shape().data(),
                    input.shape().size(), input_strides);
            }
        }
        else
        {
            if (input_contiguous)
            {
                plast_cpu_min_reduction_dim_float(input.data_as<const float>(),
                                                  output.data_as<float>(), input.shape().data(),
                                                  input.shape().size(), output.shape().data(),
                                                  output.shape().size(), dim_);
            }
            else
            {
                plast_cpu_min_reduction_dim_strided_float(
                    input.data_as<const float>(), output.data_as<float>(), input.shape().data(),
                    input.shape().size(), input_strides, output_shape_arr, output.ndim(), dim_);
            }
        }
        break;
    case core::DType::INT32:
        if (full_reduction_)
        {
            if (input_contiguous)
            {
                plast_cpu_min_full_reduction_int32(input.data_as<const int32_t>(),
                                                   output.data_as<int32_t>(), input.shape().data(),
                                                   input.shape().size());
            }
            else
            {
                plast_cpu_min_full_reduction_strided_int32(
                    input.data_as<const int32_t>(), output.data_as<int32_t>(), input.shape().data(),
                    input.shape().size(), input_strides);
            }
        }
        else
        {
            if (input_contiguous)
            {
                plast_cpu_min_reduction_dim_int32(
                    input.data_as<const int32_t>(), output.data_as<int32_t>(), input.shape().data(),
                    input.shape().size(), output.shape().data(), output.shape().size(), dim_);
            }
            else
            {
                plast_cpu_min_reduction_dim_strided_int32(
                    input.data_as<const int32_t>(), output.data_as<int32_t>(), input.shape().data(),
                    input.shape().size(), input_strides, output_shape_arr, output.ndim(), dim_);
            }
        }
        break;
    // Add more types as needed
    default:
        if (!input_contiguous)
        {
            delete[] input_strides;
            delete[] output_shape_arr;
        }
        throw std::runtime_error("Unsupported DType for Min operation on CPU.");
    }

    if (!input_contiguous)
    {
        delete[] input_strides;
        delete[] output_shape_arr;
    }

    return output;
}

tensor::Tensor MinOperation::execute_cuda(const std::vector<const tensor::Tensor*>& inputs) const
{
#ifdef PLAST_CUDA_ENABLED
    // TODO: Implement CUDA kernels for Min operation
    throw std::runtime_error("CUDA Min operation not yet implemented.");
#else
    throw std::runtime_error("CUDA is not enabled. Cannot execute Min operation on CUDA device.");
#endif
}

} // namespace ops
} // namespace plast
