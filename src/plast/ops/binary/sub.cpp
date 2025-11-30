#include "plast/ops/binary/sub.h"
#include "plast/core/device_management.h"
#include "plast/core/shape_utils_cpp.h" // Added for broadcasting and strides
#include "plast/core/types.h"
#include "plast/kernels/cpu/binary_kernels.h"
#include "plast/kernels/cuda/binary_kernels.h"

#include <cstring>
#include <numeric>
#include <stdexcept>

namespace plast
{
namespace ops
{

tensor::Tensor SubOperation::execute_cpu(const std::vector<const tensor::Tensor*>& inputs) const
{
    const tensor::Tensor& lhs = *inputs[0];
    const tensor::Tensor& rhs = *inputs[1];

    if (lhs.dtype() != rhs.dtype())
    {
        throw std::runtime_error("DType mismatch for Sub operation on CPU.");
    }

    core::DType dtype = lhs.dtype();

    // 1. Determine output shape and strides based on broadcasting rules
    std::vector<size_t> output_shape_vec = core::broadcast_shapes(lhs.shape(), rhs.shape());

    // Convert output_shape_vec to size_t*
    size_t* output_shape = new size_t[output_shape_vec.size()];
    for (size_t i = 0; i < output_shape_vec.size(); ++i)
    {
        output_shape[i] = output_shape_vec[i];
    }
    size_t output_ndim = output_shape_vec.size();

    // Allocate output tensor
    tensor::Tensor output(output_shape_vec, dtype, core::DeviceType::CPU);

    // 2. Compute strides for lhs and rhs based on the broadcasted output shape
    std::vector<size_t> lhs_strides_vec = core::compute_strides(lhs.shape(), output_shape_vec);
    std::vector<size_t> rhs_strides_vec = core::compute_strides(rhs.shape(), output_shape_vec);

    size_t* lhs_strides = new size_t[lhs_strides_vec.size()];
    for (size_t i = 0; i < lhs_strides_vec.size(); ++i)
    {
        lhs_strides[i] = lhs_strides_vec[i];
    }

    size_t* rhs_strides = new size_t[rhs_strides_vec.size()];
    for (size_t i = 0; i < rhs_strides_vec.size(); ++i)
    {
        rhs_strides[i] = rhs_strides_vec[i];
    }

    // 3. Check if we can use the optimized contiguous kernels
    bool lhs_contiguous_and_matches = lhs.is_contiguous() && (lhs.shape() == output_shape_vec);
    bool rhs_contiguous_and_matches = rhs.is_contiguous() && (rhs.shape() == output_shape_vec);

    if (lhs_contiguous_and_matches && rhs_contiguous_and_matches)
    {
        // Both inputs are contiguous and match the output shape, use simple element-wise kernel
        size_t num_elements = output.num_elements();
        switch (dtype)
        {
        case core::DType::FLOAT32:
            plast_cpu_sub_kernel_float(output.data_as<float>(), lhs.data_as<const float>(),
                                       rhs.data_as<const float>(), num_elements);
            break;
        case core::DType::INT32:
            plast_cpu_sub_kernel_int32(output.data_as<int32_t>(), lhs.data_as<const int32_t>(),
                                       rhs.data_as<const int32_t>(), num_elements);
            break;
        default:
            delete[] output_shape;
            delete[] lhs_strides;
            delete[] rhs_strides;
            throw std::runtime_error("Unsupported DType for Sub operation on CPU.");
        }
    }
    else
    {
        // Use strided kernels for non-contiguous or broadcasted inputs
        switch (dtype)
        {
        case core::DType::FLOAT32:
            plast_cpu_sub_kernel_strided_float(output.data_as<float>(), lhs.data_as<const float>(),
                                               rhs.data_as<const float>(), output_shape,
                                               output_ndim, lhs_strides, rhs_strides);
            break;
        case core::DType::INT32:
            plast_cpu_sub_kernel_strided_int32(output.data_as<int32_t>(), lhs.data_as<const int32_t>(),
                                               rhs.data_as<const int32_t>(), output_shape,
                                               output_ndim, lhs_strides, rhs_strides);
            break;
        default:
            delete[] output_shape;
            delete[] lhs_strides;
            delete[] rhs_strides;
            throw std::runtime_error("Unsupported DType for Sub operation on CPU.");
        }
    }

    delete[] output_shape;
    delete[] lhs_strides;
    delete[] rhs_strides;

    return output;
}

tensor::Tensor SubOperation::execute_cuda(const std::vector<const tensor::Tensor*>& inputs) const
{
#ifdef PLAST_CUDA_ENABLED
    const tensor::Tensor& lhs = *inputs[0];
    const tensor::Tensor& rhs = *inputs[1];

    if (lhs.dtype() != rhs.dtype())
    {
        throw std::runtime_error("DType mismatch for Sub operation on CUDA.");
    }
    if (lhs.shape() != rhs.shape())
    {
        throw std::runtime_error(
            "Shape mismatch for Sub operation on CUDA (broadcasting not yet implemented).");
    }

    size_t num_elements = lhs.num_elements();
    core::DType dtype = lhs.dtype();

    // Allocate output tensor on CUDA device
    tensor::Tensor output(lhs.shape(), dtype, core::DeviceType::CUDA);

    // Dispatch to type-specific CUDA kernel
    switch (dtype)
    {
    case core::DType::FLOAT32:
        plast_cuda_sub_kernel_float(output.data_as<float>(), lhs.data_as<const float>(),
                                    rhs.data_as<const float>(), num_elements);
        break;
    case core::DType::INT32:
        plast_cuda_sub_kernel_int32(output.data_as<int32_t>(), lhs.data_as<const int32_t>(),
                                    rhs.data_as<const int32_t>(), num_elements);
        break;
    // Add more types as needed
    default:
        throw std::runtime_error("Unsupported DType for Sub operation on CUDA.");
    }

    return output;
#else
    throw std::runtime_error("CUDA is not enabled. Cannot execute Sub operation on CUDA device.");
#endif
}

} // namespace ops
} // namespace plast
