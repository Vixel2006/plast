#include "plast/ops/binary/sub.h"
#include "plast/core/types.h"
#include "plast/core/device_management.h" // For plast_CUDA_CHECK
#include "plast/kernels/cpu/binary_kernels.h"
#include "plast/kernels/cuda/binary_kernels.h"

#include <stdexcept>
#include <numeric> // For std::accumulate
#include <cstring> // For std::memcpy

namespace plast {
namespace ops {

tensor::Tensor SubOperation::execute_cpu(const std::vector<const tensor::Tensor*>& inputs) const {
    const tensor::Tensor& lhs = *inputs[0];
    const tensor::Tensor& rhs = *inputs[1];

    if (lhs.dtype() != rhs.dtype()) {
        throw std::runtime_error("DType mismatch for Sub operation on CPU.");
    }
    if (lhs.shape() != rhs.shape()) {
        throw std::runtime_error("Shape mismatch for Sub operation on CPU (broadcasting not yet implemented).");
    }

    size_t num_elements = lhs.num_elements();
    core::DType dtype = lhs.dtype();

    // Allocate output tensor
    tensor::Tensor output(lhs.shape(), dtype, core::DeviceType::CPU);

    // Dispatch to type-specific C CPU kernel
    switch (dtype) {
        case core::DType::FLOAT32:
            plast_cpu_sub_kernel_float(output.data_as<float>(), lhs.data_as<const float>(), rhs.data_as<const float>(), num_elements);
            break;
        case core::DType::INT32:
            plast_cpu_sub_kernel_int32(output.data_as<int32_t>(), lhs.data_as<const int32_t>(), rhs.data_as<const int32_t>(), num_elements);
            break;
        // Add more types as needed
        default:
            throw std::runtime_error("Unsupported DType for Sub operation on CPU.");
    }

    return output;
}

tensor::Tensor SubOperation::execute_cuda(const std::vector<const tensor::Tensor*>& inputs) const {
#ifdef PLAST_CUDA_ENABLED
    const tensor::Tensor& lhs = *inputs[0];
    const tensor::Tensor& rhs = *inputs[1];

    if (lhs.dtype() != rhs.dtype()) {
        throw std::runtime_error("DType mismatch for Sub operation on CUDA.");
    }
    if (lhs.shape() != rhs.shape()) {
        throw std::runtime_error("Shape mismatch for Sub operation on CUDA (broadcasting not yet implemented).");
    }

    size_t num_elements = lhs.num_elements();
    core::DType dtype = lhs.dtype();

    // Allocate output tensor on CUDA device
    tensor::Tensor output(lhs.shape(), dtype, core::DeviceType::CUDA);

    // Dispatch to type-specific CUDA kernel
    switch (dtype) {
        case core::DType::FLOAT32:
            plast_cuda_sub_kernel_float(output.data_as<float>(), lhs.data_as<const float>(), rhs.data_as<const float>(), num_elements);
            break;
        case core::DType::INT32:
            plast_cuda_sub_kernel_int32(output.data_as<int32_t>(), lhs.data_as<const int32_t>(), rhs.data_as<const int32_t>(), num_elements);
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
