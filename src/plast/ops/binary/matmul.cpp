#include "plast/ops/binary/matmul.h"
#include "plast/core/device_management.h"
#include "plast/core/shape_utils_cpp.h" // Added for broadcasting and strides
#include "plast/kernels/cpu/binary_kernels.h"
#include "plast/kernels/cuda/binary_kernels.h"

#include <cstring>
#include <numeric>
#include <stdexcept>

#ifdef PLAST_CUDA_ENABLED
#endif

namespace plast
{

namespace ops
{

tensor::Tensor MatmulOperation::execute_cpu(const std::vector<const tensor::Tensor*>& inputs) const
{
    const tensor::Tensor& lhs = *inputs[0];
    const tensor::Tensor& rhs = *inputs[1];

    if (lhs.dtype() != rhs.dtype())
    {
        throw std::runtime_error("DType mismatch for Matmul operation on cpu");
    }

    core::DType dtype = lhs.dtype();

    std::vector<std::vector<size_t>> input_shapes_vec = {lhs.shape(), rhs.shape()};

    std::vector<size_t> output_shape_vec = infer_output_shape(input_shapes_vec);

    // Allocate output tensor
    tensor::Tensor output(output_shape_vec, dtype, core::DeviceType::CPU);

    // Convert output_shape_vec to size_t*
    size_t* output_shape = new size_t[output_shape_vec.size()];
    for (size_t i = 0; i < output_shape_vec.size(); ++i)
    {
        output_shape[i] = output_shape_vec[i];
    }
    size_t output_ndim = output_shape_vec.size();

    // Compute strides for lhs and rhs based on the broadcasted output shape
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

    // Pass original input shapes to the strided kernel for K dimension determination
    size_t* lhs_original_shape = new size_t[lhs.shape().size()];
    for (size_t i = 0; i < lhs.shape().size(); ++i)
    {
        lhs_original_shape[i] = lhs.shape()[i];
    }

    size_t* rhs_original_shape = new size_t[rhs.shape().size()];
    for (size_t i = 0; i < rhs.shape().size(); ++i)
    {
        rhs_original_shape[i] = rhs.shape()[i];
    }

    switch (dtype)
    {
    case core::DType::FLOAT32:
        plast_cpu_matmul_kernel_strided_float(
            output.data_as<float>(), lhs.data_as<const float>(), rhs.data_as<const float>(),
            output_shape, output_ndim, lhs_strides, rhs_strides, lhs_original_shape,
            rhs_original_shape);
        break;
    case core::DType::INT32:
        plast_cpu_matmul_kernel_strided_int32(
            output.data_as<int32_t>(), lhs.data_as<const int32_t>(), rhs.data_as<const int32_t>(),
            output_shape, output_ndim, lhs_strides, rhs_strides, lhs_original_shape,
            rhs_original_shape);
        break;
    default:
        throw std::runtime_error("Unsupoorted DType for Matmul operation on CPU.");
    }

    delete[] output_shape;
    delete[] lhs_strides;
    delete[] rhs_strides;
    delete[] lhs_original_shape;
    delete[] rhs_original_shape;

    return output;
}

tensor::Tensor MatmulOperation::execute_cuda(const std::vector<const tensor::Tensor*>& inputs) const
{
#ifdef PLAST_CUDA_ENABLED
    const tensor::Tensor& lhs = *inputs[0];
    const tensor::Tensor& rhs = *inputs[1];

    if (lhs.dtype() != rhs.dtype())
    {
        throw std::runtime_error("DType mismatch for Matmul operation on cpu");
    }

    core::DType dtype = lhs.dtype();

    std::vector<std::vector<size_t>> input_shapes = {lhs.shape(), rhs.shape()};

    std::vector<size_t> output_shape = infer_output_shape(input_shapes);

    int B = 1;
    for (int i = 0; i < output_shape.size() - 2; ++i)
    {
        B *= output_shape[i];
    }

    int N = output_shape[output_shape.size() - 2];
    int M = output_shape[output_shape.size() - 1];
    int K = lhs.shape()[output_shape.size() - 1];

    tensor::Tensor output(output_shape, lhs.dtype(), core::DeviceType::CPU);
    return output;
#else
    throw std::runtime_error(
        "CUDA is not enabled. Cannot execute Matmul operation on CUDA device.");
#endif
}

} // namespace ops
} // namespace plast
