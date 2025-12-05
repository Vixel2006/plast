#include "plast/ops/binary/matmul.h"
#include "plast/core/device_management.h"
#include "plast/core/shape_utils_cpp.h"
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

    size_t output_ndim = output_shape_vec.size();

    // Compute strides for lhs and rhs based on the broadcasted output shape
    std::vector<size_t> lhs_strides_vec =
        core::get_effective_broadcast_strides(lhs.shape(), lhs.strides(), output_shape_vec);
    std::vector<size_t> rhs_strides_vec =
        core::get_effective_broadcast_strides(rhs.shape(), rhs.strides(), output_shape_vec);

    // Pass original input shapes to the strided kernel for K dimension determination
    const std::vector<size_t>& lhs_original_shape = lhs.shape();
    const std::vector<size_t>& rhs_original_shape = rhs.shape();

    // Replicate effective shape logic from infer_output_shape to get K_dim
    std::vector<size_t> effective_lhs_shape = lhs_original_shape;
    std::vector<size_t> effective_rhs_shape = rhs_original_shape;

    size_t lhs_ndim_eff = lhs_original_shape.size();
    size_t rhs_ndim_eff = rhs_original_shape.size();

    if (lhs_ndim_eff == 1)
    {
        effective_lhs_shape.insert(effective_lhs_shape.begin(), 1); // (D) -> (1, D)
        lhs_ndim_eff++;
    }
    if (rhs_ndim_eff == 1)
    {
        effective_rhs_shape.push_back(1); // (D) -> (D, 1)
        rhs_ndim_eff++;
    }

    size_t K_dim = effective_lhs_shape[lhs_ndim_eff - 1];

    switch (dtype)
    {
    case core::DType::FLOAT32:
        plast_cpu_matmul_kernel_strided_float(
            output.data_as<float>(), lhs.data_as<const float>(), rhs.data_as<const float>(),
            output_shape_vec.data(), output_ndim, lhs_strides_vec.data(), rhs_strides_vec.data(),
            lhs_original_shape.data(), rhs_original_shape.data(), K_dim);
        break;
    case core::DType::INT32:
        plast_cpu_matmul_kernel_strided_int32(
            output.data_as<int32_t>(), lhs.data_as<const int32_t>(), rhs.data_as<const int32_t>(),
            output_shape_vec.data(), output_ndim, lhs_strides_vec.data(), rhs_strides_vec.data(),
            lhs_original_shape.data(), rhs_original_shape.data(), K_dim);
        break;
    default:
        throw std::runtime_error("Unsupoorted DType for Matmul operation on CPU.");
    }

    return output;
}

tensor::Tensor MatmulOperation::execute_cuda(const std::vector<const tensor::Tensor*>& inputs) const
{
#ifdef PLAST_CUDA_ENABLED
    const tensor::Tensor& lhs = *inputs[0];
    const tensor::Tensor& rhs = *inputs[1];

    if (lhs.dtype() != rhs.dtype())
    {
        throw std::runtime_error("DType mismatch for Matmul operation on CUDA");
    }

    core::DType dtype = lhs.dtype();

    std::vector<std::vector<size_t>> input_shapes_vec = {lhs.shape(), rhs.shape()};

    std::vector<size_t> output_shape_vec = infer_output_shape(input_shapes_vec);

    // Allocate output tensor on CUDA device
    tensor::Tensor output(output_shape_vec, dtype, core::DeviceType::CUDA);

    size_t output_ndim = output_shape_vec.size();

    // Compute strides for lhs and rhs based on the broadcasted output shape
    std::vector<size_t> lhs_strides_vec =
        core::get_effective_broadcast_strides(lhs.shape(), lhs.strides(), output_shape_vec);
    std::vector<size_t> rhs_strides_vec =
        core::get_effective_broadcast_strides(rhs.shape(), rhs.strides(), output_shape_vec);

    // Pass original input shapes to the strided kernel for K dimension determination
    const std::vector<size_t>& lhs_original_shape = lhs.shape();
    const std::vector<size_t>& rhs_original_shape = rhs.shape();

    // Replicate effective shape logic from infer_output_shape to get K_dim
    std::vector<size_t> effective_lhs_shape = lhs_original_shape;
    std::vector<size_t> effective_rhs_shape = rhs_original_shape;

    size_t lhs_ndim_eff = lhs_original_shape.size();
    size_t rhs_ndim_eff = rhs_original_shape.size();

    if (lhs_ndim_eff == 1)
    {
        effective_lhs_shape.insert(effective_lhs_shape.begin(), 1); // (D) -> (1, D)
        lhs_ndim_eff++;
    }
    if (rhs_ndim_eff == 1)
    {
        effective_rhs_shape.push_back(1); // (D) -> (D, 1)
        rhs_ndim_eff++;
    }

    size_t K_dim = effective_lhs_shape[lhs_ndim_eff - 1];

    switch (dtype)
    {
    case core::DType::FLOAT32:
    {
        // Replicate effective shape logic from infer_output_shape to get B, N, M, K
        std::vector<size_t> current_lhs_shape = lhs.shape();
        std::vector<size_t> current_rhs_shape = rhs.shape();

        size_t current_lhs_ndim = current_lhs_shape.size();
        size_t current_rhs_ndim = current_rhs_shape.size();

        if (current_lhs_ndim == 1)
        {
            current_lhs_shape.insert(current_lhs_shape.begin(), 1); // (D) -> (1, D)
            current_lhs_ndim++;
        }
        if (current_rhs_ndim == 1)
        {
            current_rhs_shape.push_back(1); // (D) -> (D, 1)
            current_rhs_ndim++;
        }

        std::vector<size_t> lhs_batch_shape(current_lhs_shape.begin(), current_lhs_shape.end() - 2);
        std::vector<size_t> rhs_batch_shape(current_rhs_shape.begin(), current_rhs_shape.end() - 2);

        std::vector<size_t> output_batch_shape =
            core::broadcast_shapes(lhs_batch_shape, rhs_batch_shape);

        int B_dim = 1;
        for (size_t dim_size : output_batch_shape)
        {
            B_dim *= dim_size;
        }

        int N_dim = current_lhs_shape[current_lhs_ndim - 2];
        int K_dim_val = current_lhs_shape[current_lhs_ndim - 1];
        int M_dim = current_rhs_shape[current_rhs_ndim - 1];

        plast_cuda_matmul_kernel_float(output.data_as<float>(), lhs.data_as<const float>(),
                                       rhs.data_as<const float>(), B_dim, N_dim, M_dim, K_dim_val);
    }
    break;
    case core::DType::INT32:
    {
        // Replicate effective shape logic from infer_output_shape to get B, N, M, K
        std::vector<size_t> current_lhs_shape = lhs.shape();
        std::vector<size_t> current_rhs_shape = rhs.shape();

        size_t current_lhs_ndim = current_lhs_shape.size();
        size_t current_rhs_ndim = current_rhs_shape.size();

        if (current_lhs_ndim == 1)
        {
            current_lhs_shape.insert(current_lhs_shape.begin(), 1); // (D) -> (1, D)
            current_lhs_ndim++;
        }
        if (current_rhs_ndim == 1)
        {
            current_rhs_shape.push_back(1); // (D) -> (D, 1)
            current_rhs_ndim++;
        }

        std::vector<size_t> lhs_batch_shape(current_lhs_shape.begin(), current_lhs_shape.end() - 2);
        std::vector<size_t> rhs_batch_shape(current_rhs_shape.begin(), current_rhs_shape.end() - 2);

        std::vector<size_t> output_batch_shape =
            core::broadcast_shapes(lhs_batch_shape, rhs_batch_shape);

        int B_dim = 1;
        for (size_t dim_size : output_batch_shape)
        {
            B_dim *= dim_size;
        }

        int N_dim = current_lhs_shape[current_lhs_ndim - 2];
        int K_dim_val = current_lhs_shape[current_lhs_ndim - 1];
        int M_dim = current_rhs_shape[current_rhs_ndim - 1];

        plast_cuda_matmul_kernel_int32(output.data_as<int32_t>(), lhs.data_as<const int32_t>(),
                                       rhs.data_as<const int32_t>(), B_dim, N_dim, M_dim, K_dim_val);
    }
    break;
    default:
        throw std::runtime_error("Unsupported DType for Matmul operation on CUDA.");
    }

    return output;
#else
    throw std::runtime_error(
        "CUDA is not enabled. Cannot execute Matmul operation on CUDA device.");
#endif
}

} // namespace ops
} // namespace plast
