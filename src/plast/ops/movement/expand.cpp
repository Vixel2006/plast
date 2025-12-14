#include "plast/ops/movement/expand.h"
#include "plast/kernels/cpu/expand_kernels.h" // New include
#include "plast/kernels/cuda/expand_kernels.h" // New include
#include "plast/tensor/tensor.h"               // For get_dtype_size

#include <numeric>
#include <stdexcept>

// Forward declaration for get_dtype_size (defined in tensor.cpp)
namespace plast
{
namespace tensor
{
size_t get_dtype_size(core::DType dtype);
}
} // namespace plast

namespace plast
{
namespace ops
{

tensor::Tensor ExpandOperation::execute_cpu(const std::vector<const tensor::Tensor*>& inputs) const
{
    if (inputs.size() != 1)
    {
        throw std::runtime_error("ExpandOperation expects exactly one input tensor.");
    }

    const tensor::Tensor* input_tensor = inputs[0];
    const std::vector<size_t>& input_shape = input_tensor->shape();

    // Use the infer_output_shape logic to get the actual output shape and validate expandability
    std::vector<size_t> output_shape = infer_output_shape({input_shape});

    // Create a new contiguous output tensor
    tensor::Tensor output_tensor(output_shape, input_tensor->dtype(), input_tensor->device());

    // Get raw pointers and sizes for the kernel
    const void* input_data = input_tensor->data();
    void* output_data = output_tensor.data();
    size_t item_size = plast::tensor::get_dtype_size(input_tensor->dtype());

    // Convert std::vector to C-style arrays for kernel
    std::vector<size_t> input_shape_vec = input_tensor->shape();
    std::vector<size_t> input_strides_vec = input_tensor->strides();
    std::vector<size_t> output_shape_vec = output_tensor.shape();

    cpu_expand_kernel(input_data, input_shape_vec.data(), input_strides_vec.data(),
                      input_shape_vec.size(), output_data, output_shape_vec.data(),
                      output_shape_vec.size(), item_size);

    return output_tensor;
}

tensor::Tensor ExpandOperation::execute_cuda(const std::vector<const tensor::Tensor*>& inputs) const
{
    if (inputs.size() != 1)
    {
        throw std::runtime_error("ExpandOperation expects exactly one input tensor.");
    }

    const tensor::Tensor* input_tensor = inputs[0];
    const std::vector<size_t>& input_shape = input_tensor->shape();

    // Use the infer_output_shape logic to get the actual output shape and validate expandability
    std::vector<size_t> output_shape = infer_output_shape({input_shape});

    // Create a new contiguous output tensor
    tensor::Tensor output_tensor(output_shape, input_tensor->dtype(), input_tensor->device());

    // Get raw pointers and sizes for the kernel
    const void* input_data = input_tensor->data();
    void* output_data = output_tensor.data();
    size_t item_size = plast::tensor::get_dtype_size(input_tensor->dtype());

    // Convert std::vector to C-style arrays for kernel
    std::vector<size_t> input_shape_vec = input_tensor->shape();
    std::vector<size_t> input_strides_vec = input_tensor->strides();
    std::vector<size_t> output_shape_vec = output_tensor.shape();

    cuda_expand_kernel(input_data, input_shape_vec.data(), input_strides_vec.data(),
                       input_shape_vec.size(), output_data, output_shape_vec.data(),
                       output_shape_vec.size(), item_size);

    return output_tensor;
}

} // namespace ops
} // namespace plast
