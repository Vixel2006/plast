#include "plast/tensor/tensor.h"
#include "plast/core/data_buffer.h"
#include "plast/core/types.h"
#include <iostream>

#include <cstring>
#include <numeric>
#include <stdexcept>

#ifdef PLAST_CUDA_ENABLED
#include <cuda_runtime.h>
#endif

namespace plast
{
namespace tensor
{

// Helper to get size of DType in bytes (moved from .h to .cpp)
size_t get_dtype_size(core::DType dtype)
{
    switch (dtype)
    {
    case core::DType::FLOAT32:
        return sizeof(float);
    case core::DType::FLOAT64:
        return sizeof(double);
    case core::DType::INT8:
        return sizeof(int8_t);
    case core::DType::INT16:
        return sizeof(int16_t);
    case core::DType::INT32:
        return sizeof(int32_t);
    case core::DType::INT64:
        return sizeof(int64_t);
    case core::DType::UINT8:
        return sizeof(uint8_t);
    case core::DType::UINT16:
        return sizeof(uint16_t);
    case core::DType::UINT32:
        return sizeof(uint32_t);
    case core::DType::UINT64:
        return sizeof(uint64_t);
    case core::DType::BOOL:
        return sizeof(bool);
    case core::DType::UNKNOWN:
    default:
        throw std::runtime_error("Unknown or unsupported DType.");
    }
}

// Helper to calculate contiguous strides (moved from .h to .cpp)
std::vector<size_t> calculate_contiguous_strides(const std::vector<size_t>& shape)
{
    std::vector<size_t> strides(shape.size());
    if (shape.empty())
    {
        return strides;
    }
    size_t stride = 1;
    for (int i = shape.size() - 1; i >= 0; --i)
    {
        strides[i] = stride;
        stride *= shape[i];
    }
    return strides;
}

// Helper to increment multi-dimensional coordinates
void increment_coords(std::vector<size_t>& coords, const std::vector<size_t>& shape)
{
    for (int i = coords.size() - 1; i >= 0; --i)
    {
        coords[i]++;
        if (coords[i] < shape[i])
        {
            return;
        }
        coords[i] = 0;
    }
}

// Constructor for creating a new tensor with allocated memory (contiguous strides)
Tensor::Tensor(const std::vector<size_t>& shape, core::DType dtype, core::DeviceType device)
    : shape_(shape), dtype_(dtype)
{
    if (shape_.empty())
    {
        shape_.push_back(1); // Scalar tensor
    }
    strides_ = calculate_contiguous_strides(shape_);
    size_t bytes = num_elements() * get_dtype_size(dtype_);
    buffer_ = std::make_shared<core::DataBuffer>(bytes, device);
}

Tensor::Tensor(const std::vector<size_t>& shape, const std::vector<size_t>& strides,
               core::DType dtype, core::DeviceType device)
    : shape_(shape), strides_(strides), dtype_(dtype)
{
    if (shape.empty())
    {
        shape_.push_back(1);
    }
    size_t bytes = num_elements() * get_dtype_size(dtype_);
    buffer_ = std::make_shared<core::DataBuffer>(bytes, device);
}

// Constructor for creating a view (shares DataBuffer)
Tensor::Tensor(std::shared_ptr<core::DataBuffer> buffer, const std::vector<size_t>& shape,
               const std::vector<size_t>& strides, core::DType dtype)
    : buffer_(buffer), shape_(shape), strides_(strides), dtype_(dtype)
{
    if (!buffer_)
    {
        throw std::runtime_error("DataBuffer cannot be null for a Tensor view.");
    }
    if (shape_.size() != strides_.size())
    {
        throw std::runtime_error("Shape and strides must have the same number of dimensions.");
    }
}

// Move constructor
Tensor::Tensor(Tensor&& other) noexcept
    : buffer_(std::move(other.buffer_)), shape_(std::move(other.shape_)),
      strides_(std::move(other.strides_)), dtype_(other.dtype_)
{
}

// Move assignment operator
Tensor& Tensor::operator=(Tensor&& other) noexcept
{
    if (this != &other)
    {
        buffer_ = std::move(other.buffer_);
        shape_ = std::move(other.shape_);
        strides_ = std::move(other.strides_);
        dtype_ = std::move(other.dtype_);
    }
    return *this;
}

void* Tensor::data() const
{
    if (!buffer_)
    {
        throw std::runtime_error("Attempted to access data from a Tensor with no DataBuffer.");
    }
    return buffer_->data();
}

core::DeviceType Tensor::device() const
{
    if (!buffer_)
    {
        throw std::runtime_error("Attempted to access device from a Tensor with no DataBuffer.");
    }
    return buffer_->device();
}

size_t Tensor::num_elements() const
{
    return std::accumulate(shape_.begin(), shape_.end(), 1ULL, std::multiplies<size_t>());
}

size_t Tensor::nbytes() const
{
    if (!buffer_)
    {
        throw std::runtime_error("Attempted to access nbytes from a Tensor with no DataBuffer.");
    }
    return buffer_->nbytes();
}

bool Tensor::is_contiguous() const
{
    if (shape_.empty())
    {
        return true; // Scalar is contiguous
    }

    size_t expected_stride = 1;
    for (int i = shape_.size() - 1; i >= 0; --i)
    {
        if (strides_[i] != expected_stride)
        {
            return false;
        }
        expected_stride *= shape_[i];
    }
    return true;
}

Tensor Tensor::to(core::DeviceType target_device) const
{
    if (device() == target_device)
    {
        return clone(); // Already on target device, return a copy
    }

    // Create a new tensor on the target device
    Tensor new_tensor(shape_, strides_, dtype_, target_device);

    // Copy data from current buffer to new buffer
    size_t bytes = nbytes();
    if (bytes > 0)
    {
        if (device() == core::DeviceType::CPU && target_device == core::DeviceType::CUDA)
        {
#ifdef PLAST_CUDA_ENABLED
            PLAST_CUDA_CHECK(cudaMemcpy(new_tensor.data(), data(), bytes, cudaMemcpyHostToDevice));
#else
            throw std::runtime_error("CUDA is not enabled. Cannot perform CUDA memcpy.");
#endif
        }
        else if (device() == core::DeviceType::CUDA && target_device == core::DeviceType::CPU)
        {
#ifdef PLAST_CUDA_ENABLED
            PLAST_CUDA_CHECK(
                cudaMemcpy(new_tensor.data(), data(), bytes, cudaMemcpyDeviceToHost));
#else
            throw std::runtime_error("CUDA is not enabled. Cannot perform CUDA memcpy.");
#endif
        }
        else if (device() == core::DeviceType::CUDA && target_device == core::DeviceType::CUDA)
        {
#ifdef PLAST_CUDA_ENABLED
            PLAST_CUDA_CHECK(
                cudaMemcpy(new_tensor.data(), data(), bytes, cudaMemcpyDeviceToDevice));
#else
            throw std::runtime_error("CUDA is not enabled. Cannot perform CUDA memcpy.");
#endif
        }
        else if (device() == core::DeviceType::CPU && target_device == core::DeviceType::CPU)
        {
            std::memcpy(new_tensor.data(), data(), bytes);
        }
        else
        {
            throw std::runtime_error("Unsupported cross-device copy combination.");
        }
    }
    return new_tensor;
}

Tensor Tensor::clone() const
{
    // Create a new contiguous tensor with the same shape, dtype, and device
    Tensor new_tensor(shape_, dtype_, device());

    if (is_contiguous())
    {
        // If the input tensor is contiguous, a simple data copy is sufficient
        size_t bytes = nbytes();
        if (bytes > 0)
        {
            if (device() == core::DeviceType::CPU)
            {
                std::memcpy(new_tensor.data(), data(), bytes);
            }
            else if (device() == core::DeviceType::CUDA)
            {
#ifdef PLAST_CUDA_ENABLED
                PLAST_CUDA_CHECK(
                    cudaMemcpy(new_tensor.data(), data(), bytes, cudaMemcpyDeviceToDevice));
#else
                throw std::runtime_error("CUDA is not enabled. Cannot perform CUDA memcpy.");
#endif
            }
        }
    }
    else
    {
        // If not contiguous, use strided copy
        if (device() == core::DeviceType::CPU)
        {
            size_t num_elements_ = num_elements();
            size_t item_size = get_dtype_size(dtype_);

            char* src_data = static_cast<char*>(data());
            char* dst_data = static_cast<char*>(new_tensor.data());

            std::vector<size_t> current_coords(shape_.size(), 0);
            for (size_t i = 0; i < num_elements_; ++i)
            {
                size_t src_offset = 0;
                for (size_t dim = 0; dim < shape_.size(); ++dim)
                {
                    src_offset += current_coords[dim] * strides_[dim];
                }
                std::memcpy(dst_data + i * item_size, src_data + src_offset * item_size, item_size);
                increment_coords(current_coords, shape_);
            }
        }
        else if (device() == core::DeviceType::CUDA)
        {
            throw std::runtime_error(
                "Non-contiguous CUDA clone is not supported without strided copy kernels.");
        }
    }
    return new_tensor;
}

Tensor Tensor::reshape(const std::vector<size_t>& new_shape) const
{
    size_t new_num_elements =
        std::accumulate(new_shape.begin(), new_shape.end(), 1ULL, std::multiplies<size_t>());
    if (new_num_elements != num_elements())
    {
        throw std::runtime_error("Reshape operation requires the total number of elements to "
                                 "remain constant.");
    }

    // Calculate new contiguous strides for the new shape
    std::vector<size_t> new_strides = calculate_contiguous_strides(new_shape);

    // Create a new Tensor that views the same DataBuffer, but with a new shape and new strides.
    return Tensor(buffer_, new_shape, new_strides, dtype_);
}

Tensor Tensor::reshape(const std::vector<size_t>& new_shape,
                       const std::vector<size_t>& new_strides) const
{
    size_t new_num_elements =
        std::accumulate(new_shape.begin(), new_shape.end(), 1ULL, std::multiplies<size_t>());
    if (new_num_elements != num_elements())
    {
        throw std::runtime_error("Reshape operation requires the total number of elements to "
                                 "remain constant.");
    }
    if (new_shape.size() != new_strides.size())
    {
        throw std::runtime_error("New shape and new strides must have the same number of "
                                 "dimensions for reshape.");
    }

    // Create a new Tensor that views the same DataBuffer, but with a new shape and new strides.
    return Tensor(buffer_, new_shape, new_strides, dtype_);
}

Tensor Tensor::view(const std::vector<size_t>& new_shape,
                    const std::vector<size_t>& new_strides) const
{
    // Create a new Tensor object that shares the same DataBuffer
    return Tensor(buffer_, new_shape, new_strides, dtype_);
}

} // namespace tensor
} // namespace plast
