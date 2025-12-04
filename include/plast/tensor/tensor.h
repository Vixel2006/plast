#pragma once

#include <memory>
#include <numeric>
#include <stdexcept>
#include <vector>

#include "plast/core/data_buffer.h"
#include "plast/core/types.h"

namespace plast
{
namespace tensor
{

// Forward declaration for helper functions
size_t get_dtype_size(core::DType dtype);
std::vector<size_t> calculate_contiguous_strides(const std::vector<size_t>& shape);

class Tensor
{
  public:
    // Constructor for creating a new tensor with allocated memory (contiguous strides)
    Tensor(const std::vector<size_t>& shape, core::DType dtype, core::DeviceType device);
    Tensor(const std::vector<size_t>& shape, const std::vector<size_t>& strides, core::DType dtype,
           core::DeviceType device);

    // Constructor for creating a view (shares DataBuffer)
    Tensor(std::shared_ptr<core::DataBuffer> buffer, const std::vector<size_t>& shape,
           const std::vector<size_t>& strides, core::DType dtype);

    // Destructor (shared_ptr handles DataBuffer deallocation)
    ~Tensor() = default;

    // Delete copy constructor and assignment operator to prevent accidental deep copies
    // For explicit copies, use the .clone() method
    Tensor(const Tensor&) = delete;
    Tensor& operator=(const Tensor&) = delete;

    // Move constructor and assignment operator
    Tensor(Tensor&& other) noexcept;
    Tensor& operator=(Tensor&& other) noexcept;

    // Accessors
    void* data() const;
    const std::vector<size_t>& shape() const { return shape_; }
    const std::vector<size_t>& strides() const { return strides_; }
    core::DType dtype() const { return dtype_; }
    size_t ndim() const { return shape_.size(); }
    core::DeviceType device() const;
    size_t num_elements() const;
    size_t nbytes() const;

    // Device transfer
    Tensor to(core::DeviceType target_device) const;

    // Clone method for explicit deep copy
    Tensor clone() const;

    // Utility methods
    bool is_contiguous() const;
    template <typename T> T* data_as() const
    {
        if (sizeof(T) != get_dtype_size(dtype_))
        {
            throw std::runtime_error("Data type mismatch for data_as() call.");
        }
        return static_cast<T*>(data());
    }

    // Reshape method
    Tensor reshape(const std::vector<size_t>& new_shape) const;
    Tensor reshape(const std::vector<size_t>& new_shape,
                   const std::vector<size_t>& new_strides) const;

    // View method for creating a new tensor with different shape/strides but same data
    Tensor view(const std::vector<size_t>& new_shape, const std::vector<size_t>& new_strides) const;

  private:
    std::shared_ptr<core::DataBuffer> buffer_;
    std::vector<size_t> shape_;
    std::vector<size_t> strides_;
    core::DType dtype_;
};

} // namespace tensor
} // namespace plast
