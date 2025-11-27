#pragma once

#include <vector>
#include <memory> // For std::shared_ptr or std::unique_ptr
#include <numeric> // For std::accumulate
#include <stdexcept> // For std::runtime_error

#include "plast/core/types.h"

namespace plast {
namespace tensor {

class Tensor {
public:
    // Constructors
    Tensor(void* data, const std::vector<size_t>& shape, core::DType dtype, core::DeviceType device, bool owns_data = true);
    // Constructor for empty tensor (e.g., for output allocation)
    Tensor(const std::vector<size_t>& shape, core::DType dtype, core::DeviceType device);

    // Destructor
    ~Tensor();

    // Delete copy constructor and assignment operator to prevent accidental deep copies
    // For explicit copies, use the .clone() method
    Tensor(const Tensor&) = delete;
    Tensor& operator=(const Tensor&) = delete;

    // Move constructor and assignment operator
    Tensor(Tensor&& other) noexcept;
    Tensor& operator=(Tensor&& other) noexcept;

    // Accessors
    void* data() const { return data_; }
    const std::vector<size_t>& shape() const { return shape_; }
    core::DType dtype() const { return dtype_; }
    core::DeviceType device() const { return device_; }
    size_t num_elements() const;
    size_t nbytes() const;

    // Device transfer
    Tensor to(core::DeviceType target_device) const;

    // Clone method for explicit deep copy
    Tensor clone() const;

    // Utility methods
    template<typename T>
    T* data_as() const {
        if (sizeof(T) != nbytes() / num_elements()) {
            throw std::runtime_error("Data type mismatch for data_as() call.");
        }
        return static_cast<T*>(data_);
    }

private:
    void* data_;
    std::vector<size_t> shape_;
    core::DType dtype_;
    core::DeviceType device_;
    bool owns_data_; // If true, Tensor manages data memory

    // Helper for memory allocation
    void allocate_data();
    // Helper for memory deallocation
    void deallocate_data();
    // Helper for deep copy of data
    void copy_data_from(const Tensor& other);
};

} // namespace tensor
} // namespace plast
