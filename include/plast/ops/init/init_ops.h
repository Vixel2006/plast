#pragma once

#include <vector>
#include "plast/tensor/tensor.h"
#include "plast/core/types.h"

namespace plast {
namespace ops {
namespace init {

plast::tensor::Tensor zeros(const std::vector<size_t>& shape, plast::core::DType dtype, plast::core::DeviceType device);
plast::tensor::Tensor ones(const std::vector<size_t>& shape, plast::core::DType dtype, plast::core::DeviceType device);
plast::tensor::Tensor randn(const std::vector<size_t>& shape, plast::core::DType dtype, plast::core::DeviceType device, int seed);
plast::tensor::Tensor uniform(const std::vector<size_t>& shape, plast::core::DType dtype, plast::core::DeviceType device, float low, float high);
plast::tensor::Tensor from_data(void* data, const std::vector<size_t>& shape, plast::core::DType dtype, plast::core::DeviceType device);

} // namespace init
} // namespace ops
} // namespace plast
