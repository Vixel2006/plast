#pragma once

#include "plast/core/types.h"
#include "plast/tensor/tensor.h"
#include <vector>

namespace plast
{
namespace ops
{
namespace init
{

std::shared_ptr<plast::tensor::Tensor>
zeros(const std::vector<size_t>& shape, plast::core::DType dtype, plast::core::DeviceType device);
std::shared_ptr<plast::tensor::Tensor>
ones(const std::vector<size_t>& shape, plast::core::DType dtype, plast::core::DeviceType device);
std::shared_ptr<plast::tensor::Tensor> randn(const std::vector<size_t>& shape,
                                             plast::core::DType dtype,
                                             plast::core::DeviceType device, int seed);
std::shared_ptr<plast::tensor::Tensor> uniform(const std::vector<size_t>& shape,
                                               plast::core::DType dtype,
                                               plast::core::DeviceType device, float low,
                                               float high);
std::shared_ptr<plast::tensor::Tensor> from_data(void* data, const std::vector<size_t>& shape,
                                                 plast::core::DType dtype,
                                                 plast::core::DeviceType device);

} // namespace init
} // namespace ops
} // namespace plast
