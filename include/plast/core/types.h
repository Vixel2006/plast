#pragma once

#include <cstdint>

// Define a maximum number of dimensions for tensors to allow fixed-size arrays in CUDA kernels
#define MAX_TENSOR_DIMS 8

// Define a maximum number of dimensions for tensors to allow fixed-size arrays in CUDA kernels
#define MAX_TENSOR_DIMS 8

namespace plast
{
namespace core
{

enum class DType
{
    UNKNOWN = 0,
    FLOAT32,
    FLOAT64,
    INT8,
    INT16,
    INT32,
    INT64,
    UINT8,
    UINT16,
    UINT32,
    UINT64,
    BOOL
};

enum class DeviceType
{
    CPU = 0,
    CUDA
};

} // namespace core
} // namespace plast
