#pragma once

#include <cstdint> // For int32_t, etc.

namespace plast {
namespace core {

enum class DType {
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

enum class DeviceType {
    CPU = 0,
    CUDA
};

} // namespace core
} // namespace plast
