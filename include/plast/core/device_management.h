#pragma once

#include "plast/core/types.h"
#include <string>
#include <vector>

#ifdef PLAST_CUDA_ENABLED
#include <cuda_runtime.h>
#endif

namespace plast
{
namespace core
{

// Function to count available devices of a specific type
int count_devices(DeviceType type);

// Check if CUDA is available and enabled
bool is_cuda_available();

// Get the currently active CUDA device ID
int get_current_cuda_device();

// Set the active CUDA device
void set_cuda_device(int device_id);

// Get device properties (e.g., name, memory)
std::string get_device_properties(DeviceType type, int device_id);

// Get free and total memory for a CUDA device
void get_cuda_memory_info(int device_id, size_t* free_mem, size_t* total_mem);

// Utility for CUDA error checking
#ifdef PLAST_CUDA_ENABLED
#define PLAST_CUDA_CHECK(ans)                                                                      \
    {                                                                                              \
        plast::core::plast_cuda_assert((ans), __FILE__, __LINE__);                                 \
    }
inline void plast_cuda_assert(cudaError_t code, const char* file, int line, bool abort = true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr, "CUDA Error: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}
#else
#define PLAST_CUDA_CHECK(ans)                                                                      \
    {                                                                                              \
    } // No-op if CUDA is not enabled
#endif

} // namespace core
} // namespace plast
