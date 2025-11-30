#include "plast/core/device_management.h"
#include "plast/core/data_buffer.h" // Include for PLAST_CUDA_CHECK
#include <iostream>
#include <stdexcept>

namespace plast
{
namespace core
{

int count_devices(DeviceType type)
{
    switch (type)
    {
    case DeviceType::CPU:
        return 1; // Always at least one CPU
    case DeviceType::CUDA:
#ifdef PLAST_CUDA_ENABLED
    {
        int count;
        cudaError_t err = cudaGetDeviceCount(&count);
        if (err != cudaSuccess)
        {
            // Handle error, e.g., no CUDA devices found or driver issue
            return 0;
        }
        return count;
    }
#else
        return 0; // CUDA not enabled
#endif
    default:
        throw std::runtime_error("Unsupported device type.");
    }
}

bool is_cuda_available()
{
#ifdef PLAST_CUDA_ENABLED
    int count;
    cudaError_t err = cudaGetDeviceCount(&count);
    return (err == cudaSuccess && count > 0);
#else
    return false;
#endif
}

int get_current_cuda_device()
{
#ifdef PLAST_CUDA_ENABLED
    int device;
    PLAST_CUDA_CHECK(cudaGetDevice(&device));
    return device;
#else
    throw std::runtime_error("CUDA is not enabled.");
#endif
}

void set_cuda_device(int device_id)
{
#ifdef PLAST_CUDA_ENABLED
    PLAST_CUDA_CHECK(cudaSetDevice(device_id));
#else
    throw std::runtime_error("CUDA is not enabled.");
#endif
}

std::string get_device_properties(DeviceType type, int device_id)
{
    std::string props_str;
    switch (type)
    {
    case DeviceType::CPU:
        props_str = "CPU Device (ID: " + std::to_string(device_id) + ")";
        // Could add more detailed CPU info if needed
        break;
    case DeviceType::CUDA:
#ifdef PLAST_CUDA_ENABLED
    {
        cudaDeviceProp prop;
        PLAST_CUDA_CHECK(cudaGetDeviceProperties(&prop, device_id));
        props_str = "CUDA Device (ID: " + std::to_string(device_id) + "): " + prop.name;
        props_str +=
            "\n  Total Global Memory: " + std::to_string(prop.totalGlobalMem / (1024 * 1024)) +
            " MB";
        props_str += "\n  Compute Capability: " + std::to_string(prop.major) + "." +
                     std::to_string(prop.minor);
        // Add more properties as needed
    }
#else
        props_str = "CUDA is not enabled.";
#endif
    break;
    default:
        props_str = "Unsupported device type.";
    }
    return props_str;
}

void get_cuda_memory_info(int device_id, size_t* free_mem, size_t* total_mem)
{
#ifdef PLAST_CUDA_ENABLED
    set_cuda_device(device_id); // Ensure we query the correct device
    PLAST_CUDA_CHECK(cudaMemGetInfo(free_mem, total_mem));
#else
    throw std::runtime_error("CUDA is not enabled.");
#endif
}

} // namespace core
} // namespace plast
