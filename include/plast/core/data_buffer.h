#pragma once

#include "plast/core/types.h"
#include <cstddef>
#include <iostream>
#include <stdexcept>

#ifdef PLAST_CUDA_ENABLED
#include <cuda_runtime.h>
#define PLAST_CUDA_CHECK(call)                                                                     \
    do                                                                                             \
    {                                                                                              \
        cudaError_t err = call;                                                                    \
        if (err != cudaSuccess)                                                                    \
        {                                                                                          \
            std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " at " << __FILE__ << ":"    \
                      << __LINE__ << std::endl;                                                    \
            throw std::runtime_error(cudaGetErrorString(err));                                     \
        }                                                                                          \
    } while (0)
#endif

namespace plast
{
namespace core
{

class DataBuffer
{
  public:
    DataBuffer(size_t nbytes, DeviceType device) : data_(nullptr), nbytes_(nbytes), device_(device)
    {
        if (nbytes_ == 0)
        {
            return;
        }

        switch (device_)
        {
        case DeviceType::CPU:
            data_ = new char[nbytes_];
            break;
        case DeviceType::CUDA:
#ifdef PLAST_CUDA_ENABLED
            PLAST_CUDA_CHECK(cudaMalloc(&data_, nbytes_));
#else
            throw std::runtime_error("CUDA is not enabled. Cannot allocate CUDA memory.");
#endif
            break;
        default:
            throw std::runtime_error("Unsupported device type for DataBuffer allocation.");
        }
    }

    ~DataBuffer()
    {
        if (data_ == nullptr)
        {
            return;
        }

        switch (device_)
        {
        case DeviceType::CPU:
            delete[] static_cast<char*>(data_);
            break;
        case DeviceType::CUDA:
#ifdef PLAST_CUDA_ENABLED
        {
            cudaError_t err = cudaFree(data_);
            if (err != cudaSuccess)
            {
                // Log error but don't throw in destructor
                std::cerr << "CUDA memory deallocation failed: " << cudaGetErrorString(err)
                          << std::endl;
            }
        }
#else
            // Should not happen if allocation checked PLAST_CUDA_ENABLED
#endif
        break;
        default:
            break;
        }
        data_ = nullptr;
    }

    void* data() const { return data_; }
    size_t nbytes() const { return nbytes_; }
    DeviceType device() const { return device_; }

    // Delete copy constructor and assignment operator
    DataBuffer(const DataBuffer&) = delete;
    DataBuffer& operator=(const DataBuffer&) = delete;

    // Move constructor and assignment operator (optional, but good practice)
    DataBuffer(DataBuffer&& other) noexcept
        : data_(other.data_), nbytes_(other.nbytes_), device_(other.device_)
    {
        other.data_ = nullptr;
        other.nbytes_ = 0;
    }

    DataBuffer& operator=(DataBuffer&& other) noexcept
    {
        if (this != &other)
        {
            // Deallocate current resources
            if (data_ != nullptr)
            {
                switch (device_)
                {
                case DeviceType::CPU:
                    delete[] static_cast<char*>(data_);
                    break;
                case DeviceType::CUDA:
#ifdef PLAST_CUDA_ENABLED
                {
                    cudaError_t err = cudaFree(data_);
                    if (err != cudaSuccess)
                    {
                        std::cerr << "CUDA memory deallocation failed during move assignment: "
                                  << cudaGetErrorString(err) << std::endl;
                    }
                }
#endif
                break;
                default:
                    break;
                }
            }

            data_ = other.data_;
            nbytes_ = other.nbytes_;
            device_ = other.device_;

            other.data_ = nullptr;
            other.nbytes_ = 0;
        }
        return *this;
    }

  private:
    void* data_;
    size_t nbytes_;
    DeviceType device_;
};

} // namespace core
} // namespace plast
