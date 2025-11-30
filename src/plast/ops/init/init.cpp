#include "plast/core/device_management.h"
#include "plast/ops/init/init_ops.h"

#include <algorithm>
#include <cstring>
#include <random>
#include <stdexcept>

#ifdef PLAST_CUDA_ENABLED
#include <cuda_runtime.h>
// Declare CUDA kernels for initialization
extern "C" void plast_cuda_zeros_float(float* out, size_t num_elements);
extern "C" void plast_cuda_zeros_int(int* out, size_t num_elements);
extern "C" void plast_cuda_ones_float(float* out, size_t num_elements);
extern "C" void plast_cuda_ones_int(int* out, size_t num_elements);
extern "C" void plast_cuda_randn_float(float* out, size_t num_elements, int seed);
extern "C" void plast_cuda_uniform_float(float* out, size_t num_elements, float low, float high);
#endif

namespace plast
{
namespace ops
{
namespace init
{

std::shared_ptr<plast::tensor::Tensor> zeros(const std::vector<size_t>& shape, plast::core::DType dtype,
                            plast::core::DeviceType device)
{
    auto output = std::make_shared<plast::tensor::Tensor>(shape, dtype, device);
    // Dispatch to CPU or CUDA kernel
    if (device == plast::core::DeviceType::CPU)
    {
        // Assuming a CPU kernel for zeros exists
        // For now, we'll just set to 0 manually or use memset
        if (dtype == plast::core::DType::FLOAT32)
        {
            std::fill((float*) output->data(), (float*) output->data() + output->num_elements(), 0.0f);
        }
        else if (dtype == plast::core::DType::INT32)
        {
            std::fill((int*) output->data(), (int*) output->data() + output->num_elements(), 0);
        }
        else
        {
            throw std::runtime_error("Unsupported DType for zeros on CPU.");
        }
    }
    else if (device == plast::core::DeviceType::CUDA)
    {
#ifdef PLAST_CUDA_ENABLED
        // Assuming a CUDA kernel for zeros exists
        if (dtype == plast::core::DType::FLOAT32)
        {
            plast_cuda_zeros_float((float*) output->data(), output->num_elements());
        }
        else if (dtype == plast::core::DType::INT32)
        {
            plast_cuda_zeros_int((int*) output->data(), output->num_elements());
        }
        else
        {
            throw std::runtime_error("Unsupported DType for zeros on CUDA.");
        }
#else
        throw std::runtime_error("CUDA is not enabled. Cannot create zeros tensor on CUDA device.");
#endif
    }
    else
    {
        throw std::runtime_error("Unsupported device type for zeros.");
    }
    return output;
}

std::shared_ptr<plast::tensor::Tensor> ones(const std::vector<size_t>& shape, plast::core::DType dtype,
                           plast::core::DeviceType device)
{
    auto output = std::make_shared<plast::tensor::Tensor>(shape, dtype, device);
    // Dispatch to CPU or CUDA kernel
    if (device == plast::core::DeviceType::CPU)
    {
        if (dtype == plast::core::DType::FLOAT32)
        {
            std::fill((float*) output->data(), (float*) output->data() + output->num_elements(), 1.0f);
        }
        else if (dtype == plast::core::DType::INT32)
        {
            std::fill((int*) output->data(), (int*) output->data() + output->num_elements(), 1);
        }
        else
        {
            throw std::runtime_error("Unsupported DType for ones on CPU.");
        }
    }
    else if (device == plast::core::DeviceType::CUDA)
    {
#ifdef PLAST_CUDA_ENABLED
        if (dtype == plast::core::DType::FLOAT32)
        {
            plast_cuda_ones_float((float*) output->data(), output->num_elements());
        }
        else if (dtype == plast::core::DType::INT32)
        {
            plast_cuda_ones_int((int*) output->data(), output->num_elements());
        }
        else
        {
            throw std::runtime_error("Unsupported DType for ones on CUDA.");
        }
#else
        throw std::runtime_error("CUDA is not enabled. Cannot create ones tensor on CUDA device.");
#endif
    }
    else
    {
        throw std::runtime_error("Unsupported device type for ones.");
    }
    return output;
}

std::shared_ptr<plast::tensor::Tensor> randn(const std::vector<size_t>& shape, plast::core::DType dtype,
                            plast::core::DeviceType device, int seed)
{
    auto output = std::make_shared<plast::tensor::Tensor>(shape, dtype, device);
    // Dispatch to CPU or CUDA kernel
    if (device == plast::core::DeviceType::CPU)
    {
        std::mt19937 generator(seed);
        std::normal_distribution<float> distribution(0.0f, 1.0f);
        if (dtype == plast::core::DType::FLOAT32)
        {
            float* data_ptr = (float*) output->data();
            for (size_t i = 0; i < output->num_elements(); ++i)
            {
                data_ptr[i] = distribution(generator);
            }
        }
        else
        {
            throw std::runtime_error("Unsupported DType for randn on CPU.");
        }
    }
    else if (device == plast::core::DeviceType::CUDA)
    {
#ifdef PLAST_CUDA_ENABLED
        if (dtype == plast::core::DType::FLOAT32)
        {
            plast_cuda_randn_float((float*) output->data(), output->num_elements(), seed);
        }
        else
        {
            throw std::runtime_error("Unsupported DType for randn on CUDA.");
        }
#else
        throw std::runtime_error("CUDA is not enabled. Cannot create randn tensor on CUDA device.");
#endif
    }
    else
    {
        throw std::runtime_error("Unsupported device type for randn.");
    }
    return output;
}

std::shared_ptr<plast::tensor::Tensor> uniform(const std::vector<size_t>& shape, plast::core::DType dtype,
                              plast::core::DeviceType device, float low, float high)
{
    auto output = std::make_shared<plast::tensor::Tensor>(shape, dtype, device);
    // Dispatch to CPU or CUDA kernel
    if (device == plast::core::DeviceType::CPU)
    {
        std::mt19937 generator(std::random_device{}()); // Use random_device for seed
        std::uniform_real_distribution<float> distribution(low, high);
        if (dtype == plast::core::DType::FLOAT32)
        {
            float* data_ptr = (float*) output->data();
            for (size_t i = 0; i < output->num_elements(); ++i)
            {
                data_ptr[i] = distribution(generator);
            }
        }
        else
        {
            throw std::runtime_error("Unsupported DType for uniform on CPU.");
        }
    }
    else if (device == plast::core::DeviceType::CUDA)
    {
#ifdef PLAST_CUDA_ENABLED
        if (dtype == plast::core::DType::FLOAT32)
        {
            plast_cuda_uniform_float((float*) output->data(), output->num_elements(), low, high);
        }
        else
        {
            throw std::runtime_error("Unsupported DType for uniform on CUDA.");
        }
#else
        throw std::runtime_error(
            "CUDA is not enabled. Cannot create uniform tensor on CUDA device.");
#endif
    }
    else
    {
        throw std::runtime_error("Unsupported device type for uniform.");
    }
    return output;
}

std::shared_ptr<plast::tensor::Tensor> from_data(void* data, const std::vector<size_t>& shape,
                                plast::core::DType dtype, plast::core::DeviceType device)
{
    auto output = std::make_shared<plast::tensor::Tensor>(shape, dtype, device);
    size_t num_elements = output->num_elements();
    size_t nbytes = output->nbytes();

    if (device == plast::core::DeviceType::CPU)
    {
        std::memcpy(output->data(), data, nbytes);
    }
    else if (device == plast::core::DeviceType::CUDA)
    {
#ifdef PLAST_CUDA_ENABLED
        PLAST_CUDA_CHECK(cudaMemcpy(output->data(), data, nbytes, cudaMemcpyHostToDevice));
#else
        throw std::runtime_error(
            "CUDA is not enabled. Cannot create tensor from data on CUDA device.");
#endif
    }
    else
    {
        throw std::runtime_error("Unsupported device type for from_data.");
    }
    return output;
}

} // namespace init
} // namespace ops
} // namespace plast
