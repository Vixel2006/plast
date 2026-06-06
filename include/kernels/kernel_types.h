#pragma once
#include "core/definitions.h"
#include "core/tensor.h"

// CPU Constants
#define SIMD_WIDTH 8

// CUDA Constants
#define CUDA_BLOCK_SIZE 256
#define CUDA_GRID_SIZE(n) (((n) + CUDA_BLOCK_SIZE - 1) / CUDA_BLOCK_SIZE)
