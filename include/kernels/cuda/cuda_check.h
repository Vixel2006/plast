#pragma once

#include <cuda_runtime.h>
#include <stdio.h>

#define CUDA_CHECK(expr)                                                       \
  do {                                                                         \
    cudaError_t _err = (expr);                                                 \
    if (_err != cudaSuccess) {                                                 \
      fprintf(stderr, "CUDA error at %s:%d: '%s': %s\n", __FILE__, __LINE__,   \
              #expr, cudaGetErrorString(_err));                                \
      return;                                                                  \
    }                                                                          \
  } while (0)

#define CUDA_CHECK_GOTO(expr, label)                                           \
  do {                                                                         \
    cudaError_t _err = (expr);                                                 \
    if (_err != cudaSuccess) {                                                 \
      fprintf(stderr, "CUDA error at %s:%d: '%s': %s\n", __FILE__, __LINE__,   \
              #expr, cudaGetErrorString(_err));                                \
      goto label;                                                              \
    }                                                                          \
  } while (0)
