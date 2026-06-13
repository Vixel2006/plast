#include "core/arena.h"
#include "core/arena_cuda.h"
#include <cuda_runtime.h>

#define CUDA_CHECK_ERROR(err)                                                                      \
  {                                                                                                \
    if (err != cudaSuccess) {                                                                      \
      fprintf(stderr, "CUDA Error: %s %s %d\n", cudaGetErrorString(err), __FILE__, __LINE__);      \
      exit(EXIT_FAILURE);                                                                          \
    }                                                                                              \
  }

extern "C" void *arena_alloc_cuda(u64 size) {
  void *ptr;
  CUDA_CHECK_ERROR(cudaMalloc(&ptr, size));
  return ptr;
}
extern "C" void arena_free_cuda(void *ptr) {
  CUDA_CHECK_ERROR(cudaFree(ptr));
}
extern "C" void arena_memset_cuda(void *ptr, int value, u64 size) {
  CUDA_CHECK_ERROR(cudaMemset(ptr, value, size));
}
extern "C" void arena_memcpy_h2d_cuda(void *dest, const void *src, u64 size) {
  CUDA_CHECK_ERROR(cudaMemcpy(dest, src, size, cudaMemcpyHostToDevice));
}
extern "C" void arena_memcpy_d2h_cuda(void *dest, const void *src, u64 size) {
  CUDA_CHECK_ERROR(cudaMemcpy(dest, src, size, cudaMemcpyDeviceToHost));
}
extern "C" void arena_memcpy_d2d_cuda(void *dest, const void *src, u64 size) {
  CUDA_CHECK_ERROR(cudaMemcpy(dest, src, size, cudaMemcpyDeviceToDevice));
}
