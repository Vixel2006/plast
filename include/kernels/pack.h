#pragma once

#include "definitions.h"
#include "tensor.h"

typedef struct {
  void *data;
  void *_buf;
} TensorPack;

void tensor_pack_init(TensorPack *p, const Tensor *t);
void tensor_pack_release(TensorPack *p);

#ifdef __cplusplus
#include <cuda_runtime.h>

typedef struct {
  void *data;
  void *_d_buf;
} CudaTensorPack;

void cuda_tensor_pack_init(CudaTensorPack *p, const Tensor *t);
void cuda_tensor_pack_release(CudaTensorPack *p);
#endif
