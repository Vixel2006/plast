#include "data/dataset.h"
#include "data/dataloader.h"
#include <string.h>
#include <stdlib.h>

#ifdef CUDA_AVAILABLE
#include "core/arena_cuda.h"
#endif

TensorDataset *create_tensor_dataset(Tensor **tensors, u64 num_tensors) {
  if (num_tensors == 0 || tensors == NULL) return NULL;
  
  TensorDataset *dataset = malloc(sizeof(TensorDataset));
  dataset->num_tensors = num_tensors;
  dataset->tensors = malloc(sizeof(Tensor *) * num_tensors);
  for (u64 i = 0; i < num_tensors; i++) {
    dataset->tensors[i] = tensors[i];
  }
  dataset->size = tensors[0]->shape[0];
  return dataset;
}

void free_tensor_dataset(TensorDataset *dataset) {
  if (dataset) {
    free(dataset->tensors);
    free(dataset);
  }
}

DataLoader *create_dataloader(TensorDataset *dataset, u64 batch_size, bool shuffle, bool drop_last, DEVICE device) {
  DataLoader *loader = malloc(sizeof(DataLoader));
  loader->dataset = dataset;
  loader->batch_size = batch_size;
  loader->shuffle = shuffle;
  loader->drop_last = drop_last;
  loader->device = device;
  loader->indices = (u64 *)malloc(sizeof(u64) * dataset->size);
  for (u64 i = 0; i < dataset->size; i++) {
    loader->indices[i] = i;
  }
  loader->current_index = 0;
  return loader;
}

void free_dataloader(DataLoader *loader) {
  if (loader) {
    free(loader->indices);
    free(loader);
  }
}

void reset_dataloader_iterator(DataLoader *loader) {
  loader->current_index = 0;
  if (loader->shuffle) {
    u64 n = loader->dataset->size;
    for (u64 i = 0; i < n - 1; i++) {
      u64 j = i + rand() % (n - i);
      u64 temp = loader->indices[i];
      loader->indices[i] = loader->indices[j];
      loader->indices[j] = temp;
    }
  } else {
    for (u64 i = 0; i < loader->dataset->size; i++) {
      loader->indices[i] = i;
    }
  }
}

bool dataloader_next_batch(DataLoader *loader, Arena *meta_arena, Arena *data_arena, Tensor **out_batches) {
  u64 n = loader->dataset->size;
  if (loader->current_index >= n) {
    return false;
  }
  
  u64 b_size = loader->batch_size;
  if (loader->current_index + b_size > n) {
    if (loader->drop_last) {
      return false;
    }
    b_size = n - loader->current_index;
  }
  
  for (u64 col = 0; col < loader->dataset->num_tensors; col++) {
    Tensor *src_tensor = loader->dataset->tensors[col];
    
    u64 batch_shape[MAX_NDIM];
    batch_shape[0] = b_size;
    for (u64 d = 1; d < src_tensor->ndim; d++) {
      batch_shape[d] = src_tensor->shape[d];
    }
    
    Tensor *batch_tensor = init(meta_arena, data_arena, loader->device, src_tensor->dtype, 
                                batch_shape, src_tensor->ndim, false, NULL);
    
    u64 element_size = dtype_size(src_tensor->dtype);
    u64 sample_numel = 1;
    for (u64 d = 1; d < src_tensor->ndim; d++) {
      sample_numel *= src_tensor->shape[d];
    }
    u64 sample_bytes = sample_numel * element_size;
    
    for (u64 i = 0; i < b_size; i++) {
      u64 sample_idx = loader->indices[loader->current_index + i];
      void *src_ptr = (char *)src_tensor->data + sample_idx * sample_bytes;
      void *dst_ptr = (char *)batch_tensor->data + i * sample_bytes;
      
      if (src_tensor->device == CPU && loader->device == CPU) {
        memcpy(dst_ptr, src_ptr, sample_bytes);
      } else if (src_tensor->device == CPU && loader->device == CUDA) {
#ifdef CUDA_AVAILABLE
        arena_memcpy_h2d_cuda(dst_ptr, src_ptr, sample_bytes);
#else
        fprintf(stderr, "CUDA not available\n");
        exit(EXIT_FAILURE);
#endif
      } else if (src_tensor->device == CUDA && loader->device == CPU) {
#ifdef CUDA_AVAILABLE
        arena_memcpy_d2h_cuda(dst_ptr, src_ptr, sample_bytes);
#else
        fprintf(stderr, "CUDA not available\n");
        exit(EXIT_FAILURE);
#endif
      } else if (src_tensor->device == CUDA && loader->device == CUDA) {
#ifdef CUDA_AVAILABLE
        arena_memcpy_d2d_cuda(dst_ptr, src_ptr, sample_bytes);
#else
        fprintf(stderr, "CUDA not available\n");
        exit(EXIT_FAILURE);
#endif
      }
    }
    
    out_batches[col] = batch_tensor;
  }
  
  loader->current_index += b_size;
  return true;
}
