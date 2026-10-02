#pragma once

#include "data/dataset.h"

typedef struct DataLoader {
  TensorDataset *dataset;
  u64 batch_size;
  bool shuffle;
  bool drop_last;
  DEVICE device;
  u64 *indices;
  u64 current_index;
} DataLoader;

#ifdef __cplusplus
extern "C" {
#endif

DataLoader *create_dataloader(TensorDataset *dataset, u64 batch_size, bool shuffle, bool drop_last, DEVICE device);
void free_dataloader(DataLoader *loader);
void reset_dataloader_iterator(DataLoader *loader);
bool dataloader_next_batch(DataLoader *loader, Arena *meta_arena, Arena *data_arena, Tensor **out_batches);

#ifdef __cplusplus
}
#endif
