#include "plast/model.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// .plast format:
// [magic: "PLWT" (4 bytes)]
// [version: u32 (1)]
// [num_tensors: u32]
// For each tensor:
//   [name_len: u32]
//   [name: name_len bytes]
//   [ndim: u8]
//   [shape: ndim * u64]
//   [dtype: u8]
//   [data_size: u64]
//   [data: data_size bytes]

#define PLAST_MAGIC "PLWT"
#define PLAST_VERSION 1

static void write_u32(FILE *f, u32 v) {
  fwrite(&v, sizeof(u32), 1, f);
}

static void write_u64(FILE *f, u64 v) {
  fwrite(&v, sizeof(u64), 1, f);
}

static void write_u8(FILE *f, u8 v) {
  fwrite(&v, sizeof(u8), 1, f);
}

static u32 read_u32(FILE *f) {
  u32 v = 0;
  fread(&v, sizeof(u32), 1, f);
  return v;
}

static u64 read_u64(FILE *f) {
  u64 v = 0;
  fread(&v, sizeof(u64), 1, f);
  return v;
}

static u8 read_u8(FILE *f) {
  u8 v = 0;
  fread(&v, sizeof(u8), 1, f);
  return v;
}

void plast_model_save(const PlastModel *m, const char *path) {
  FILE *f = fopen(path, "wb");
  if (!f) {
    fprintf(stderr, "plast: cannot open '%s' for writing\n", path);
    return;
  }

  fwrite(PLAST_MAGIC, 4, 1, f);
  write_u32(f, PLAST_VERSION);
  write_u32(f, (u32)m->num_params);

  for (int i = 0; i < m->num_params; ++i) {
    const Tensor *t = m->params[i];
    u32 name_len = (u32)strlen(m->param_names[i]);
    write_u32(f, name_len);
    fwrite(m->param_names[i], 1, name_len, f);
    write_u8(f, (u8)t->ndim);
    for (u64 j = 0; j < t->ndim; ++j)
      write_u64(f, t->shape[j]);
    write_u8(f, (u8)t->dtype);

    u64 n = numel(t);
    u64 data_size = n * (u64)dtype_size(t->dtype);
    write_u64(f, data_size);
    fwrite(t->data, 1, data_size, f);
  }

  fclose(f);
}

PlastModel *plast_model_load(const char *path, DEVICE device) {
  FILE *f = fopen(path, "rb");
  if (!f) {
    fprintf(stderr, "plast: cannot open '%s' for reading\n", path);
    return NULL;
  }

  char magic[4];
  if (fread(magic, 4, 1, f) != 1 || memcmp(magic, PLAST_MAGIC, 4) != 0) {
    fprintf(stderr, "plast: invalid file format (bad magic)\n");
    fclose(f);
    return NULL;
  }

  u32 version = read_u32(f);
  if (version > PLAST_VERSION) {
    fprintf(stderr, "plast: unsupported version %u\n", (unsigned)version);
    fclose(f);
    return NULL;
  }

  u32 num_tensors = read_u32(f);

  PlastModel *m = plast_model_create(device);
  if (!m) {
    fclose(f);
    return NULL;
  }

  m->compiled = true;
  m->meta = arena_create(Mib(64), CPU);
  m->data = arena_create(Mib(512), device);

  for (u32 i = 0; i < num_tensors; ++i) {
    u32 name_len = read_u32(f);
    char name[PLAST_MAX_NAME];
    if (name_len >= PLAST_MAX_NAME)
      name_len = PLAST_MAX_NAME - 1;
    fread(name, 1, name_len, f);
    name[name_len] = '\0';

    u8 ndim = read_u8(f);
    u64 shape[MAX_NDIM] = {0};
    for (u8 j = 0; j < ndim; ++j)
      shape[j] = read_u64(f);

    u8 dtype = read_u8(f);
    u64 data_size = read_u64(f);

    // Allocate tensor in arena
    u64 *strides = compute_strides(shape, ndim);
    Tensor *t = arena_tensor_alloc(&m->meta, &m->data, shape, ndim, strides, (DTYPE)dtype, false,
                                   NULL, device);
    free(strides);

    // Read data
    fread(t->data, 1, data_size, f);

    // Register as parameter
    if (m->num_params < PLAST_MAX_PARAMS) {
      m->params[m->num_params] = t;
      snprintf(m->param_names[m->num_params], PLAST_MAX_NAME, "%s", name);
      m->num_params++;
    }
  }

  fclose(f);
  return m;
}
