#pragma once

#include "core/definitions.h"
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>

#define Kib(n) ((u64)(n) << 10)
#define Mib(n) ((u64)(n) << 20)
#define Gib(n) ((u64)(n) << 30)

#define MIN(a, b) (((a) < (b)) ? (a) : (b))
#define MAX(a, b) (((a) > (b)) ? (a) : (b))
#define ALIGN_UP(n, p) (((u64)(n) + ((u64)(p) - 1)) & (~((u64)(p) - 1)))

typedef enum DEVICE { CPU, CUDA } DEVICE;

typedef struct ArenaBlock {
  struct ArenaBlock *prev;
  u64 capacity;
  u64 offset;
  void *buffer;
} ArenaBlock;

typedef struct Arena {
  ArenaBlock *current;
  DEVICE device;
} Arena;

#ifdef __cplusplus
extern "C" {
#endif

Arena arena_create(u64 capacity, DEVICE device);
void arena_release(Arena *a);
void arena_reset(Arena *a);
void *arena_alloc(Arena *a, u64 size, u64 align);
void arena_memcpy_h2d(Arena *a, void *dest, const void *src, u64 size);
void arena_memcpy_d2h(Arena *a, void *dest, const void *src, u64 size);

#ifdef __cplusplus
}
#endif
