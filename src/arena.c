#include "arena.h"
#include "arena_cpu.h"
#ifdef CUDA_AVAILABLE
#include "arena_cuda.h"
#endif
#include <string.h>

Arena arena_create(u64 capacity, DEVICE device) {
  Arena a = {0};
  a.device = device;
  ArenaBlock *b = (ArenaBlock *)malloc(sizeof(ArenaBlock));

  if (b) {
    b->prev = NULL;
    b->capacity = capacity;
    b->offset = 0;
    if (device == CPU) {
      b->buffer = arena_alloc_cpu(capacity);
    }
#ifdef CUDA_AVAILABLE
    else {
      b->buffer = arena_alloc_cuda(capacity);
    }
#endif
    a.current = b;
  }
  return a;
}

void arena_release(Arena *a) {
  ArenaBlock *curr = a->current;
  while (curr != NULL) {
    ArenaBlock *prev = curr->prev;
    if (a->device == CPU) {
      arena_free_cpu(curr->buffer);
    }
#ifdef CUDA_AVAILABLE
    else {
      arena_free_cuda(curr->buffer);
    }
#endif
    free(curr);
    curr = prev;
  }
  a->current = NULL;
}

void arena_reset(Arena *a) {
  ArenaBlock *curr = a->current;
  while (curr->prev != NULL) {
    ArenaBlock *prev = curr->prev;
    if (a->device == CPU) {
      arena_free_cpu(curr->buffer);
    }
#ifdef CUDA_AVAILABLE
    else {
      arena_free_cuda(curr->buffer);
    }
#endif
    free(curr);
    curr = prev;
  }
  a->current = curr;
  a->current->offset = 0;
}

void *arena_alloc(Arena *a, u64 size, u64 align) {
  ArenaBlock *curr = a->current;
  u64 next_pointer = ALIGN_UP(curr->offset, align);

  if (next_pointer + size > curr->capacity) {
    u64 new_capacity = MAX(curr->capacity, size);
    ArenaBlock *new_block = (ArenaBlock *)malloc(sizeof(ArenaBlock));
    if (!new_block)
      return NULL;

    new_block->prev = curr;
    new_block->capacity = new_capacity;
    new_block->offset = 0;
    if (a->device == CPU) {
      new_block->buffer = arena_alloc_cpu(new_capacity);
    }
#ifdef CUDA_AVAILABLE
    else {
      new_block->buffer = arena_alloc_cuda(new_capacity);
    }
#endif

    a->current = new_block;
    curr = new_block;
    next_pointer = ALIGN_UP(0, align);
  }

  void *ptr = (char *)curr->buffer + next_pointer;
  if (a->device == CPU) {
    arena_memset_cpu(ptr, 0, size);
  }
#ifdef CUDA_AVAILABLE
  else {
    arena_memset_cuda(ptr, 0, size);
  }
#endif
  curr->offset = next_pointer + size;

  return ptr;
}

void arena_memcpy_h2d(Arena *a, void *dest, const void *src, u64 size) {
  if (a->device == CPU) {
    arena_memcpy_h2d_cpu(dest, src, size);
  }
#ifdef CUDA_AVAILABLE
  else {
    arena_memcpy_h2d_cuda(dest, src, size);
  }
#endif
}

void arena_memcpy_d2h(Arena *a, void *dest, const void *src, u64 size) {
  if (a->device == CPU) {
    arena_memcpy_d2h_cpu(dest, src, size);
  }
#ifdef CUDA_AVAILABLE
  else {
    arena_memcpy_d2h_cuda(dest, src, size);
  }
#endif
}
