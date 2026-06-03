#include "arena.h"
#include <string.h>

void *arena_alloc_cpu(u64 size) {
  return malloc(size);
}

void arena_free_cpu(void *ptr) {
  free(ptr);
}

void arena_memset_cpu(void *ptr, int value, u64 size) {
  memset(ptr, value, size);
}
void arena_memcpy_h2d_cpu(void *dest, const void *src, u64 size) {
  memcpy(dest, src, size);
}
void arena_memcpy_d2h_cpu(void *dest, const void *src, u64 size) {
  memcpy(dest, src, size);
}
