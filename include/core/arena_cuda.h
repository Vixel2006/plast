#pragma once

#include "core/definitions.h"

#ifdef __cplusplus
extern "C" {
#endif

void *arena_alloc_cuda(u64 size);
void arena_free_cuda(void *ptr);
void arena_memset_cuda(void *ptr, int value, u64 size);
void arena_memcpy_h2d_cuda(void *dest, const void *src, u64 size);
void arena_memcpy_d2h_cuda(void *dest, const void *src, u64 size);

#ifdef __cplusplus
}
#endif
