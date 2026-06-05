#pragma once

#include "core/definitions.h" // For u64, etc.

void *arena_alloc_cpu(u64 size);
void arena_free_cpu(void *ptr);
void arena_memset_cpu(void *ptr, int value, u64 size);
void arena_memcpy_h2d_cpu(void *dest, const void *src, u64 size);
void arena_memcpy_d2h_cpu(void *dest, const void *src, u64 size);
