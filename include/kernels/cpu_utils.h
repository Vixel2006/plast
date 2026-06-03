#pragma once

#include "definitions.h"
#include "tensor.h"
#include <stdbool.h>

u64 get_offset(const u64 *coords, const u64 *strides, u64 ndim);

void linear_to_coords(u64 linear_idx, const u64 *shape, u64 ndim, u64 *coords);

void compute_view_strides(const u64 *old_shape, const u64 *old_strides,
                          u64 old_ndim, const u64 *new_shape, u64 new_ndim,
                          u64 *new_strides);

void compute_unsqueeze_shape_strides(const u64 *old_shape,
                                     const u64 *old_strides, u64 old_ndim,
                                     u64 axis, u64 *new_shape, u64 *new_strides,
                                     u64 *new_ndim_ptr);

void compute_squeeze_shape_strides(const u64 *old_shape, const u64 *old_strides,
                                   u64 old_ndim, u64 axis, u64 *new_shape,
                                   u64 *new_strides, u64 *new_ndim_ptr);

void compute_expand_strides(const u64 *old_shape, const u64 *old_strides,
                            u64 old_ndim, const u64 *target_shape,
                            u64 target_ndim, u64 *new_strides);

void compute_broadcast_strides(const u64 *old_shape, const u64 *old_strides,
                               u64 old_ndim, const u64 *target_shape,
                               u64 target_ndim, u64 *new_strides);

bool are_shapes_broadcastable(const u64 *shape1, u64 ndim1, const u64 *shape2,
                              u64 ndim2);

void get_broadcast_shape(const u64 *shape1, u64 ndim1, const u64 *shape2,
                         u64 ndim2, u64 *broadcast_shape,
                         u64 *broadcast_ndim_ptr);

void compute_reduction_shape_strides(const u64 *old_shape, u64 old_ndim,
                                     u64 dim, bool keepdim,
                                     u64 *new_shape, u64 *new_ndim_ptr,
                                     u64 *new_strides);
