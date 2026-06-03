#include "kernels/cpu_utils.h"
#include <string.h>

u64 get_offset(const u64 *coords, const u64 *strides, u64 ndim) {
  u64 offset = 0;
  for (u64 i = 0; i < ndim; ++i) {
    offset += coords[i] * strides[i];
  }
  return offset;
}

void linear_to_coords(u64 linear_idx, const u64 *shape, u64 ndim, u64 *coords) {
  for (u64 i = ndim; i-- > 0;) {
    coords[i] = linear_idx % shape[i];
    linear_idx /= shape[i];
  }
}

void compute_view_strides(const u64 *old_shape, const u64 *old_strides, u64 old_ndim,
                          const u64 *new_shape, u64 new_ndim, u64 *new_strides) {
  u64 current_stride = 1;
  for (u64 i = new_ndim; i-- > 0;) {
    new_strides[i] = current_stride;
    current_stride *= new_shape[i];
  }
}

void compute_unsqueeze_shape_strides(const u64 *old_shape, const u64 *old_strides, u64 old_ndim,
                                     u64 axis, u64 *new_shape, u64 *new_strides,
                                     u64 *new_ndim_ptr) {
  *new_ndim_ptr = old_ndim + 1;
  for (u64 i = 0; i < axis; ++i) {
    new_shape[i] = old_shape[i];
    new_strides[i] = old_strides[i];
  }
  new_shape[axis] = 1;
  new_strides[axis] = 0; // Stride for a new dimension of size 1 is 0
  for (u64 i = axis; i < old_ndim; ++i) {
    new_shape[i + 1] = old_shape[i];
    new_strides[i + 1] = old_strides[i];
  }
}

void compute_squeeze_shape_strides(const u64 *old_shape, const u64 *old_strides, u64 old_ndim,
                                   u64 axis, u64 *new_shape, u64 *new_strides, u64 *new_ndim_ptr) {
  *new_ndim_ptr = old_ndim - 1;
  u64 j = 0;
  for (u64 i = 0; i < old_ndim; ++i) {
    if (i == axis) {
      // Skip this dimension
      continue;
    }
    new_shape[j] = old_shape[i];
    new_strides[j] = old_strides[i];
    j++;
  }
}

void compute_expand_strides(const u64 *old_shape, const u64 *old_strides, u64 old_ndim,
                            const u64 *target_shape, u64 target_ndim, u64 *new_strides) {
  // NOTE: Assuming target_ndim >= old_ndim and dimensions are compatible for
  // expansion (i.e., old_shape[i] == 1 or old_shape[i] == target_shape[i])
  u64 offset_diff = target_ndim - old_ndim;
  for (u64 i = 0; i < target_ndim; ++i) {
    if (i < offset_diff) { // New leading dimensions
      new_strides[i] = 0;
    } else {
      u64 old_idx = i - offset_diff;
      if (old_shape[old_idx] == 1 && target_shape[i] > 1) {
        new_strides[i] = 0; // Broadcasted dimension
      } else {
        new_strides[i] = old_strides[old_idx];
      }
    }
  }
}

bool are_shapes_broadcastable(const u64 *shape1, u64 ndim1, const u64 *shape2, u64 ndim2) {
  u64 max_ndim = (ndim1 > ndim2) ? ndim1 : ndim2;
  for (u64 i = 0; i < max_ndim; ++i) {
    u64 dim1 = (i < ndim1) ? shape1[ndim1 - 1 - i] : 1;
    u64 dim2 = (i < ndim2) ? shape2[ndim2 - 1 - i] : 1;
    if (dim1 != dim2 && dim1 != 1 && dim2 != 1) {
      return false;
    }
  }
  return true;
}

void get_broadcast_shape(const u64 *shape1, u64 ndim1, const u64 *shape2, u64 ndim2,
                         u64 *broadcast_shape, u64 *broadcast_ndim_ptr) {
  u64 max_ndim = (ndim1 > ndim2) ? ndim1 : ndim2;
  *broadcast_ndim_ptr = max_ndim;

  for (u64 i = 0; i < max_ndim; ++i) {
    u64 dim1 = (i < ndim1) ? shape1[ndim1 - 1 - i] : 1;
    u64 dim2 = (i < ndim2) ? shape2[ndim2 - 1 - i] : 1;
    broadcast_shape[max_ndim - 1 - i] = (dim1 > dim2) ? dim1 : dim2;
  }
}

void compute_broadcast_strides(const u64 *old_shape, const u64 *old_strides, u64 old_ndim,
                               const u64 *target_shape, u64 target_ndim, u64 *new_strides) {
  // NOTE: This function assumes that old_shape is broadcastable to target_shape
  // and target_ndim is the broadcasted ndim.
  u64 old_idx_offset = target_ndim - old_ndim;
  for (u64 i = 0; i < target_ndim; ++i) {
    if (i < old_idx_offset) { // New leading dimensions
      new_strides[i] = 0;
    } else {
      u64 old_idx = i - old_idx_offset;
      if (old_shape[old_idx] == 1 && target_shape[i] > 1) {
        new_strides[i] = 0; // Broadcasted dimension
      } else {
        new_strides[i] = old_strides[old_idx];
      }
    }
  }
}

void compute_reduction_shape_strides(const u64 *old_shape, u64 old_ndim, u64 dim, bool keepdim,
                                     u64 *new_shape, u64 *new_ndim_ptr, u64 *new_strides) {
  u64 output_ndim = 0;

  if (keepdim) {
    for (u64 i = 0; i < old_ndim; ++i) {
      if (i == dim) {
        new_shape[i] = 1;
      } else {
        new_shape[i] = old_shape[i];
      }
    }
    output_ndim = old_ndim;
  } else {
    for (u64 i = 0; i < old_ndim; ++i) {
      if (i == dim) {
        continue;
      }
      new_shape[output_ndim++] = old_shape[i];
    }
    if (output_ndim == 0) { // Handle case where all dimensions are reduced
      new_shape[0] = 1;
      output_ndim = 1;
    }
  }
  *new_ndim_ptr = output_ndim;

  // Manually compute strides for new_strides
  u64 current_stride = 1;
  for (int i = output_ndim - 1; i >= 0; --i) {
    new_strides[i] = current_stride;
    current_stride *= new_shape[i];
  }
}
