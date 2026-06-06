#include "kernels/ops/binary.h"
#include "kernels/cpu_utils.h"
#include "kernels/kernel_macros.h"
#include <omp.h>
#include <immintrin.h>

static inline bool shapes_equal(const u64 *shape1, u64 ndim1, const u64 *shape2, u64 ndim2) {
  if (ndim1 != ndim2)
    return false;
  for (u64 i = 0; i < ndim1; ++i) {
    if (shape1[i] != shape2[i])
      return false;
  }
  return true;
}

// ── ADD Helpers ──
#define ADD_OP(x, y) ((x) + (y))
#define ADD_SIMD_OP(x, y) _mm256_add_ps(x, y)
#define ADD_GRAD_A(grad, x, y) (grad)
#define ADD_GRAD_B(grad, x, y) (grad)
#define ADD_SIMD_GRAD_A(grad, x, y) (grad)
#define ADD_SIMD_GRAD_B(grad, x, y) (grad)

DEFINE_BINARY_CPU_FORWARD(add, ADD_OP, ADD_SIMD_OP)
DEFINE_BINARY_CPU_BACKWARD(add, ADD_GRAD_A, ADD_GRAD_B, ADD_SIMD_GRAD_A, ADD_SIMD_GRAD_B)

// ── SUB Helpers ──
#define SUB_OP(x, y) ((x) - (y))
#define SUB_SIMD_OP(x, y) _mm256_sub_ps(x, y)
#define SUB_GRAD_A(grad, x, y) (grad)
#define SUB_GRAD_B(grad, x, y) (-(grad))
#define SUB_SIMD_GRAD_A(grad, x, y) (grad)
#define SUB_SIMD_GRAD_B(grad, x, y) _mm256_sub_ps(_mm256_set1_ps(0.0f), grad)

DEFINE_BINARY_CPU_FORWARD(sub, SUB_OP, SUB_SIMD_OP)
DEFINE_BINARY_CPU_BACKWARD(sub, SUB_GRAD_A, SUB_GRAD_B, SUB_SIMD_GRAD_A, SUB_SIMD_GRAD_B)

// ── MUL Helpers ──
#define MUL_OP(x, y) ((x) * (y))
#define MUL_SIMD_OP(x, y) _mm256_mul_ps(x, y)
#define MUL_GRAD_A(grad, x, y) ((grad) * (y))
#define MUL_GRAD_B(grad, x, y) ((grad) * (x))
#define MUL_SIMD_GRAD_A(grad, x, y) _mm256_mul_ps(grad, y)
#define MUL_SIMD_GRAD_B(grad, x, y) _mm256_mul_ps(grad, x)

DEFINE_BINARY_CPU_FORWARD(mul, MUL_OP, MUL_SIMD_OP)
DEFINE_BINARY_CPU_BACKWARD(mul, MUL_GRAD_A, MUL_GRAD_B, MUL_SIMD_GRAD_A, MUL_SIMD_GRAD_B)

// ── DIV Helpers ──
#define DIV_OP(x, y) ((x) / (y))
#define DIV_SIMD_OP(x, y) _mm256_div_ps(x, y)
#define DIV_GRAD_A(grad, x, y) ((grad) / (y))
#define DIV_GRAD_B(grad, x, y) (-(grad) * (x) / ((y) * (y)))
#define DIV_SIMD_GRAD_A(grad, x, y) _mm256_div_ps(grad, y)
#define DIV_SIMD_GRAD_B(grad, x, y)                                                                \
  _mm256_div_ps(_mm256_mul_ps(_mm256_sub_ps(_mm256_set1_ps(0.0f), grad), x), _mm256_mul_ps(y, y))

DEFINE_BINARY_CPU_FORWARD(div, DIV_OP, DIV_SIMD_OP)
DEFINE_BINARY_CPU_BACKWARD(div, DIV_GRAD_A, DIV_GRAD_B, DIV_SIMD_GRAD_A, DIV_SIMD_GRAD_B)
