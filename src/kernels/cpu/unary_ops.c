#include "kernels/ops/unary.h"
#include "kernels/cpu_utils.h"
#include "kernels/kernel_macros.h"
#include <math.h>
#include <omp.h>
#include <immintrin.h>

// Helper math operations for non-contiguous macros
#define OP_SIN(x) sinf(x)
#define GRAD_SIN(dout, x) ((dout) * cosf(x))

#define OP_COS(x) cosf(x)
#define GRAD_COS(dout, x) ((dout) * (-sinf(x)))

#define OP_TAN(x) tanf(x)
#define GRAD_TAN(dout, x) ((dout) * (1.0f + tanf(x) * tanf(x)))

#define OP_EXP(x) expf(x)
#define GRAD_EXP(dout, x) ((dout) * expf(x))

#define OP_LOG(x) logf(x)
#define GRAD_LOG(dout, x) ((dout) * (1.0f / (x)))

#define OP_NEG(x) (-(x))
#define GRAD_NEG(dout, x) (-(dout))

#define OP_ABS(x) fabsf(x)
#define GRAD_ABS(dout, x) (((x) > 0) ? (dout) : (((x) < 0) ? -(dout) : 0.0f))

#define OP_LEAKY_RELU(x, alpha) (((x) > 0) ? (x) : ((x) * (alpha)))
#define GRAD_LEAKY_RELU(dout, x, alpha) (((x) > 0) ? (dout) : ((dout) * (alpha)))

// ── sin ──
void sin_cpu_forward_float_contig_kernel(const float *a, float *c, u64 num_elements) {
  for (u64 i = 0; i < num_elements; ++i) {
    c[i] = sinf(a[i]);
  }
}
DEFINE_UNARY_NONCONTIG_FORWARD(sin_cpu_forward_float_non_contig_kernel, OP_SIN)

void sin_cpu_backward_float_contig_kernel(const float *dout, const float *a, float *da,
                                          u64 num_elements) {
  for (u64 i = 0; i < num_elements; ++i) {
    if (da)
      da[i] += dout[i] * cosf(a[i]);
  }
}
DEFINE_UNARY_NONCONTIG_BACKWARD(sin_cpu_backward_float_non_contig_kernel, GRAD_SIN)

DEFINE_UNARY_CPU_FORWARD(sin)
DEFINE_UNARY_CPU_BACKWARD(sin)

// ── cos ──
void cos_cpu_forward_float_contig_kernel(const float *a, float *c, u64 num_elements) {
  for (u64 i = 0; i < num_elements; ++i) {
    c[i] = cosf(a[i]);
  }
}
DEFINE_UNARY_NONCONTIG_FORWARD(cos_cpu_forward_float_non_contig_kernel, OP_COS)

void cos_cpu_backward_float_contig_kernel(const float *dout, const float *a, float *da,
                                          u64 num_elements) {
  for (u64 i = 0; i < num_elements; ++i) {
    if (da)
      da[i] += dout[i] * (-sinf(a[i]));
  }
}
DEFINE_UNARY_NONCONTIG_BACKWARD(cos_cpu_backward_float_non_contig_kernel, GRAD_COS)

DEFINE_UNARY_CPU_FORWARD(cos)
DEFINE_UNARY_CPU_BACKWARD(cos)

// ── tan ──
void tan_cpu_forward_float_contig_kernel(const float *a, float *c, u64 num_elements) {
  for (u64 i = 0; i < num_elements; ++i) {
    c[i] = tanf(a[i]);
  }
}
DEFINE_UNARY_NONCONTIG_FORWARD(tan_cpu_forward_float_non_contig_kernel, OP_TAN)

void tan_cpu_backward_float_contig_kernel(const float *dout, const float *a, float *da,
                                          u64 num_elements) {
  for (u64 i = 0; i < num_elements; ++i) {
    if (da) {
      float t = tanf(a[i]);
      da[i] += dout[i] * (1.0f + t * t);
    }
  }
}
DEFINE_UNARY_NONCONTIG_BACKWARD(tan_cpu_backward_float_non_contig_kernel, GRAD_TAN)

DEFINE_UNARY_CPU_FORWARD(tan)
DEFINE_UNARY_CPU_BACKWARD(tan)

// ── exp ──
void exp_cpu_forward_float_contig_kernel(const float *a, float *c, u64 num_elements) {
  for (u64 i = 0; i < num_elements; ++i) {
    c[i] = expf(a[i]);
  }
}
DEFINE_UNARY_NONCONTIG_FORWARD(exp_cpu_forward_float_non_contig_kernel, OP_EXP)

void exp_cpu_backward_float_contig_kernel(const float *dout, const float *a, float *da,
                                          u64 num_elements) {
  for (u64 i = 0; i < num_elements; ++i) {
    if (da)
      da[i] += dout[i] * expf(a[i]);
  }
}
DEFINE_UNARY_NONCONTIG_BACKWARD(exp_cpu_backward_float_non_contig_kernel, GRAD_EXP)

DEFINE_UNARY_CPU_FORWARD(exp)
DEFINE_UNARY_CPU_BACKWARD(exp)

// ── log ──
void log_cpu_forward_float_contig_kernel(const float *a, float *c, u64 num_elements) {
  for (u64 i = 0; i < num_elements; ++i) {
    c[i] = logf(a[i]);
  }
}
DEFINE_UNARY_NONCONTIG_FORWARD(log_cpu_forward_float_non_contig_kernel, OP_LOG)

void log_cpu_backward_float_contig_kernel(const float *dout, const float *a, float *da,
                                          u64 num_elements) {
  for (u64 i = 0; i < num_elements; ++i) {
    if (da)
      da[i] += dout[i] * (1.0f / a[i]);
  }
}
DEFINE_UNARY_NONCONTIG_BACKWARD(log_cpu_backward_float_non_contig_kernel, GRAD_LOG)

DEFINE_UNARY_CPU_FORWARD(log)
DEFINE_UNARY_CPU_BACKWARD(log)

// ── neg ──
void neg_cpu_forward_float_contig_kernel(const float *a, float *c, u64 num_elements) {
  u64 i = 0;
  for (; i + SIMD_WIDTH - 1 < num_elements; i += SIMD_WIDTH) {
    __m256 x = _mm256_loadu_ps(a + i);
    __m256 z = _mm256_sub_ps(_mm256_set1_ps(0.0f), x);
    _mm256_storeu_ps(c + i, z);
  }
  for (; i < num_elements; ++i) {
    c[i] = -a[i];
  }
}
DEFINE_UNARY_NONCONTIG_FORWARD(neg_cpu_forward_float_non_contig_kernel, OP_NEG)

void neg_cpu_backward_float_contig_kernel(const float *dout, const float *a, float *da,
                                          u64 num_elements) {
  u64 i = 0;
  for (; i + SIMD_WIDTH - 1 < num_elements; i += SIMD_WIDTH) {
    __m256 out_grad = _mm256_loadu_ps(dout + i);
    if (da) {
      __m256 a_grad = _mm256_loadu_ps(da + i);
      __m256 new_grad = _mm256_sub_ps(a_grad, out_grad);
      _mm256_storeu_ps(da + i, new_grad);
    }
  }
  for (; i < num_elements; ++i) {
    if (da)
      da[i] -= dout[i];
  }
}
DEFINE_UNARY_NONCONTIG_BACKWARD(neg_cpu_backward_float_non_contig_kernel, GRAD_NEG)

DEFINE_UNARY_CPU_FORWARD(neg)
DEFINE_UNARY_CPU_BACKWARD(neg)

// ── abs ──
void abs_cpu_forward_float_contig_kernel(const float *a, float *c, u64 num_elements) {
  u64 i = 0;
  for (; i + SIMD_WIDTH - 1 < num_elements; i += SIMD_WIDTH) {
    __m256 x = _mm256_loadu_ps(a + i);
    __m256 abs_mask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7fffffff));
    __m256 y = _mm256_and_ps(x, abs_mask);
    _mm256_storeu_ps(c + i, y);
  }
  for (; i < num_elements; ++i) {
    c[i] = fabsf(a[i]);
  }
}
DEFINE_UNARY_NONCONTIG_FORWARD(abs_cpu_forward_float_non_contig_kernel, OP_ABS)

void abs_cpu_backward_float_contig_kernel(const float *dout, const float *a, float *da,
                                          u64 num_elements) {
  u64 i = 0;
  __m256 zero_vec = _mm256_set1_ps(0.0f);
  __m256 one_vec = _mm256_set1_ps(1.0f);
  __m256 neg_one_vec = _mm256_set1_ps(-1.0f);
  for (; i + SIMD_WIDTH - 1 < num_elements; i += SIMD_WIDTH) {
    __m256 x = _mm256_loadu_ps(a + i);
    __m256 out_grad = _mm256_loadu_ps(dout + i);
    __m256 mask_gt_zero = _mm256_cmp_ps(x, zero_vec, _CMP_GT_OQ);
    __m256 mask_lt_zero = _mm256_cmp_ps(x, zero_vec, _CMP_LT_OQ);
    __m256 grad_multiplier = _mm256_set1_ps(0.0f);
    grad_multiplier = _mm256_blendv_ps(grad_multiplier, one_vec, mask_gt_zero);
    grad_multiplier = _mm256_blendv_ps(grad_multiplier, neg_one_vec, mask_lt_zero);
    if (da) {
      __m256 a_grad = _mm256_loadu_ps(da + i);
      __m256 new_grad = _mm256_add_ps(a_grad, _mm256_mul_ps(out_grad, grad_multiplier));
      _mm256_storeu_ps(da + i, new_grad);
    }
  }
  for (; i < num_elements; ++i) {
    if (da) {
      float grad_multiplier = (a[i] > 0) ? 1.0f : ((a[i] < 0) ? -1.0f : 0.0f);
      da[i] += dout[i] * grad_multiplier;
    }
  }
}
DEFINE_UNARY_NONCONTIG_BACKWARD(abs_cpu_backward_float_non_contig_kernel, GRAD_ABS)

DEFINE_UNARY_CPU_FORWARD(abs)
DEFINE_UNARY_CPU_BACKWARD(abs)

// ── leaky_relu ──
void leaky_relu_cpu_forward_float_contig_kernel(const float *a, float *c, u64 num_elements,
                                                float alpha) {
  u64 i = 0;
  __m256 alpha_vec = _mm256_set1_ps(alpha);
  __m256 zero_vec = _mm256_set1_ps(0.0f);
  for (; i + SIMD_WIDTH - 1 < num_elements; i += SIMD_WIDTH) {
    __m256 x = _mm256_loadu_ps(a + i);
    __m256 mask = _mm256_cmp_ps(x, zero_vec, _CMP_GT_OQ);
    __m256 res = _mm256_blendv_ps(_mm256_mul_ps(x, alpha_vec), x, mask);
    _mm256_storeu_ps(c + i, res);
  }
  for (; i < num_elements; ++i) {
    c[i] = a[i] > 0 ? a[i] : a[i] * alpha;
  }
}
DEFINE_UNARY_NONCONTIG_FORWARD_PARAM(leaky_relu_cpu_forward_float_non_contig_kernel, OP_LEAKY_RELU,
                                     float, alpha)

void leaky_relu_cpu_backward_float_contig_kernel(const float *dout, const float *a, float *da,
                                                 u64 num_elements, float alpha) {
  u64 i = 0;
  __m256 alpha_vec = _mm256_set1_ps(alpha);
  __m256 one_vec = _mm256_set1_ps(1.0f);
  __m256 zero_vec = _mm256_set1_ps(0.0f);
  for (; i + SIMD_WIDTH - 1 < num_elements; i += SIMD_WIDTH) {
    __m256 x = _mm256_loadu_ps(a + i);
    __m256 out_grad = _mm256_loadu_ps(dout + i);
    __m256 mask = _mm256_cmp_ps(x, zero_vec, _CMP_GT_OQ);
    __m256 grad_multiplier = _mm256_blendv_ps(alpha_vec, one_vec, mask);
    if (da) {
      __m256 a_grad = _mm256_loadu_ps(da + i);
      __m256 new_grad = _mm256_add_ps(a_grad, _mm256_mul_ps(out_grad, grad_multiplier));
      _mm256_storeu_ps(da + i, new_grad);
    }
  }
  for (; i < num_elements; ++i) {
    if (da) {
      float grad_multiplier = a[i] > 0 ? 1.0f : alpha;
      da[i] += dout[i] * grad_multiplier;
    }
  }
}
DEFINE_UNARY_NONCONTIG_BACKWARD_PARAM(leaky_relu_cpu_backward_float_non_contig_kernel,
                                      GRAD_LEAKY_RELU, float, alpha)

DEFINE_UNARY_CPU_FORWARD_PARAM(leaky_relu, params.fval)
DEFINE_UNARY_CPU_BACKWARD_PARAM(leaky_relu, params.fval)
