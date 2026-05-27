#include "kaguya/kernels/special_ops.h"
#include "kaguya/cpu_features.h"

#include <cmath>
#include <algorithm>

#if defined(__AVX512F__)
#include <immintrin.h>
#endif

namespace kaguya::kernels {

// ============================================================================
// Scalar SiLU: x[i] = x[i] / (1.0f + expf(-x[i]))
// ============================================================================

void silu_scalar(float* x, int64_t n) {
    for (int64_t i = 0; i < n; ++i) {
        x[i] = x[i] / (1.0f + std::exp(-x[i]));
    }
}

// ============================================================================
// Scalar GELU (approximate): 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
// ============================================================================

void gelu_scalar(float* x, int64_t n) {
    constexpr float sqrt_2_over_pi = 0.7978845608028654f;  // sqrt(2/pi)
    constexpr float coeff = 0.044715f;

    for (int64_t i = 0; i < n; ++i) {
        const float v = x[i];
        const float inner = sqrt_2_over_pi * (v + coeff * v * v * v);
        x[i] = 0.5f * v * (1.0f + std::tanh(inner));
    }
}

// ============================================================================
// AVX-512 implementations
// ============================================================================

#if defined(__AVX512F__)

/// Vectorized exp(x) approximation using 2^(x/ln2) approach.
/// Same implementation as softmax.cpp — Taylor series of exp(f) for f in [-ln2/2, ln2/2].
__attribute__((target("avx512f,avx512vl")))
static inline __m512 exp_avx512_act(__m512 x) {
    x = _mm512_min_ps(x, _mm512_set1_ps(88.0f));
    x = _mm512_max_ps(x, _mm512_set1_ps(-88.0f));

    const float ln2 = 0.6931471805599453f;
    const float inv_ln2 = 1.4426950408889634f;

    __m512 t = _mm512_mul_ps(x, _mm512_set1_ps(inv_ln2));
    t = _mm512_roundscale_ps(t, _MM_FROUND_TO_NEAREST_INT);

    __m512 f = _mm512_fnmadd_ps(t, _mm512_set1_ps(ln2), x);

    // Taylor expansion of exp(f) around 0 (6th order)
    __m512 p = _mm512_set1_ps(1.388888888888889e-3f);   // 1/720
    p = _mm512_fmadd_ps(p, f, _mm512_set1_ps(8.333333333333333e-3f));  // 1/120
    p = _mm512_fmadd_ps(p, f, _mm512_set1_ps(4.166666666666667e-2f));  // 1/24
    p = _mm512_fmadd_ps(p, f, _mm512_set1_ps(1.666666666666667e-1f));  // 1/6
    p = _mm512_fmadd_ps(p, f, _mm512_set1_ps(5.000000000000000e-1f));  // 1/2
    p = _mm512_fmadd_ps(p, f, _mm512_set1_ps(1.0f));                    // 1
    p = _mm512_fmadd_ps(p, f, _mm512_set1_ps(1.0f));                    // +1 (constant)

    return _mm512_scalef_ps(p, t);
}

/// Vectorized tanh(x) using the identity: tanh(x) = 1 - 2 / (1 + exp(2x))
/// This leverages the accurate exp_avx512_act for precision.
__attribute__((target("avx512f,avx512vl")))
static inline __m512 tanh_avx512(__m512 x) {
    const __m512 vtwo = _mm512_set1_ps(2.0f);
    const __m512 vone = _mm512_set1_ps(1.0f);

    // tanh(x) = 1 - 2 / (1 + exp(2x))
    __m512 exp2x = exp_avx512_act(_mm512_mul_ps(vtwo, x));
    __m512 denom = _mm512_add_ps(vone, exp2x);
    __m512 result = _mm512_fnmadd_ps(vtwo, _mm512_rcp14_ps(denom), vone);
    // Use rcp14 for speed, then do one Newton-Raphson refinement
    __m512 rcp = _mm512_rcp14_ps(denom);
    rcp = _mm512_mul_ps(rcp, _mm512_fnmadd_ps(denom, rcp, vtwo));  // rcp = rcp * (2 - denom * rcp)
    result = _mm512_fnmadd_ps(vtwo, rcp, vone);

    return result;
}

__attribute__((target("avx512f,avx512vl")))
static void silu_avx512_impl(float* x, int64_t n) {
    const __m512 vone = _mm512_set1_ps(1.0f);

    int64_t i = 0;
    for (; i + 16 <= n; i += 16) {
        __m512 vx = _mm512_loadu_ps(&x[i]);
        __m512 vneg = _mm512_sub_ps(_mm512_setzero_ps(), vx);
        __m512 vexp = exp_avx512_act(vneg);
        __m512 vsigmoid = _mm512_div_ps(vone, _mm512_add_ps(vone, vexp));
        __m512 vr = _mm512_mul_ps(vx, vsigmoid);
        _mm512_storeu_ps(&x[i], vr);
    }
    // Scalar tail
    for (; i < n; ++i) {
        x[i] = x[i] / (1.0f + std::exp(-x[i]));
    }
}

__attribute__((target("avx512f,avx512vl")))
static void gelu_avx512_impl(float* x, int64_t n) {
    const __m512 sqrt_2_over_pi = _mm512_set1_ps(0.7978845608028654f);
    const __m512 coeff = _mm512_set1_ps(0.044715f);
    const __m512 half = _mm512_set1_ps(0.5f);
    const __m512 one = _mm512_set1_ps(1.0f);

    int64_t i = 0;
    for (; i + 16 <= n; i += 16) {
        __m512 vx = _mm512_loadu_ps(&x[i]);
        // inner = sqrt(2/pi) * (x + 0.044715 * x^3)
        __m512 vx3 = _mm512_mul_ps(vx, _mm512_mul_ps(vx, vx));
        __m512 vinner = _mm512_mul_ps(sqrt_2_over_pi,
            _mm512_fmadd_ps(coeff, vx3, vx));
        __m512 vtanh = tanh_avx512(vinner);
        __m512 vr = _mm512_mul_ps(half, _mm512_mul_ps(vx, _mm512_add_ps(one, vtanh)));
        _mm512_storeu_ps(&x[i], vr);
    }
    // Scalar tail
    constexpr float c_sqrt_2_over_pi = 0.7978845608028654f;
    constexpr float c_coeff = 0.044715f;
    for (; i < n; ++i) {
        const float v = x[i];
        const float inner = c_sqrt_2_over_pi * (v + c_coeff * v * v * v);
        x[i] = 0.5f * v * (1.0f + std::tanh(inner));
    }
}

#endif // __AVX512F__

// ============================================================================
// Dispatch functions
// ============================================================================

void silu_avx512(float* x, int64_t n) {
#if defined(__AVX512F__)
    const auto& info = kaguya::CpuFeatureDetector::get();
    if (info.flags.avx512f && info.flags.os_avx512) {
        silu_avx512_impl(x, n);
        return;
    }
#endif
    silu_scalar(x, n);
}

void silu_dispatch(float* x, int64_t n) {
    silu_avx512(x, n);
}

void gelu_avx512(float* x, int64_t n) {
#if defined(__AVX512F__)
    const auto& info = kaguya::CpuFeatureDetector::get();
    if (info.flags.avx512f && info.flags.os_avx512) {
        gelu_avx512_impl(x, n);
        return;
    }
#endif
    gelu_scalar(x, n);
}

void gelu_dispatch(float* x, int64_t n) {
    gelu_avx512(x, n);
}

} // namespace kaguya::kernels
