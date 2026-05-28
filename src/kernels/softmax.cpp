#include "kaguya/kernels/special_ops.h"
#include "kaguya/cpu_features.h"

#include <cmath>
#include <algorithm>

#if defined(__AVX512F__)
#include <immintrin.h>
#endif

namespace kaguya::kernels {

// ============================================================================
// Scalar Softmax
// ============================================================================

void softmax_scalar(float* x, int64_t n, float temperature) {
    if (n <= 0) return;

    // Apply temperature and find max
    float max_val = x[0];
    if (temperature != 1.0f) {
        const float inv_temp = 1.0f / temperature;
        for (int64_t i = 0; i < n; ++i) {
            x[i] *= inv_temp;
        }
    }
    for (int64_t i = 1; i < n; ++i) {
        max_val = std::max(max_val, x[i]);
    }

    // Subtract max and exp
    float sum = 0.0f;
    for (int64_t i = 0; i < n; ++i) {
        x[i] = std::exp(x[i] - max_val);
        sum += x[i];
    }

    // Divide by sum
    const float inv_sum = 1.0f / sum;
    for (int64_t i = 0; i < n; ++i) {
        x[i] *= inv_sum;
    }
}

// ============================================================================
// AVX-512 Softmax
// ============================================================================

#if defined(__AVX512F__)

/// Vectorized exp(x) approximation using 2^(x/ln2) approach.
/// Uses Taylor series of exp(f) for f in [-ln2/2, ln2/2] ≈ [-0.347, 0.347].
///
/// Algorithm:
///   t = round(x / ln2)
///   f = x - t * ln2   (remainder, |f| ≤ ln2/2)
///   exp(x) = 2^t * exp(f)
///   exp(f) ≈ 1 + f + f²/2! + f³/3! + f⁴/4! + f⁵/5! + f⁶/6!
///
/// Accurate to ~1e-7 for single-precision float inputs.
__attribute__((target("avx512f,avx512vl")))
static inline __m512 exp_avx512(__m512 x) {
    // Clamp x to avoid overflow/underflow in scalef
    x = _mm512_min_ps(x, _mm512_set1_ps(88.0f));
    x = _mm512_max_ps(x, _mm512_set1_ps(-88.0f));

    const float ln2 = 0.6931471805599453f;
    const float inv_ln2 = 1.4426950408889634f;

    // t = round(x / ln2)
    __m512 t = _mm512_mul_ps(x, _mm512_set1_ps(inv_ln2));
    t = _mm512_roundscale_ps(t, _MM_FROUND_TO_NEAREST_INT);

    // f = x - t * ln2  (remainder in [-ln2/2, ln2/2])
    __m512 f = _mm512_fnmadd_ps(t, _mm512_set1_ps(ln2), x);

    // Taylor expansion of exp(f) around 0 (6th order):
    // exp(f) = 1 + f*(1 + f*(1/2 + f*(1/6 + f*(1/24 + f*(1/120 + f/720)))))
    __m512 p = _mm512_set1_ps(1.388888888888889e-3f);   // 1/720
    p = _mm512_fmadd_ps(p, f, _mm512_set1_ps(8.333333333333333e-3f));  // 1/120
    p = _mm512_fmadd_ps(p, f, _mm512_set1_ps(4.166666666666667e-2f));  // 1/24
    p = _mm512_fmadd_ps(p, f, _mm512_set1_ps(1.666666666666667e-1f));  // 1/6
    p = _mm512_fmadd_ps(p, f, _mm512_set1_ps(5.000000000000000e-1f));  // 1/2
    p = _mm512_fmadd_ps(p, f, _mm512_set1_ps(1.0f));                    // 1
    p = _mm512_fmadd_ps(p, f, _mm512_set1_ps(1.0f));                    // +1 (constant)

    // result = 2^t * exp(f) = exp(x)
    return _mm512_scalef_ps(p, t);
}

__attribute__((target("avx512f,avx512vl")))
static void softmax_avx512_impl(float* x, int64_t n, float temperature) {
    if (n <= 0) return;

    // Apply temperature
    if (temperature != 1.0f) {
        const float inv_temp = 1.0f / temperature;
        const __m512 vinv_temp = _mm512_set1_ps(inv_temp);
        int64_t i = 0;
        for (; i + 16 <= n; i += 16) {
            __m512 v = _mm512_loadu_ps(&x[i]);
            v = _mm512_mul_ps(v, vinv_temp);
            _mm512_storeu_ps(&x[i], v);
        }
        for (; i < n; ++i) {
            x[i] *= inv_temp;
        }
    }

    // Find max using AVX-512 reduction
    float max_val;
    int64_t i = 0;
    if (n >= 16) {
        __m512 vmax = _mm512_loadu_ps(&x[0]);
        for (i = 16; i + 16 <= n; i += 16) {
            __m512 v = _mm512_loadu_ps(&x[i]);
            vmax = _mm512_max_ps(vmax, v);
        }
        max_val = _mm512_reduce_max_ps(vmax);
        // Handle remaining elements
        for (; i < n; ++i) {
            max_val = std::max(max_val, x[i]);
        }
    } else {
        max_val = x[0];
        for (i = 1; i < n; ++i) {
            max_val = std::max(max_val, x[i]);
        }
    }

    // Subtract max and compute exp
    const __m512 vmax = _mm512_set1_ps(max_val);
    __m512 vsum = _mm512_setzero_ps();
    i = 0;
    for (; i + 16 <= n; i += 16) {
        __m512 v = _mm512_loadu_ps(&x[i]);
        v = _mm512_sub_ps(v, vmax);
        v = exp_avx512(v);
        _mm512_storeu_ps(&x[i], v);
        vsum = _mm512_add_ps(vsum, v);
    }

    // Scalar tail for exp
    float sum_tail = 0.0f;
    for (; i < n; ++i) {
        x[i] = std::exp(x[i] - max_val);
        sum_tail += x[i];
    }

    // Total sum
    float sum = _mm512_reduce_add_ps(vsum) + sum_tail;
    const float inv_sum = 1.0f / sum;
    const __m512 vinv_sum = _mm512_set1_ps(inv_sum);

    // Divide by sum
    i = 0;
    for (; i + 16 <= n; i += 16) {
        __m512 v = _mm512_loadu_ps(&x[i]);
        v = _mm512_mul_ps(v, vinv_sum);
        _mm512_storeu_ps(&x[i], v);
    }
    for (; i < n; ++i) {
        x[i] *= inv_sum;
    }
}

#endif // __AVX512F__

void softmax_avx512(float* x, int64_t n, float temperature) {
#if defined(__AVX512F__)
    const auto& info = kaguya::CpuFeatureDetector::get();
    if (info.flags.avx512f && info.flags.os_avx512) {
        softmax_avx512_impl(x, n, temperature);
        return;
    }
#endif
    softmax_scalar(x, n, temperature);
}

void softmax_dispatch(float* x, int64_t n, float temperature) {
    softmax_avx512(x, n, temperature);
}

} // namespace kaguya::kernels
