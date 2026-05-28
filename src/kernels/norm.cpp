#include "kaguya/kernels/special_ops.h"
#include "kaguya/cpu_features.h"

#include <cmath>
#include <algorithm>

#if defined(__AVX512F__)
#include <immintrin.h>
#endif

namespace kaguya::kernels {

// ============================================================================
// Scalar RMSNorm
// ============================================================================

void rmsnorm_scalar(float* out, const float* x, const float* weight, int64_t n, float eps) {
    if (n <= 0) return;

    // Compute sum of squares
    float sum_sq = 0.0f;
    for (int64_t i = 0; i < n; ++i) {
        sum_sq += x[i] * x[i];
    }

    // RMS = sqrt(mean(x^2) + eps)
    const float mean_sq = sum_sq / static_cast<float>(n);
    const float inv_rms = 1.0f / std::sqrt(mean_sq + eps);

    // Normalize and optionally apply weight
    if (weight) {
        for (int64_t i = 0; i < n; ++i) {
            out[i] = (x[i] * inv_rms) * weight[i];
        }
    } else {
        for (int64_t i = 0; i < n; ++i) {
            out[i] = x[i] * inv_rms;
        }
    }
}

// ============================================================================
// AVX-512 RMSNorm
// ============================================================================

#if defined(__AVX512F__)

__attribute__((target("avx512f,avx512vl")))
static void rmsnorm_avx512_impl(float* out, const float* x, const float* weight, int64_t n, float eps) {
    if (n <= 0) return;

    // Compute sum of squares using AVX-512
    __m512 vsum = _mm512_setzero_ps();
    int64_t i = 0;
    for (; i + 16 <= n; i += 16) {
        __m512 vx = _mm512_loadu_ps(&x[i]);
        vsum = _mm512_fmadd_ps(vx, vx, vsum);
    }

    // Scalar tail
    float sum_tail = 0.0f;
    for (; i < n; ++i) {
        sum_tail += x[i] * x[i];
    }

    const float sum_sq = _mm512_reduce_add_ps(vsum) + sum_tail;
    const float mean_sq = sum_sq / static_cast<float>(n);
    const float inv_rms = 1.0f / std::sqrt(mean_sq + eps);
    const __m512 vinv_rms = _mm512_set1_ps(inv_rms);

    // Normalize and apply weight
    i = 0;
    if (weight) {
        for (; i + 16 <= n; i += 16) {
            __m512 vx = _mm512_loadu_ps(&x[i]);
            __m512 vw = _mm512_loadu_ps(&weight[i]);
            __m512 vr = _mm512_mul_ps(_mm512_mul_ps(vx, vinv_rms), vw);
            _mm512_storeu_ps(&out[i], vr);
        }
        for (; i < n; ++i) {
            out[i] = (x[i] * inv_rms) * weight[i];
        }
    } else {
        for (; i + 16 <= n; i += 16) {
            __m512 vx = _mm512_loadu_ps(&x[i]);
            __m512 vr = _mm512_mul_ps(vx, vinv_rms);
            _mm512_storeu_ps(&out[i], vr);
        }
        for (; i < n; ++i) {
            out[i] = x[i] * inv_rms;
        }
    }
}

#endif // __AVX512F__

void rmsnorm_avx512(float* out, const float* x, const float* weight, int64_t n, float eps) {
#if defined(__AVX512F__)
    const auto& info = kaguya::CpuFeatureDetector::get();
    if (info.flags.avx512f && info.flags.os_avx512) {
        rmsnorm_avx512_impl(out, x, weight, n, eps);
        return;
    }
#endif
    rmsnorm_scalar(out, x, weight, n, eps);
}

void rmsnorm_dispatch(float* out, const float* x, const float* weight, int64_t n, float eps) {
    rmsnorm_avx512(out, x, weight, n, eps);
}

// ============================================================================
// Scalar LayerNorm
// ============================================================================

void layernorm_scalar(float* out, const float* x, const float* weight, const float* bias, int64_t n, float eps) {
    if (n <= 0) return;

    // Compute mean
    float sum = 0.0f;
    for (int64_t i = 0; i < n; ++i) {
        sum += x[i];
    }
    const float mean = sum / static_cast<float>(n);

    // Compute variance
    float var_sum = 0.0f;
    for (int64_t i = 0; i < n; ++i) {
        const float d = x[i] - mean;
        var_sum += d * d;
    }
    const float var = var_sum / static_cast<float>(n);
    const float inv_std = 1.0f / std::sqrt(var + eps);

    // Normalize and apply affine transform
    if (weight && bias) {
        for (int64_t i = 0; i < n; ++i) {
            out[i] = ((x[i] - mean) * inv_std) * weight[i] + bias[i];
        }
    } else if (weight) {
        for (int64_t i = 0; i < n; ++i) {
            out[i] = ((x[i] - mean) * inv_std) * weight[i];
        }
    } else if (bias) {
        for (int64_t i = 0; i < n; ++i) {
            out[i] = (x[i] - mean) * inv_std + bias[i];
        }
    } else {
        for (int64_t i = 0; i < n; ++i) {
            out[i] = (x[i] - mean) * inv_std;
        }
    }
}

// ============================================================================
// AVX-512 LayerNorm
// ============================================================================

#if defined(__AVX512F__)

__attribute__((target("avx512f,avx512vl")))
static void layernorm_avx512_impl(float* out, const float* x, const float* weight, const float* bias, int64_t n, float eps) {
    if (n <= 0) return;

    // Compute sum using AVX-512
    __m512 vsum = _mm512_setzero_ps();
    int64_t i = 0;
    for (; i + 16 <= n; i += 16) {
        __m512 vx = _mm512_loadu_ps(&x[i]);
        vsum = _mm512_add_ps(vsum, vx);
    }
    float sum_tail = 0.0f;
    for (; i < n; ++i) {
        sum_tail += x[i];
    }
    const float mean = (_mm512_reduce_add_ps(vsum) + sum_tail) / static_cast<float>(n);
    const __m512 vmean = _mm512_set1_ps(mean);

    // Compute variance using AVX-512
    __m512 vvar_sum = _mm512_setzero_ps();
    i = 0;
    for (; i + 16 <= n; i += 16) {
        __m512 vx = _mm512_loadu_ps(&x[i]);
        __m512 vd = _mm512_sub_ps(vx, vmean);
        vvar_sum = _mm512_fmadd_ps(vd, vd, vvar_sum);
    }
    float var_tail = 0.0f;
    for (; i < n; ++i) {
        const float d = x[i] - mean;
        var_tail += d * d;
    }
    const float var = (_mm512_reduce_add_ps(vvar_sum) + var_tail) / static_cast<float>(n);
    const float inv_std = 1.0f / std::sqrt(var + eps);
    const __m512 vinv_std = _mm512_set1_ps(inv_std);

    // Normalize and apply affine transform
    i = 0;
    if (weight && bias) {
        for (; i + 16 <= n; i += 16) {
            __m512 vx = _mm512_loadu_ps(&x[i]);
            __m512 vw = _mm512_loadu_ps(&weight[i]);
            __m512 vb = _mm512_loadu_ps(&bias[i]);
            __m512 vr = _mm512_fmadd_ps(_mm512_mul_ps(_mm512_sub_ps(vx, vmean), vinv_std), vw, vb);
            _mm512_storeu_ps(&out[i], vr);
        }
        for (; i < n; ++i) {
            out[i] = ((x[i] - mean) * inv_std) * weight[i] + bias[i];
        }
    } else if (weight) {
        for (; i + 16 <= n; i += 16) {
            __m512 vx = _mm512_loadu_ps(&x[i]);
            __m512 vw = _mm512_loadu_ps(&weight[i]);
            __m512 vr = _mm512_mul_ps(_mm512_mul_ps(_mm512_sub_ps(vx, vmean), vinv_std), vw);
            _mm512_storeu_ps(&out[i], vr);
        }
        for (; i < n; ++i) {
            out[i] = ((x[i] - mean) * inv_std) * weight[i];
        }
    } else if (bias) {
        for (; i + 16 <= n; i += 16) {
            __m512 vx = _mm512_loadu_ps(&x[i]);
            __m512 vb = _mm512_loadu_ps(&bias[i]);
            __m512 vr = _mm512_fmadd_ps(_mm512_sub_ps(vx, vmean), vinv_std, vb);
            _mm512_storeu_ps(&out[i], vr);
        }
        for (; i < n; ++i) {
            out[i] = (x[i] - mean) * inv_std + bias[i];
        }
    } else {
        for (; i + 16 <= n; i += 16) {
            __m512 vx = _mm512_loadu_ps(&x[i]);
            __m512 vr = _mm512_mul_ps(_mm512_sub_ps(vx, vmean), vinv_std);
            _mm512_storeu_ps(&out[i], vr);
        }
        for (; i < n; ++i) {
            out[i] = (x[i] - mean) * inv_std;
        }
    }
}

#endif // __AVX512F__

void layernorm_avx512(float* out, const float* x, const float* weight, const float* bias, int64_t n, float eps) {
#if defined(__AVX512F__)
    const auto& info = kaguya::CpuFeatureDetector::get();
    if (info.flags.avx512f && info.flags.os_avx512) {
        layernorm_avx512_impl(out, x, weight, bias, n, eps);
        return;
    }
#endif
    layernorm_scalar(out, x, weight, bias, n, eps);
}

void layernorm_dispatch(float* out, const float* x, const float* weight, const float* bias, int64_t n, float eps) {
    layernorm_avx512(out, x, weight, bias, n, eps);
}

} // namespace kaguya::kernels
