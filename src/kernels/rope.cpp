#include "kaguya/kernels/special_ops.h"
#include "kaguya/cpu_features.h"

#include <cmath>
#include <algorithm>

#if defined(__AVX512F__)
#include <immintrin.h>
#endif

namespace kaguya::kernels {

// ============================================================================
// Scalar RoPE
// ============================================================================

void rope_scalar(float* x, int64_t n_heads, int64_t head_dim, int64_t pos,
                 float freq_base, float freq_scale) {
    if (n_heads <= 0 || head_dim <= 0) return;

    const int64_t half_dim = head_dim / 2;

    for (int64_t h = 0; h < n_heads; ++h) {
        float* head = x + h * head_dim;

        for (int64_t i = 0; i < half_dim; ++i) {
            // theta = pos * freq_base^(-2i/head_dim) * freq_scale
            const float theta = static_cast<float>(pos) *
                std::pow(freq_base, -2.0f * static_cast<float>(i) / static_cast<float>(head_dim)) *
                freq_scale;

            const float cos_t = std::cos(theta);
            const float sin_t = std::sin(theta);

            const float x0 = head[2 * i];
            const float x1 = head[2 * i + 1];

            head[2 * i]     = x0 * cos_t - x1 * sin_t;
            head[2 * i + 1] = x0 * sin_t + x1 * cos_t;
        }
    }
}

// ============================================================================
// AVX-512 RoPE
// ============================================================================

#if defined(__AVX512F__)

/// Process 8 pairs (16 floats) at once using AVX-512
__attribute__((target("avx512f,avx512vl")))
static void rope_avx512_impl(float* x, int64_t n_heads, int64_t head_dim, int64_t pos,
                              float freq_base, float freq_scale) {
    if (n_heads <= 0 || head_dim <= 0) return;

    const int64_t half_dim = head_dim / 2;

    for (int64_t h = 0; h < n_heads; ++h) {
        float* head = x + h * head_dim;

        // Process pairs in chunks of 8 (16 floats per chunk)
        int64_t i = 0;
        for (; i + 8 <= half_dim; i += 8) {
            // Precompute cos/sin for 8 pairs (scalar math)
            alignas(64) float cos_interleaved[16];
            alignas(64) float sin_interleaved[16];
            for (int64_t j = 0; j < 8; ++j) {
                const int64_t idx = i + j;
                const float theta = static_cast<float>(pos) *
                    std::pow(freq_base, -2.0f * static_cast<float>(idx) / static_cast<float>(head_dim)) *
                    freq_scale;
                const float c = std::cos(theta);
                const float s = std::sin(theta);
                cos_interleaved[2 * j]     = c;
                cos_interleaved[2 * j + 1] = c;
                sin_interleaved[2 * j]     = s;
                sin_interleaved[2 * j + 1] = s;
            }

            __m512 vcos = _mm512_load_ps(cos_interleaved);
            __m512 vsin = _mm512_load_ps(sin_interleaved);

            // Load 16 floats: [x0, x1, x2, x3, ...] where (x[2j], x[2j+1]) is a pair
            __m512 vx = _mm512_loadu_ps(&head[2 * i]);

            // Extract into even-duplicated and odd-duplicated layouts
            // vx_even[j] = x[2*(j/2)] for even j, x[2*(j/2)] for odd j
            // i.e., vx_even = [x0, x0, x2, x2, x4, x4, ...]
            //       vx_odd  = [x1, x1, x3, x3, x5, x5, ...]
            alignas(64) float x_raw[16];
            _mm512_store_ps(x_raw, vx);

            alignas(64) float x_even[16];
            alignas(64) float x_odd[16];
            for (int64_t j = 0; j < 8; ++j) {
                x_even[2 * j]     = x_raw[2 * j];
                x_even[2 * j + 1] = x_raw[2 * j];
                x_odd[2 * j]      = x_raw[2 * j + 1];
                x_odd[2 * j + 1]  = x_raw[2 * j + 1];
            }

            __m512 vx_even = _mm512_load_ps(x_even);
            __m512 vx_odd  = _mm512_load_ps(x_odd);

            // For each pair (x[2j], x[2j+1]):
            //   result[2j]   = x[2j]*cos[j] - x[2j+1]*sin[j]
            //   result[2j+1] = x[2j]*sin[j] + x[2j+1]*cos[j]
            //
            // In interleaved layout:
            //   vout_even = [x0*c0-x1*s0, x0*c0-x1*s0, x2*c1-x3*s1, ...]  (correct for even positions)
            //   vout_odd  = [x0*s0+x1*c0, x0*s0+x1*c0, x2*s1+x3*c1, ...]  (correct for odd positions)
            __m512 vout_even = _mm512_fnmadd_ps(vx_odd, vsin, _mm512_mul_ps(vx_even, vcos));
            __m512 vout_odd  = _mm512_fmadd_ps(vx_even, vsin, _mm512_mul_ps(vx_odd, vcos));

            // Merge: even positions from vout_even, odd positions from vout_odd
            // mask bit 1 for odd indices: 0b1010101010101010 = 0xAAAA
            const __mmask16 odd_mask = 0xAAAA;
            __m512 vresult = _mm512_mask_blend_ps(odd_mask, vout_even, vout_odd);

            _mm512_storeu_ps(&head[2 * i], vresult);
        }

        // Scalar tail for remaining pairs
        for (; i < half_dim; ++i) {
            const float theta = static_cast<float>(pos) *
                std::pow(freq_base, -2.0f * static_cast<float>(i) / static_cast<float>(head_dim)) *
                freq_scale;

            const float cos_t = std::cos(theta);
            const float sin_t = std::sin(theta);

            const float x0 = head[2 * i];
            const float x1 = head[2 * i + 1];

            head[2 * i]     = x0 * cos_t - x1 * sin_t;
            head[2 * i + 1] = x0 * sin_t + x1 * cos_t;
        }
    }
}

#endif // __AVX512F__

void rope_avx512(float* x, int64_t n_heads, int64_t head_dim, int64_t pos,
                 float freq_base, float freq_scale) {
#if defined(__AVX512F__)
    const auto& info = kaguya::CpuFeatureDetector::get();
    if (info.flags.avx512f && info.flags.os_avx512) {
        rope_avx512_impl(x, n_heads, head_dim, pos, freq_base, freq_scale);
        return;
    }
#endif
    rope_scalar(x, n_heads, head_dim, pos, freq_base, freq_scale);
}

void rope_dispatch(float* x, int64_t n_heads, int64_t head_dim, int64_t pos,
                   float freq_base, float freq_scale) {
    rope_avx512(x, n_heads, head_dim, pos, freq_base, freq_scale);
}

} // namespace kaguya::kernels
