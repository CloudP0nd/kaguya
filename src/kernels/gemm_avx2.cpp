#include "kaguya/kernels/gemm.h"
#include "kaguya/cpu_features.h"

#include <cstring>
#include <algorithm>

#if defined(__AVX2__)
#include <immintrin.h>
#endif

namespace kaguya::kernels {

// ============================================================================
// AVX2 + FMA FP32 GEMM
// 4x8 register-blocking micro-kernel with FMA instructions
// C[M,N] = alpha * A[M,K] * B[K,N] + beta * C[M,N]
// ============================================================================
#if defined(__AVX2__) && defined(__FMA__)

__attribute__((target("avx2,fma")))
static void gemm_avx2_fma_kernel(const GemmParams& params) {
    const int64_t M = params.M;
    const int64_t N = params.N;
    const int64_t K = params.K;
    const float* A = params.A;
    const float* B = params.B;
    float* C = params.C;
    const int64_t lda = params.lda ? params.lda : K;
    const int64_t ldb = params.ldb ? params.ldb : N;
    const int64_t ldc = params.ldc ? params.ldc : N;
    const float alpha = params.alpha;
    const float beta = params.beta;

    // Handle beta scaling: C = beta * C
    if (beta == 0.0f) {
        std::memset(C, 0, sizeof(float) * static_cast<size_t>(M) * static_cast<size_t>(ldc));
    } else if (beta != 1.0f) {
        for (int64_t i = 0; i < M; ++i) {
            for (int64_t j = 0; j < N; ++j) {
                C[i * ldc + j] *= beta;
            }
        }
    }

    // Early exit for zero alpha or empty dimensions
    if (alpha == 0.0f || M == 0 || N == 0 || K == 0) {
        return;
    }

    const __m256 valpha = _mm256_set1_ps(alpha);

    constexpr int64_t MR = 4;   // 4 rows of A in registers
    constexpr int64_t NR = 8;   // 8 floats per __m256

    for (int64_t ii = 0; ii < M; ii += MR) {
        const int64_t mr = std::min(MR, M - ii);

        for (int64_t jj = 0; jj < N; jj += NR) {
            const int64_t nr = std::min(NR, N - jj);

            // Accumulators for 4 rows x 8 columns
            __m256 c0 = _mm256_setzero_ps();
            __m256 c1 = _mm256_setzero_ps();
            __m256 c2 = _mm256_setzero_ps();
            __m256 c3 = _mm256_setzero_ps();

            for (int64_t k = 0; k < K; ++k) {
                // Broadcast A values (with boundary guard for partial rows)
                const float a0 = (mr > 0) ? A[(ii + 0) * lda + k] : 0.0f;
                const float a1 = (mr > 1) ? A[(ii + 1) * lda + k] : 0.0f;
                const float a2 = (mr > 2) ? A[(ii + 2) * lda + k] : 0.0f;
                const float a3 = (mr > 3) ? A[(ii + 3) * lda + k] : 0.0f;

                // Load B row (with mask for partial columns)
                __m256 bv;
                if (nr == NR) {
                    bv = _mm256_loadu_ps(&B[k * ldb + jj]);
                } else {
                    __m256i mask = _mm256_set_epi32(
                        nr > 7 ? -1 : 0, nr > 6 ? -1 : 0, nr > 5 ? -1 : 0, nr > 4 ? -1 : 0,
                        nr > 3 ? -1 : 0, nr > 2 ? -1 : 0, nr > 1 ? -1 : 0, nr > 0 ? -1 : 0);
                    bv = _mm256_maskload_ps(&B[k * ldb + jj], mask);
                }

                // FMA: c[i] += a[i] * b[j]
                c0 = _mm256_fmadd_ps(_mm256_set1_ps(a0), bv, c0);
                c1 = _mm256_fmadd_ps(_mm256_set1_ps(a1), bv, c1);
                c2 = _mm256_fmadd_ps(_mm256_set1_ps(a2), bv, c2);
                c3 = _mm256_fmadd_ps(_mm256_set1_ps(a3), bv, c3);
            }

            // Scale by alpha
            c0 = _mm256_mul_ps(valpha, c0);
            c1 = _mm256_mul_ps(valpha, c1);
            c2 = _mm256_mul_ps(valpha, c2);
            c3 = _mm256_mul_ps(valpha, c3);

            // Store: C += alpha * A * B (beta already handled)
            auto store_row = [&](int64_t row, const __m256& cv) {
                if (nr == NR) {
                    __m256 old_c = _mm256_loadu_ps(&C[row * ldc + jj]);
                    _mm256_storeu_ps(&C[row * ldc + jj], _mm256_add_ps(old_c, cv));
                } else {
                    __m256i mask = _mm256_set_epi32(
                        nr > 7 ? -1 : 0, nr > 6 ? -1 : 0, nr > 5 ? -1 : 0, nr > 4 ? -1 : 0,
                        nr > 3 ? -1 : 0, nr > 2 ? -1 : 0, nr > 1 ? -1 : 0, nr > 0 ? -1 : 0);
                    __m256 old_c = _mm256_maskload_ps(&C[row * ldc + jj], mask);
                    __m256 result = _mm256_add_ps(old_c, cv);
                    _mm256_maskstore_ps(&C[row * ldc + jj], mask, result);
                }
            };

            if (mr > 0) store_row(ii + 0, c0);
            if (mr > 1) store_row(ii + 1, c1);
            if (mr > 2) store_row(ii + 2, c2);
            if (mr > 3) store_row(ii + 3, c3);
        }
    }
}

#endif // __AVX2__ && __FMA__

// ============================================================================
// AVX2-only FP32 GEMM (no FMA)
// 4x8 register-blocking micro-kernel using separate mul+add
// Fallback for CPUs/VMs where AVX2 is available but FMA CPUID bit is not set
// ============================================================================
#if defined(__AVX2__)

__attribute__((target("avx2")))
static void gemm_avx2_nofma_kernel(const GemmParams& params) {
    const int64_t M = params.M;
    const int64_t N = params.N;
    const int64_t K = params.K;
    const float* A = params.A;
    const float* B = params.B;
    float* C = params.C;
    const int64_t lda = params.lda ? params.lda : K;
    const int64_t ldb = params.ldb ? params.ldb : N;
    const int64_t ldc = params.ldc ? params.ldc : N;
    const float alpha = params.alpha;
    const float beta = params.beta;

    // Handle beta scaling
    if (beta == 0.0f) {
        std::memset(C, 0, sizeof(float) * static_cast<size_t>(M) * static_cast<size_t>(ldc));
    } else if (beta != 1.0f) {
        for (int64_t i = 0; i < M; ++i) {
            for (int64_t j = 0; j < N; ++j) {
                C[i * ldc + j] *= beta;
            }
        }
    }

    if (alpha == 0.0f || M == 0 || N == 0 || K == 0) {
        return;
    }

    const __m256 valpha = _mm256_set1_ps(alpha);

    constexpr int64_t MR = 4;
    constexpr int64_t NR = 8;

    for (int64_t ii = 0; ii < M; ii += MR) {
        const int64_t mr = std::min(MR, M - ii);

        for (int64_t jj = 0; jj < N; jj += NR) {
            const int64_t nr = std::min(NR, N - jj);

            __m256 c0 = _mm256_setzero_ps();
            __m256 c1 = _mm256_setzero_ps();
            __m256 c2 = _mm256_setzero_ps();
            __m256 c3 = _mm256_setzero_ps();

            for (int64_t k = 0; k < K; ++k) {
                const float a0 = (mr > 0) ? A[(ii + 0) * lda + k] : 0.0f;
                const float a1 = (mr > 1) ? A[(ii + 1) * lda + k] : 0.0f;
                const float a2 = (mr > 2) ? A[(ii + 2) * lda + k] : 0.0f;
                const float a3 = (mr > 3) ? A[(ii + 3) * lda + k] : 0.0f;

                __m256 bv;
                if (nr == NR) {
                    bv = _mm256_loadu_ps(&B[k * ldb + jj]);
                } else {
                    __m256i mask = _mm256_set_epi32(
                        nr > 7 ? -1 : 0, nr > 6 ? -1 : 0, nr > 5 ? -1 : 0, nr > 4 ? -1 : 0,
                        nr > 3 ? -1 : 0, nr > 2 ? -1 : 0, nr > 1 ? -1 : 0, nr > 0 ? -1 : 0);
                    bv = _mm256_maskload_ps(&B[k * ldb + jj], mask);
                }

                // Separate multiply and add (no FMA)
                c0 = _mm256_add_ps(c0, _mm256_mul_ps(_mm256_set1_ps(a0), bv));
                c1 = _mm256_add_ps(c1, _mm256_mul_ps(_mm256_set1_ps(a1), bv));
                c2 = _mm256_add_ps(c2, _mm256_mul_ps(_mm256_set1_ps(a2), bv));
                c3 = _mm256_add_ps(c3, _mm256_mul_ps(_mm256_set1_ps(a3), bv));
            }

            // Scale by alpha
            c0 = _mm256_mul_ps(valpha, c0);
            c1 = _mm256_mul_ps(valpha, c1);
            c2 = _mm256_mul_ps(valpha, c2);
            c3 = _mm256_mul_ps(valpha, c3);

            // Store
            auto store_row = [&](int64_t row, const __m256& cv) {
                if (nr == NR) {
                    __m256 old_c = _mm256_loadu_ps(&C[row * ldc + jj]);
                    _mm256_storeu_ps(&C[row * ldc + jj], _mm256_add_ps(old_c, cv));
                } else {
                    __m256i mask = _mm256_set_epi32(
                        nr > 7 ? -1 : 0, nr > 6 ? -1 : 0, nr > 5 ? -1 : 0, nr > 4 ? -1 : 0,
                        nr > 3 ? -1 : 0, nr > 2 ? -1 : 0, nr > 1 ? -1 : 0, nr > 0 ? -1 : 0);
                    __m256 old_c = _mm256_maskload_ps(&C[row * ldc + jj], mask);
                    __m256 result = _mm256_add_ps(old_c, cv);
                    _mm256_maskstore_ps(&C[row * ldc + jj], mask, result);
                }
            };

            if (mr > 0) store_row(ii + 0, c0);
            if (mr > 1) store_row(ii + 1, c1);
            if (mr > 2) store_row(ii + 2, c2);
            if (mr > 3) store_row(ii + 3, c3);
        }
    }
}

#endif // __AVX2__

void gemm_avx2(const GemmParams& params) {
    if (!validate_gemm_params(params)) return;

    const auto& info = kaguya::CpuFeatureDetector::get();

#if defined(__AVX2__) && defined(__FMA__)
    // Fast path: AVX2 + FMA
    if (info.flags.avx2 && info.flags.fma && info.flags.os_avx) {
        gemm_avx2_fma_kernel(params);
        return;
    }
#endif

#if defined(__AVX2__)
    // Fallback: AVX2 only (no FMA) — handles KVM environments where
    // CPUID doesn't report FMA but hardware supports AVX2
    if (info.flags.avx2 && info.flags.os_avx) {
        gemm_avx2_nofma_kernel(params);
        return;
    }
#endif

    // Final fallback: scalar
    gemm_scalar(params);
}

} // namespace kaguya::kernels
