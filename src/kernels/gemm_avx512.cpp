#include "kaguya/kernels/gemm.h"
#include "kaguya/cpu_features.h"

#include <cstring>
#include <algorithm>
#include <vector>

#if defined(__AVX512F__)
#include <immintrin.h>
#endif

namespace kaguya::kernels {

// ============================================================================
// Helper: BF16 → FP32 conversion (always available for fallback path)
// ============================================================================

/// Convert BF16 (stored as uint16_t) to FP32
static inline float bf16_to_f32(uint16_t bf16_val) {
    uint32_t bits = static_cast<uint32_t>(bf16_val) << 16;
    float result;
    std::memcpy(&result, &bits, sizeof(float));
    return result;
}

// ============================================================================
// Helper: cast __m512i to __m512bh (GCC 14 doesn't have _mm512_castsi512_bh)
// ============================================================================
#if defined(__AVX512BF16__)
static inline __m512bh si512_to_bh(__m512i v) {
    __m512bh result;
    std::memcpy(&result, &v, sizeof(__m512bh));
    return result;
}

/// Interleave two BF16 rows into dpbf16_ps format:
/// BF16[2*j] = row0[j], BF16[2*j+1] = row1[j] for j=0..15
/// This matches the pair layout expected by _mm512_dpbf16_ps where
/// each output lane i accumulates: a.bf16[2*i]*b.bf16[2*i] + a.bf16[2*i+1]*b.bf16[2*i+1]
__attribute__((target("avx512f,avx512bf16,avx512vl")))
static inline __m512bh interleave_bf16_rows(__m256i row0, __m256i row1) {
    // _mm256_unpacklo/hi_epi16 operate per 128-bit lane:
    //   unpacklo: lane0=[r0[0..3]||r1[0..3]], lane1=[r0[8..11]||r1[8..11]]
    //   unpackhi: lane0=[r0[4..7]||r1[4..7]], lane1=[r0[12..15]||r1[12..15]]
    __m256i lo = _mm256_unpacklo_epi16(row0, row1);
    __m256i hi = _mm256_unpackhi_epi16(row0, row1);

    // Extract 128-bit lanes and reassemble in column order:
    //   [0-3 interleaved, 4-7 interleaved, 8-11 interleaved, 12-15 interleaved]
    __m512i result = _mm512_castsi128_si512(_mm256_extracti128_si256(lo, 0));
    result = _mm512_inserti32x4(result, _mm256_extracti128_si256(hi, 0), 1);
    result = _mm512_inserti32x4(result, _mm256_extracti128_si256(lo, 1), 2);
    result = _mm512_inserti32x4(result, _mm256_extracti128_si256(hi, 1), 3);

    return si512_to_bh(result);
}
#endif

// ============================================================================
// AVX-512 FP32 GEMM
// 4x16 register-blocking micro-kernel using AVX-512F + FMA
// C[M,N] = alpha * A[M,K] * B[K,N] + beta * C[M,N]
// ============================================================================
#if defined(__AVX512F__)

__attribute__((target("avx512f,avx512vl")))
static void gemm_avx512_fp32_kernel(const GemmParams& params) {
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

    if (alpha == 0.0f || M == 0 || N == 0 || K == 0) return;

    const __m512 valpha = _mm512_set1_ps(alpha);

    constexpr int64_t MR = 4;   // 4 rows of A
    constexpr int64_t NR = 16;  // 16 floats per __m512

    for (int64_t ii = 0; ii < M; ii += MR) {
        const int64_t mr = std::min(MR, M - ii);

        for (int64_t jj = 0; jj < N; jj += NR) {
            const int64_t nr = std::min(NR, N - jj);

            __m512 c0 = _mm512_setzero_ps();
            __m512 c1 = _mm512_setzero_ps();
            __m512 c2 = _mm512_setzero_ps();
            __m512 c3 = _mm512_setzero_ps();

            for (int64_t k = 0; k < K; ++k) {
                const float a0 = (mr > 0) ? A[(ii + 0) * lda + k] : 0.0f;
                const float a1 = (mr > 1) ? A[(ii + 1) * lda + k] : 0.0f;
                const float a2 = (mr > 2) ? A[(ii + 2) * lda + k] : 0.0f;
                const float a3 = (mr > 3) ? A[(ii + 3) * lda + k] : 0.0f;

                // Software prefetch: hint next iteration's data into L1
#if defined(__AVX512F__)
                if (k + 1 < K) {
                    _mm_prefetch((const char*)&B[(k + 1) * ldb + jj], _MM_HINT_T0);
                    if (mr > 0) _mm_prefetch((const char*)&A[(ii + 0) * lda + k + 1], _MM_HINT_T0);
                    if (mr > 1) _mm_prefetch((const char*)&A[(ii + 1) * lda + k + 1], _MM_HINT_T0);
                    if (mr > 2) _mm_prefetch((const char*)&A[(ii + 2) * lda + k + 1], _MM_HINT_T0);
                    if (mr > 3) _mm_prefetch((const char*)&A[(ii + 3) * lda + k + 1], _MM_HINT_T0);
                }
#endif

                __m512 bv;
                if (nr == NR) {
                    bv = _mm512_loadu_ps(&B[k * ldb + jj]);
                } else {
                    const __mmask16 mask = static_cast<__mmask16>((1U << nr) - 1U);
                    bv = _mm512_maskz_loadu_ps(mask, &B[k * ldb + jj]);
                }

                c0 = _mm512_fmadd_ps(_mm512_set1_ps(a0), bv, c0);
                c1 = _mm512_fmadd_ps(_mm512_set1_ps(a1), bv, c1);
                c2 = _mm512_fmadd_ps(_mm512_set1_ps(a2), bv, c2);
                c3 = _mm512_fmadd_ps(_mm512_set1_ps(a3), bv, c3);
            }

            c0 = _mm512_mul_ps(valpha, c0);
            c1 = _mm512_mul_ps(valpha, c1);
            c2 = _mm512_mul_ps(valpha, c2);
            c3 = _mm512_mul_ps(valpha, c3);

            auto store_row = [&](int64_t row, const __m512& cv) {
                if (nr == NR) {
                    __m512 old_c = _mm512_loadu_ps(&C[row * ldc + jj]);
                    _mm512_storeu_ps(&C[row * ldc + jj], _mm512_add_ps(old_c, cv));
                } else {
                    const __mmask16 mask = static_cast<__mmask16>((1U << nr) - 1U);
                    __m512 old_c = _mm512_maskz_loadu_ps(mask, &C[row * ldc + jj]);
                    _mm512_mask_storeu_ps(&C[row * ldc + jj], mask, _mm512_add_ps(old_c, cv));
                }
            };

            if (mr > 0) store_row(ii + 0, c0);
            if (mr > 1) store_row(ii + 1, c1);
            if (mr > 2) store_row(ii + 2, c2);
            if (mr > 3) store_row(ii + 3, c3);
        }
    }
}

#endif // __AVX512F__

void gemm_avx512(const GemmParams& params) {
    if (!validate_gemm_params(params)) return;

#if defined(__AVX512F__)
    const auto& info = kaguya::CpuFeatureDetector::get();
    if (info.flags.avx512f && info.flags.os_avx512) {
        gemm_avx512_fp32_kernel(params);
        return;
    }
#endif
    gemm_scalar(params);
}

// ============================================================================
// AVX-512 BF16 GEMM — PRIMARY KERNEL
// Uses _mm512_dpbf16_ps for BF16 dot product
// A is BF16, B is BF16, C is FP32
//
// _mm512_dpbf16_ps(acc, a, b) computes for each lane i (0..15):
//   acc[i] += (float)a.bf16[2*i] * (float)b.bf16[2*i]
//           + (float)a.bf16[2*i+1] * (float)b.bf16[2*i+1]
//
// We process K in steps of 2. For each pair (k, k+1):
// - b_bf16: BF16[2*j] = B[k,jj+j], BF16[2*j+1] = B[k+1,jj+j] (interleaved rows)
// - a_bf16: BF16[2*j] = A[row,k], BF16[2*j+1] = A[row,k+1] (same pair for all j)
// Then acc[j] += A[row,k]*B[k,jj+j] + A[row,k+1]*B[k+1,jj+j]
// ============================================================================
#if defined(__AVX512BF16__)

__attribute__((target("avx512f,avx512bf16,avx512vl")))
static void gemm_bf16_avx512_impl(const GemmParamsBF16& params) {
    const int64_t M = params.M;
    const int64_t N = params.N;
    const int64_t K = params.K;
    const auto* A = static_cast<const uint16_t*>(params.A);
    const auto* B = static_cast<const uint16_t*>(params.B);
    float* C = params.C;
    const int64_t lda = params.lda ? params.lda : K;
    const int64_t ldb = params.ldb ? params.ldb : N;
    const int64_t ldc = params.ldc ? params.ldc : N;

    if (M == 0 || N == 0 || K == 0) return;

    std::memset(C, 0, sizeof(float) * static_cast<size_t>(M) * static_cast<size_t>(ldc));

    constexpr int64_t MR = 4;
    constexpr int64_t NR = 16;

    for (int64_t ii = 0; ii < M; ii += MR) {
        const int64_t mr = std::min(MR, M - ii);

        for (int64_t jj = 0; jj < N; jj += NR) {
            const int64_t nr = std::min(NR, N - jj);

            __m512 c0 = _mm512_setzero_ps();
            __m512 c1 = _mm512_setzero_ps();
            __m512 c2 = _mm512_setzero_ps();
            __m512 c3 = _mm512_setzero_ps();

            // Main loop: process K in steps of 2 using dpbf16_ps
            int64_t k = 0;
            for (; k + 1 < K; k += 2) {
                // Load B rows and interleave for dpbf16_ps pair layout
                __m256i b_row0, b_row1;
                if (nr == NR) {
                    b_row0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&B[k * ldb + jj]));
                    b_row1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&B[(k + 1) * ldb + jj]));
                } else {
                    const __mmask32 mask = (1U << nr) - 1U;
                    b_row0 = _mm256_maskz_loadu_epi16(mask, &B[k * ldb + jj]);
                    b_row1 = _mm256_maskz_loadu_epi16(mask, &B[(k + 1) * ldb + jj]);
                }
                __m512bh b_bf16 = interleave_bf16_rows(b_row0, b_row1);

                // For each row of A, pack the two BF16 values as a 32-bit pair
                // and broadcast across all 16 lanes
                if (mr > 0) {
                    uint16_t a0_k0 = A[(ii + 0) * lda + k];
                    uint16_t a0_k1 = A[(ii + 0) * lda + k + 1];
                    uint32_t a0_pair = (static_cast<uint32_t>(a0_k1) << 16) | static_cast<uint32_t>(a0_k0);
                    __m512bh a0_bf16 = si512_to_bh(_mm512_set1_epi32(static_cast<int32_t>(a0_pair)));
                    c0 = _mm512_dpbf16_ps(c0, a0_bf16, b_bf16);
                }
                if (mr > 1) {
                    uint16_t a1_k0 = A[(ii + 1) * lda + k];
                    uint16_t a1_k1 = A[(ii + 1) * lda + k + 1];
                    uint32_t a1_pair = (static_cast<uint32_t>(a1_k1) << 16) | static_cast<uint32_t>(a1_k0);
                    __m512bh a1_bf16 = si512_to_bh(_mm512_set1_epi32(static_cast<int32_t>(a1_pair)));
                    c1 = _mm512_dpbf16_ps(c1, a1_bf16, b_bf16);
                }
                if (mr > 2) {
                    uint16_t a2_k0 = A[(ii + 2) * lda + k];
                    uint16_t a2_k1 = A[(ii + 2) * lda + k + 1];
                    uint32_t a2_pair = (static_cast<uint32_t>(a2_k1) << 16) | static_cast<uint32_t>(a2_k0);
                    __m512bh a2_bf16 = si512_to_bh(_mm512_set1_epi32(static_cast<int32_t>(a2_pair)));
                    c2 = _mm512_dpbf16_ps(c2, a2_bf16, b_bf16);
                }
                if (mr > 3) {
                    uint16_t a3_k0 = A[(ii + 3) * lda + k];
                    uint16_t a3_k1 = A[(ii + 3) * lda + k + 1];
                    uint32_t a3_pair = (static_cast<uint32_t>(a3_k1) << 16) | static_cast<uint32_t>(a3_k0);
                    __m512bh a3_bf16 = si512_to_bh(_mm512_set1_epi32(static_cast<int32_t>(a3_pair)));
                    c3 = _mm512_dpbf16_ps(c3, a3_bf16, b_bf16);
                }
            }

            // Handle odd K remainder (second BF16 in pair is zero)
            if (k < K) {
                __m256i b_row0, b_row1;
                b_row1 = _mm256_setzero_si256();
                if (nr == NR) {
                    b_row0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&B[k * ldb + jj]));
                } else {
                    const __mmask32 mask = (1U << nr) - 1U;
                    b_row0 = _mm256_maskz_loadu_epi16(mask, &B[k * ldb + jj]);
                }
                __m512bh b_bf16 = interleave_bf16_rows(b_row0, b_row1);

                if (mr > 0) {
                    uint16_t a0_k0 = A[(ii + 0) * lda + k];
                    uint32_t a0_pair = static_cast<uint32_t>(a0_k0);  // high 16 bits = 0
                    __m512bh a0_bf16 = si512_to_bh(_mm512_set1_epi32(static_cast<int32_t>(a0_pair)));
                    c0 = _mm512_dpbf16_ps(c0, a0_bf16, b_bf16);
                }
                if (mr > 1) {
                    uint16_t a1_k0 = A[(ii + 1) * lda + k];
                    uint32_t a1_pair = static_cast<uint32_t>(a1_k0);
                    __m512bh a1_bf16 = si512_to_bh(_mm512_set1_epi32(static_cast<int32_t>(a1_pair)));
                    c1 = _mm512_dpbf16_ps(c1, a1_bf16, b_bf16);
                }
                if (mr > 2) {
                    uint16_t a2_k0 = A[(ii + 2) * lda + k];
                    uint32_t a2_pair = static_cast<uint32_t>(a2_k0);
                    __m512bh a2_bf16 = si512_to_bh(_mm512_set1_epi32(static_cast<int32_t>(a2_pair)));
                    c2 = _mm512_dpbf16_ps(c2, a2_bf16, b_bf16);
                }
                if (mr > 3) {
                    uint16_t a3_k0 = A[(ii + 3) * lda + k];
                    uint32_t a3_pair = static_cast<uint32_t>(a3_k0);
                    __m512bh a3_bf16 = si512_to_bh(_mm512_set1_epi32(static_cast<int32_t>(a3_pair)));
                    c3 = _mm512_dpbf16_ps(c3, a3_bf16, b_bf16);
                }
            }

            // Store accumulators
            auto store_row = [&](int64_t row, const __m512& cv) {
                if (nr == NR) {
                    __m512 old_c = _mm512_loadu_ps(&C[row * ldc + jj]);
                    _mm512_storeu_ps(&C[row * ldc + jj], _mm512_add_ps(old_c, cv));
                } else {
                    const __mmask16 mask = static_cast<__mmask16>((1U << nr) - 1U);
                    __m512 old_c = _mm512_maskz_loadu_ps(mask, &C[row * ldc + jj]);
                    _mm512_mask_storeu_ps(&C[row * ldc + jj], mask, _mm512_add_ps(old_c, cv));
                }
            };

            if (mr > 0) store_row(ii + 0, c0);
            if (mr > 1) store_row(ii + 1, c1);
            if (mr > 2) store_row(ii + 2, c2);
            if (mr > 3) store_row(ii + 3, c3);
        }
    }
}

#endif // __AVX512BF16__

void gemm_bf16_avx512(const GemmParamsBF16& params) {
#if defined(__AVX512BF16__)
    const auto& info = kaguya::CpuFeatureDetector::get();
    if (info.flags.avx512bf16 && info.flags.avx512f && info.flags.os_avx512) {
        gemm_bf16_avx512_impl(params);
        return;
    }
#endif
    // Fallback: convert BF16 to FP32 and use scalar GEMM
    const int64_t M = params.M;
    const int64_t N = params.N;
    const int64_t K = params.K;
    const int64_t lda_bf16 = params.lda ? params.lda : K;
    const int64_t ldb_bf16 = params.ldb ? params.ldb : N;
    const int64_t ldc = params.ldc ? params.ldc : N;

    std::vector<float> A_fp32(static_cast<size_t>(M * K));
    const auto* A_bf16 = static_cast<const uint16_t*>(params.A);
    for (int64_t i = 0; i < M; ++i) {
        for (int64_t j = 0; j < K; ++j) {
            A_fp32[static_cast<size_t>(i * K + j)] = bf16_to_f32(A_bf16[i * lda_bf16 + j]);
        }
    }

    std::vector<float> B_fp32(static_cast<size_t>(K * N));
    const auto* B_bf16 = static_cast<const uint16_t*>(params.B);
    for (int64_t i = 0; i < K; ++i) {
        for (int64_t j = 0; j < N; ++j) {
            B_fp32[static_cast<size_t>(i * N + j)] = bf16_to_f32(B_bf16[i * ldb_bf16 + j]);
        }
    }

    GemmParams fp32_params{};
    fp32_params.M = M;
    fp32_params.N = N;
    fp32_params.K = K;
    fp32_params.A = A_fp32.data();
    fp32_params.B = B_fp32.data();
    fp32_params.C = params.C;
    fp32_params.lda = K;
    fp32_params.ldb = N;
    fp32_params.ldc = ldc;
    fp32_params.alpha = 1.0f;
    fp32_params.beta = 0.0f;
    gemm_scalar(fp32_params);
}

// ============================================================================
// AVX-512 VNNI GEMM
// Uses _mm512_dpbusd_epi32 for INT8 dot product
// A is uint8, B is int8, C is int32
//
// _mm512_dpbusd_epi32(acc, a, b) computes for each 32-bit lane i (0..15):
//   acc[i] += sum_{j=0..3}( a.ub[4*i+j] * b.b[4*i+j] )
//
// a is __m512i: 16 groups of 4 uint8 (same 4 bytes repeated per lane)
// b is __m512i: 16 groups of 4 int8 (VNNI layout: 4 K-steps per column)
// ============================================================================
#if defined(__AVX512VNNI__)

/// Transpose a 4x16 byte matrix (4 rows of 16 bytes) into VNNI layout
/// (16 columns of 4 bytes, each column packed into a 32-bit lane).
__attribute__((target("avx512f,avx512vnni,avx512vl,avx512bw")))
static inline __m512i transpose_4x16_bytes(__m128i r0, __m128i r1, __m128i r2, __m128i r3) {
    // Stage 1: interleave pairs at byte level
    __m128i t01_lo = _mm_unpacklo_epi8(r0, r1);
    __m128i t01_hi = _mm_unpackhi_epi8(r0, r1);
    __m128i t23_lo = _mm_unpacklo_epi8(r2, r3);
    __m128i t23_hi = _mm_unpackhi_epi8(r2, r3);

    // Stage 2: interleave at 16-bit level to get 4-byte groups
    __m128i lane_03 = _mm_unpacklo_epi16(t01_lo, t23_lo);
    __m128i lane_47 = _mm_unpackhi_epi16(t01_lo, t23_lo);
    __m128i lane_8B = _mm_unpacklo_epi16(t01_hi, t23_hi);
    __m128i lane_CF = _mm_unpackhi_epi16(t01_hi, t23_hi);

    // Combine into __m512i
    __m512i result = _mm512_castsi128_si512(lane_03);
    result = _mm512_inserti32x4(result, lane_47, 1);
    result = _mm512_inserti32x4(result, lane_8B, 2);
    result = _mm512_inserti32x4(result, lane_CF, 3);
    return result;
}

__attribute__((target("avx512f,avx512vnni,avx512vl,avx512bw")))
static void gemm_vnni_avx512_impl(const GemmParamsVNNI& params) {
    const int64_t M = params.M;
    const int64_t N = params.N;
    const int64_t K = params.K;
    const uint8_t* A = params.A;
    const int8_t* B = params.B;
    int32_t* C = params.C;
    const int64_t lda = params.lda ? params.lda : K;
    const int64_t ldb = params.ldb ? params.ldb : N;
    const int64_t ldc = params.ldc ? params.ldc : N;

    if (M == 0 || N == 0 || K == 0) return;

    std::memset(C, 0, sizeof(int32_t) * static_cast<size_t>(M) * static_cast<size_t>(ldc));

    constexpr int64_t MR = 4;
    constexpr int64_t NR = 16;
    constexpr int64_t KR = 4;  // VNNI processes 4 K-steps at once

    for (int64_t ii = 0; ii < M; ii += MR) {
        const int64_t mr = std::min(MR, M - ii);

        for (int64_t jj = 0; jj < N; jj += NR) {
            const int64_t nr = std::min(NR, N - jj);

            __m512i c0 = _mm512_setzero_si512();
            __m512i c1 = _mm512_setzero_si512();
            __m512i c2 = _mm512_setzero_si512();
            __m512i c3 = _mm512_setzero_si512();

            bool used_vector = false;
            int64_t k = 0;

            // Vector path: process K in steps of 4 using dpbusd_epi32
            if (nr == NR) {
                used_vector = true;
                for (; k + KR <= K; k += KR) {
                    // Load 4 rows of B, 16 bytes each
                    __m128i b_r0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(&B[k * ldb + jj]));
                    __m128i b_r1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(&B[(k + 1) * ldb + jj]));
                    __m128i b_r2 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(&B[(k + 2) * ldb + jj]));
                    __m128i b_r3 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(&B[(k + 3) * ldb + jj]));

                    // Transpose into VNNI layout
                    __m512i b_vnni = transpose_4x16_bytes(b_r0, b_r1, b_r2, b_r3);

                    // For each row of A, broadcast 4 uint8 values across all 16 lanes
                    if (mr > 0) {
                        int32_t a0_bits;
                        std::memcpy(&a0_bits, &A[(ii + 0) * lda + k], 4);
                        c0 = _mm512_dpbusd_epi32(c0, _mm512_set1_epi32(a0_bits), b_vnni);
                    }
                    if (mr > 1) {
                        int32_t a1_bits;
                        std::memcpy(&a1_bits, &A[(ii + 1) * lda + k], 4);
                        c1 = _mm512_dpbusd_epi32(c1, _mm512_set1_epi32(a1_bits), b_vnni);
                    }
                    if (mr > 2) {
                        int32_t a2_bits;
                        std::memcpy(&a2_bits, &A[(ii + 2) * lda + k], 4);
                        c2 = _mm512_dpbusd_epi32(c2, _mm512_set1_epi32(a2_bits), b_vnni);
                    }
                    if (mr > 3) {
                        int32_t a3_bits;
                        std::memcpy(&a3_bits, &A[(ii + 3) * lda + k], 4);
                        c3 = _mm512_dpbusd_epi32(c3, _mm512_set1_epi32(a3_bits), b_vnni);
                    }
                }
            }

            // Handle remaining K elements with scalar
            for (; k < K; ++k) {
                for (int64_t i = ii; i < ii + mr; ++i) {
                    int32_t a_val = static_cast<int32_t>(A[i * lda + k]);
                    for (int64_t j = jj; j < jj + nr; ++j) {
                        C[i * ldc + j] += a_val * static_cast<int32_t>(B[k * ldb + j]);
                    }
                }
            }

            // Store vector accumulators
            if (used_vector) {
                auto store_row = [&](int64_t row, const __m512i& cv) {
                    __m512i old_c = _mm512_loadu_si512(&C[row * ldc + jj]);
                    _mm512_storeu_si512(&C[row * ldc + jj], _mm512_add_epi32(old_c, cv));
                };
                if (mr > 0) store_row(ii + 0, c0);
                if (mr > 1) store_row(ii + 1, c1);
                if (mr > 2) store_row(ii + 2, c2);
                if (mr > 3) store_row(ii + 3, c3);
            }
        }
    }
}

#endif // __AVX512VNNI__

void gemm_vnni_avx512(const GemmParamsVNNI& params) {
#if defined(__AVX512VNNI__)
    const auto& info = kaguya::CpuFeatureDetector::get();
    if (info.flags.avx512vnni && info.flags.avx512f && info.flags.os_avx512) {
        gemm_vnni_avx512_impl(params);
        return;
    }
#endif
    // Scalar fallback
    const int64_t M = params.M;
    const int64_t N = params.N;
    const int64_t K = params.K;
    const int64_t lda = params.lda ? params.lda : K;
    const int64_t ldb = params.ldb ? params.ldb : N;
    const int64_t ldc = params.ldc ? params.ldc : N;

    std::memset(params.C, 0, sizeof(int32_t) * static_cast<size_t>(M) * static_cast<size_t>(ldc));

    for (int64_t i = 0; i < M; ++i) {
        for (int64_t k = 0; k < K; ++k) {
            int32_t a_val = static_cast<int32_t>(params.A[i * lda + k]);
            for (int64_t j = 0; j < N; ++j) {
                params.C[i * ldc + j] += a_val * static_cast<int32_t>(params.B[k * ldb + j]);
            }
        }
    }
}

} // namespace kaguya::kernels
