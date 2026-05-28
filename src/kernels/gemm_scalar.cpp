#include "kaguya/kernels/gemm.h"

#include <cstring>
#include <algorithm>

namespace kaguya::kernels {

/// Scalar FP32 GEMM with cache-friendly 64x64 micro-tiling.
/// Reference implementation for correctness testing.
/// C[M,N] = alpha * A[M,K] * B[K,N] + beta * C[M,N]
void gemm_scalar(const GemmParams& params) {
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

    // Cache-friendly tiled GEMM with 64x64 micro-tiles
    constexpr int64_t TILE = 64;

    for (int64_t ii = 0; ii < M; ii += TILE) {
        const int64_t i_end = std::min(ii + TILE, M);
        for (int64_t jj = 0; jj < N; jj += TILE) {
            const int64_t j_end = std::min(jj + TILE, N);

            // Initialize accumulator tile to zero or existing C values
            // We accumulate alpha * A * B into C
            for (int64_t kk = 0; kk < K; kk += TILE) {
                const int64_t k_end = std::min(kk + TILE, K);

                for (int64_t i = ii; i < i_end; ++i) {
                    for (int64_t k = kk; k < k_end; ++k) {
                        const float a_ik = A[i * lda + k] * alpha;
                        for (int64_t j = jj; j < j_end; ++j) {
                            C[i * ldc + j] += a_ik * B[k * ldb + j];
                        }
                    }
                }
            }
        }
    }
}

} // namespace kaguya::kernels
