#pragma once
/// @file gemm.h
/// @brief Phase 3 compute kernels: GEMM variants for Kaguya CPU inference engine.
///
/// GEMM operation: C[M,N] = alpha * A[M,K] * B[K,N] + beta * C[M,N]
///
/// Kernel hierarchy (fallback chain):
///   AMX → AVX-512 BF16/VNNI → AVX-512 FP32 → AVX2+FMA → AVX2 → Scalar
///
/// Primary target: AVX-512 BF16/VNNI (confirmed working in KVM)
/// AMX is stub-only due to KVM SIGSEGV on LDTILECFG.

#include <cstdint>
#include <cstddef>

namespace kaguya::kernels {

// ============================================================================
// GEMM parameter structures
// ============================================================================

/// FP32 GEMM operation descriptor: C[M,N] = alpha * A[M,K] * B[K,N] + beta * C[M,N]
/// A is row-major [M,K], B is row-major [K,N], C is row-major [M,N]
struct GemmParams {
    int64_t M = 0;             ///< Rows of A / C
    int64_t N = 0;             ///< Columns of B / C
    int64_t K = 0;             ///< Columns of A / Rows of B (inner dimension)
    const float* A = nullptr;  ///< LHS matrix [M,K] row-major
    const float* B = nullptr;  ///< RHS matrix [K,N] row-major
    float* C = nullptr;        ///< Output matrix [M,N] row-major
    int64_t lda = 0;           ///< Stride of A (default: K)
    int64_t ldb = 0;           ///< Stride of B (default: N)
    int64_t ldc = 0;           ///< Stride of C (default: N)
    float alpha = 1.0f;        ///< Scaling factor for A*B product
    float beta = 0.0f;         ///< Scaling factor for initial C (0 = zero, 1 = accumulate)
};

/// BF16 GEMM: A is BF16 (uint16_t), B is BF16 (uint16_t), C is FP32
/// Used for AVX-512 BF16 dot product (_mm512_dpbf16_ps)
struct GemmParamsBF16 {
    int64_t M = 0;             ///< Rows of A / C
    int64_t N = 0;             ///< Columns of B / C
    int64_t K = 0;             ///< Inner dimension
    const void* A = nullptr;   ///< BF16 data [M,K] (stored as uint16_t)
    const void* B = nullptr;   ///< BF16 data [K,N] (stored as uint16_t)
    float* C = nullptr;        ///< FP32 output [M,N]
    int64_t lda = 0;           ///< Stride of A (default: K)
    int64_t ldb = 0;           ///< Stride of B (default: N)
    int64_t ldc = 0;           ///< Stride of C (default: N)
};

/// INT8 GEMM via VNNI: A is uint8, B is int8, C is int32
/// Used for AVX-512 VNNI dot product (_mm512_dpbusd_epi32)
struct GemmParamsVNNI {
    int64_t M = 0;              ///< Rows of A / C
    int64_t N = 0;              ///< Columns of B / C
    int64_t K = 0;              ///< Inner dimension
    const uint8_t* A = nullptr;  ///< Unsigned int8 [M,K]
    const int8_t* B = nullptr;   ///< Signed int8 [K,N]
    int32_t* C = nullptr;        ///< Int32 output [M,N]
    int64_t lda = 0;            ///< Stride of A (default: K)
    int64_t ldb = 0;            ///< Stride of B (default: N)
    int64_t ldc = 0;            ///< Stride of C (default: N)
};

// ============================================================================
// GEMM parameter validation
// ============================================================================

/// Validate GEMM parameters before computation.
/// Returns true if parameters are valid, false if computation should be skipped
/// (e.g., zero dimensions are valid but require no work).
/// Returns false for: negative dimensions, null pointers when dimensions are non-zero.
bool validate_gemm_params(const GemmParams& params);

// ============================================================================
// FP32 GEMM kernels
// ============================================================================

/// Scalar FP32 GEMM with cache-friendly 64x64 micro-tiling.
/// Reference implementation — always available, used for correctness testing.
void gemm_scalar(const GemmParams& params);

/// AVX2+FMA FP32 GEMM with 4x8 register-blocking micro-kernel.
/// Falls back to AVX2-only (separate mul+add) when FMA is unavailable.
/// Falls back to scalar when AVX2 is unavailable.
void gemm_avx2(const GemmParams& params);

/// AVX-512 FP32 GEMM with 4x16 register-blocking micro-kernel.
/// Falls back to scalar when AVX-512F is unavailable.
void gemm_avx512(const GemmParams& params);

// ============================================================================
// BF16 GEMM kernel (AVX-512 BF16)
// ============================================================================

/// AVX-512 BF16 GEMM using _mm512_dpbf16_ps for 2-way BF16 dot product.
/// Falls back to scalar with BF16→FP32 conversion when AVX-512 BF16 is unavailable.
void gemm_bf16_avx512(const GemmParamsBF16& params);

// ============================================================================
// INT8 GEMM kernel (AVX-512 VNNI)
// ============================================================================

/// AVX-512 VNNI GEMM using _mm512_dpbusd_epi32 for 4-way INT8 dot product.
/// Falls back to scalar when AVX-512 VNNI is unavailable.
void gemm_vnni_avx512(const GemmParamsVNNI& params);

// ============================================================================
// Unified dispatch
// ============================================================================

/// Unified GEMM dispatch: selects the best available FP32 GEMM kernel
/// based on CPU feature detection at runtime.
/// Follows the fallback chain: AMX → AVX-512 → AVX2 → Scalar
void gemm_dispatch(const GemmParams& params);

} // namespace kaguya::kernels
