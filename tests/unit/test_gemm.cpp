#include <gtest/gtest.h>
#include <vector>
#include <cmath>
#include <cstring>
#include <random>
#include <cstdint>

#include "kaguya/kernels/gemm.h"
#include "kaguya/kernels/dispatcher.h"
#include "kaguya/cpu_features.h"

using namespace kaguya;
using namespace kaguya::kernels;

// ============================================================================
// Helper utilities
// ============================================================================

/// Convert FP32 to BF16 (truncate lower 16 bits)
static inline uint16_t f32_to_bf16(float val) {
    uint32_t bits;
    std::memcpy(&bits, &val, sizeof(float));
    return static_cast<uint16_t>(bits >> 16);
}

/// Convert BF16 to FP32
static inline float bf16_to_f32(uint16_t val) {
    uint32_t bits = static_cast<uint32_t>(val) << 16;
    float result;
    std::memcpy(&result, &bits, sizeof(float));
    return result;
}

/// Fill a vector with random floats in [-1, 1]
static void fill_random(std::vector<float>& v, unsigned seed = 42) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (auto& x : v) x = dist(rng);
}

/// Fill a vector with random uint8 values
static void fill_random_u8(std::vector<uint8_t>& v, unsigned seed = 42) {
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> dist(0, 255);
    for (auto& x : v) x = static_cast<uint8_t>(dist(rng));
}

/// Fill a vector with random int8 values
static void fill_random_i8(std::vector<int8_t>& v, unsigned seed = 42) {
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> dist(-128, 127);
    for (auto& x : v) x = static_cast<int8_t>(dist(rng));
}

/// Fill a vector with sequential values
static void fill_sequential(std::vector<float>& v) {
    for (size_t i = 0; i < v.size(); ++i) {
        v[i] = static_cast<float>(i + 1);
    }
}

/// Reference GEMM: C = alpha * A * B + beta * C (simple triple loop)
static void ref_gemm(int64_t M, int64_t N, int64_t K,
                     const float* A, int64_t lda,
                     const float* B, int64_t ldb,
                     float* C, int64_t ldc,
                     float alpha, float beta) {
    for (int64_t i = 0; i < M; ++i) {
        for (int64_t j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int64_t k = 0; k < K; ++k) {
                sum += A[i * lda + k] * B[k * ldb + j];
            }
            C[i * ldc + j] = alpha * sum + beta * C[i * ldc + j];
        }
    }
}

// ============================================================================
// Scalar GEMM tests
// ============================================================================

TEST(GemmScalar, SmallIdentity) {
    // A = I(2x2), B = I(2x2), C = I(2x2)
    const int64_t M = 2, N = 2, K = 2;
    std::vector<float> A = {1, 0, 0, 1};
    std::vector<float> B = {1, 0, 0, 1};
    std::vector<float> C(M * N, 0.0f);

    GemmParams params{};
    params.M = M; params.N = N; params.K = K;
    params.A = A.data(); params.B = B.data(); params.C = C.data();
    params.alpha = 1.0f; params.beta = 0.0f;

    gemm_scalar(params);

    EXPECT_FLOAT_EQ(C[0], 1.0f);
    EXPECT_FLOAT_EQ(C[1], 0.0f);
    EXPECT_FLOAT_EQ(C[2], 0.0f);
    EXPECT_FLOAT_EQ(C[3], 1.0f);
}

TEST(GemmScalar, SmallKnown) {
    // A = [[1,2],[3,4]], B = [[5,6],[7,8]]
    // C = A*B = [[19,22],[43,50]]
    const int64_t M = 2, N = 2, K = 2;
    std::vector<float> A = {1, 2, 3, 4};
    std::vector<float> B = {5, 6, 7, 8};
    std::vector<float> C(M * N, 0.0f);

    GemmParams params{};
    params.M = M; params.N = N; params.K = K;
    params.A = A.data(); params.B = B.data(); params.C = C.data();
    params.alpha = 1.0f; params.beta = 0.0f;

    gemm_scalar(params);

    EXPECT_FLOAT_EQ(C[0], 19.0f);
    EXPECT_FLOAT_EQ(C[1], 22.0f);
    EXPECT_FLOAT_EQ(C[2], 43.0f);
    EXPECT_FLOAT_EQ(C[3], 50.0f);
}

TEST(GemmScalar, AlphaBeta) {
    // C = 2 * A * B + 3 * C_init
    const int64_t M = 2, N = 2, K = 2;
    std::vector<float> A = {1, 0, 0, 1};
    std::vector<float> B = {1, 0, 0, 1};
    std::vector<float> C = {10.0f, 20.0f, 30.0f, 40.0f};

    GemmParams params{};
    params.M = M; params.N = N; params.K = K;
    params.A = A.data(); params.B = B.data(); params.C = C.data();
    params.alpha = 2.0f; params.beta = 3.0f;

    gemm_scalar(params);

    EXPECT_FLOAT_EQ(C[0], 2.0f * 1.0f + 3.0f * 10.0f);  // 32
    EXPECT_FLOAT_EQ(C[1], 2.0f * 0.0f + 3.0f * 20.0f);  // 60
    EXPECT_FLOAT_EQ(C[2], 2.0f * 0.0f + 3.0f * 30.0f);  // 90
    EXPECT_FLOAT_EQ(C[3], 2.0f * 1.0f + 3.0f * 40.0f);  // 122
}

TEST(GemmScalar, NonSquare) {
    // A[2,3] * B[3,4] = C[2,4]
    const int64_t M = 2, N = 4, K = 3;
    std::vector<float> A = {1, 2, 3, 4, 5, 6};
    std::vector<float> B = {
        1, 2, 3, 4,
        5, 6, 7, 8,
        9, 10, 11, 12
    };
    std::vector<float> C(M * N, 0.0f);
    std::vector<float> C_ref(M * N, 0.0f);

    ref_gemm(M, N, K, A.data(), K, B.data(), N, C_ref.data(), N, 1.0f, 0.0f);

    GemmParams params{};
    params.M = M; params.N = N; params.K = K;
    params.A = A.data(); params.B = B.data(); params.C = C.data();
    params.alpha = 1.0f; params.beta = 0.0f;

    gemm_scalar(params);

    for (int64_t i = 0; i < M * N; ++i) {
        EXPECT_FLOAT_EQ(C[i], C_ref[i]);
    }
}

TEST(GemmScalar, GEMV_M1) {
    // GEMV: M=1, N=4, K=3 (single row vector * matrix)
    const int64_t M = 1, N = 4, K = 3;
    std::vector<float> A = {1, 2, 3};
    std::vector<float> B = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    std::vector<float> C(N, 0.0f);

    GemmParams params{};
    params.M = M; params.N = N; params.K = K;
    params.A = A.data(); params.B = B.data(); params.C = C.data();

    gemm_scalar(params);

    // C[0] = 1*1 + 2*5 + 3*9 = 1+10+27 = 38
    // C[1] = 1*2 + 2*6 + 3*10 = 2+12+30 = 44
    // C[2] = 1*3 + 2*7 + 3*11 = 3+14+33 = 50
    // C[3] = 1*4 + 2*8 + 3*12 = 4+16+36 = 56
    EXPECT_FLOAT_EQ(C[0], 38.0f);
    EXPECT_FLOAT_EQ(C[1], 44.0f);
    EXPECT_FLOAT_EQ(C[2], 50.0f);
    EXPECT_FLOAT_EQ(C[3], 56.0f);
}

TEST(GemmScalar, GEMV_N1) {
    // M=2, N=1, K=3 (matrix * column vector)
    const int64_t M = 2, N = 1, K = 3;
    std::vector<float> A = {1, 2, 3, 4, 5, 6};
    std::vector<float> B = {1, 2, 3};
    std::vector<float> C(M, 0.0f);

    GemmParams params{};
    params.M = M; params.N = N; params.K = K;
    params.A = A.data(); params.B = B.data(); params.C = C.data();

    gemm_scalar(params);

    // C[0] = 1*1 + 2*2 + 3*3 = 14
    // C[1] = 4*1 + 5*2 + 6*3 = 32
    EXPECT_FLOAT_EQ(C[0], 14.0f);
    EXPECT_FLOAT_EQ(C[1], 32.0f);
}

TEST(GemmScalar, K1) {
    // K=1: outer product
    const int64_t M = 3, N = 4, K = 1;
    std::vector<float> A = {2, 3, 5};
    std::vector<float> B = {1, 2, 3, 4};
    std::vector<float> C(M * N, 0.0f);

    GemmParams params{};
    params.M = M; params.N = N; params.K = K;
    params.A = A.data(); params.B = B.data(); params.C = C.data();

    gemm_scalar(params);

    // C[i,j] = A[i] * B[j]
    EXPECT_FLOAT_EQ(C[0], 2.0f);
    EXPECT_FLOAT_EQ(C[1], 4.0f);
    EXPECT_FLOAT_EQ(C[4], 3.0f);
    EXPECT_FLOAT_EQ(C[8], 5.0f);
    EXPECT_FLOAT_EQ(C[11], 20.0f);
}

TEST(GemmScalar, LargerRandom) {
    const int64_t M = 64, N = 64, K = 64;
    std::vector<float> A(M * K), B(K * N), C(M * N, 0.0f), C_ref(M * N, 0.0f);

    fill_random(A);
    fill_random(B, 123);

    ref_gemm(M, N, K, A.data(), K, B.data(), N, C_ref.data(), N, 1.0f, 0.0f);

    GemmParams params{};
    params.M = M; params.N = N; params.K = K;
    params.A = A.data(); params.B = B.data(); params.C = C.data();

    gemm_scalar(params);

    for (int64_t i = 0; i < M * N; ++i) {
        EXPECT_NEAR(C[i], C_ref[i], 1e-3f);
    }
}

TEST(GemmScalar, OddDimensions) {
    // Dimensions that don't align to tile sizes
    const int64_t M = 7, N = 13, K = 11;
    std::vector<float> A(M * K), B(K * N), C(M * N, 0.0f), C_ref(M * N, 0.0f);

    fill_sequential(A);
    fill_sequential(B);

    ref_gemm(M, N, K, A.data(), K, B.data(), N, C_ref.data(), N, 1.0f, 0.0f);

    GemmParams params{};
    params.M = M; params.N = N; params.K = K;
    params.A = A.data(); params.B = B.data(); params.C = C.data();

    gemm_scalar(params);

    for (int64_t i = 0; i < M * N; ++i) {
        EXPECT_NEAR(C[i], C_ref[i], 1e-2f) << "at index " << i;
    }
}

TEST(GemmScalar, ZeroAlpha) {
    const int64_t M = 2, N = 2, K = 2;
    std::vector<float> A = {1, 2, 3, 4};
    std::vector<float> B = {5, 6, 7, 8};
    std::vector<float> C = {100.0f, 200.0f, 300.0f, 400.0f};

    GemmParams params{};
    params.M = M; params.N = N; params.K = K;
    params.A = A.data(); params.B = B.data(); params.C = C.data();
    params.alpha = 0.0f; params.beta = 1.0f;

    gemm_scalar(params);

    // C should be unchanged (beta * C_init)
    EXPECT_FLOAT_EQ(C[0], 100.0f);
    EXPECT_FLOAT_EQ(C[1], 200.0f);
    EXPECT_FLOAT_EQ(C[2], 300.0f);
    EXPECT_FLOAT_EQ(C[3], 400.0f);
}

TEST(GemmScalar, ZeroAlphaZeroBeta) {
    // C = 0 * A * B + 0 * C = all zeros
    const int64_t M = 3, N = 4, K = 2;
    std::vector<float> A(M * K, 5.0f);
    std::vector<float> B(K * N, 3.0f);
    std::vector<float> C(M * N, 999.0f);

    GemmParams params{};
    params.M = M; params.N = N; params.K = K;
    params.A = A.data(); params.B = B.data(); params.C = C.data();
    params.alpha = 0.0f; params.beta = 0.0f;

    gemm_scalar(params);

    for (int64_t i = 0; i < M * N; ++i) {
        EXPECT_FLOAT_EQ(C[i], 0.0f);
    }
}

TEST(GemmScalar, UnitAlphaBeta) {
    // C = 1 * A * B + 1 * C_init (accumulate into existing C)
    const int64_t M = 2, N = 3, K = 2;
    std::vector<float> A = {1, 2, 3, 4};
    std::vector<float> B = {1, 2, 3, 4, 5, 6};
    std::vector<float> C = {10, 20, 30, 40, 50, 60};

    GemmParams params{};
    params.M = M; params.N = N; params.K = K;
    params.A = A.data(); params.B = B.data(); params.C = C.data();
    params.alpha = 1.0f; params.beta = 1.0f;

    std::vector<float> C_ref = C;
    ref_gemm(M, N, K, A.data(), K, B.data(), N, C_ref.data(), N, 1.0f, 1.0f);

    gemm_scalar(params);

    for (int64_t i = 0; i < M * N; ++i) {
        EXPECT_NEAR(C[i], C_ref[i], 1e-4f);
    }
}

// ============================================================================
// Dispatcher tests
// ============================================================================

TEST(GemmDispatch, SelectsValidTarget) {
    auto target = select_kernel_target();
    EXPECT_TRUE(target == KernelTarget::AMX ||
                target == KernelTarget::AVX512 ||
                target == KernelTarget::AVX2 ||
                target == KernelTarget::Scalar);
}

TEST(GemmDispatch, DispatchMatchesScalar) {
    // gemm_dispatch should produce the same result as gemm_scalar
    const int64_t M = 4, N = 8, K = 4;
    std::vector<float> A(M * K), B(K * N), C_dispatch(M * N, 0.0f), C_scalar(M * N, 0.0f);

    fill_random(A);
    fill_random(B, 99);

    GemmParams params{};
    params.M = M; params.N = N; params.K = K;
    params.A = A.data(); params.B = B.data();

    // Scalar
    params.C = C_scalar.data();
    gemm_scalar(params);

    // Dispatch
    params.C = C_dispatch.data();
    gemm_dispatch(params);

    for (int64_t i = 0; i < M * N; ++i) {
        EXPECT_NEAR(C_dispatch[i], C_scalar[i], 1e-4f);
    }
}

TEST(GemmDispatch, LargerMatrix) {
    // Test with a moderately large matrix
    const int64_t M = 32, N = 48, K = 32;
    std::vector<float> A(M * K), B(K * N), C_dispatch(M * N, 0.0f), C_scalar(M * N, 0.0f);

    fill_random(A);
    fill_random(B, 200);

    GemmParams params{};
    params.M = M; params.N = N; params.K = K;
    params.A = A.data(); params.B = B.data();

    params.C = C_scalar.data();
    gemm_scalar(params);

    params.C = C_dispatch.data();
    gemm_dispatch(params);

    for (int64_t i = 0; i < M * N; ++i) {
        EXPECT_NEAR(C_dispatch[i], C_scalar[i], 1e-3f) << "at index " << i;
    }
}

// ============================================================================
// AVX2 GEMM tests (runtime-checked)
// ============================================================================

TEST(GemmAVX2, MatchesScalar) {
    const auto& info = CpuFeatureDetector::get();
    if (!(info.flags.avx2 && info.flags.os_avx)) {
        GTEST_SKIP() << "AVX2 not available";
    }

    const int64_t M = 16, N = 32, K = 16;
    std::vector<float> A(M * K), B(K * N), C_avx2(M * N, 0.0f), C_scalar(M * N, 0.0f);

    fill_random(A);
    fill_random(B, 77);

    GemmParams params{};
    params.M = M; params.N = N; params.K = K;
    params.A = A.data(); params.B = B.data();

    params.C = C_scalar.data();
    gemm_scalar(params);

    params.C = C_avx2.data();
    gemm_avx2(params);

    for (int64_t i = 0; i < M * N; ++i) {
        EXPECT_NEAR(C_avx2[i], C_scalar[i], 1e-4f) << "at index " << i;
    }
}

TEST(GemmAVX2, OddDimensions) {
    const auto& info = CpuFeatureDetector::get();
    if (!(info.flags.avx2 && info.flags.os_avx)) {
        GTEST_SKIP() << "AVX2 not available";
    }

    const int64_t M = 5, N = 11, K = 7;
    std::vector<float> A(M * K), B(K * N), C_avx2(M * N, 0.0f), C_scalar(M * N, 0.0f);

    fill_sequential(A);
    fill_sequential(B);

    GemmParams params{};
    params.M = M; params.N = N; params.K = K;
    params.A = A.data(); params.B = B.data();

    params.C = C_scalar.data();
    gemm_scalar(params);

    params.C = C_avx2.data();
    gemm_avx2(params);

    for (int64_t i = 0; i < M * N; ++i) {
        EXPECT_NEAR(C_avx2[i], C_scalar[i], 1e-2f) << "at index " << i;
    }
}

TEST(GemmAVX2, GEMV_M1) {
    const auto& info = CpuFeatureDetector::get();
    if (!(info.flags.avx2 && info.flags.os_avx)) {
        GTEST_SKIP() << "AVX2 not available";
    }

    const int64_t M = 1, N = 16, K = 8;
    std::vector<float> A(M * K), B(K * N), C_avx2(M * N, 0.0f), C_scalar(M * N, 0.0f);

    fill_random(A);
    fill_random(B, 88);

    GemmParams params{};
    params.M = M; params.N = N; params.K = K;
    params.A = A.data(); params.B = B.data();

    params.C = C_scalar.data();
    gemm_scalar(params);

    params.C = C_avx2.data();
    gemm_avx2(params);

    for (int64_t i = 0; i < M * N; ++i) {
        EXPECT_NEAR(C_avx2[i], C_scalar[i], 1e-4f) << "at index " << i;
    }
}

TEST(GemmAVX2, AlphaBeta) {
    const auto& info = CpuFeatureDetector::get();
    if (!(info.flags.avx2 && info.flags.os_avx)) {
        GTEST_SKIP() << "AVX2 not available";
    }

    const int64_t M = 4, N = 8, K = 4;
    std::vector<float> A(M * K), B(K * N);
    std::vector<float> C_avx2(M * N), C_scalar(M * N);

    fill_random(A);
    fill_random(B, 66);
    for (int64_t i = 0; i < M * N; ++i) {
        C_avx2[i] = C_scalar[i] = static_cast<float>(i);
    }

    GemmParams params{};
    params.M = M; params.N = N; params.K = K;
    params.A = A.data(); params.B = B.data();
    params.alpha = 2.5f; params.beta = 0.5f;

    params.C = C_scalar.data();
    gemm_scalar(params);

    params.C = C_avx2.data();
    gemm_avx2(params);

    for (int64_t i = 0; i < M * N; ++i) {
        EXPECT_NEAR(C_avx2[i], C_scalar[i], 1e-3f) << "at index " << i;
    }
}

// ============================================================================
// AVX-512 FP32 GEMM tests (runtime-checked)
// ============================================================================

TEST(GemmAVX512, MatchesScalar) {
    const auto& info = CpuFeatureDetector::get();
    if (!(info.flags.avx512f && info.flags.os_avx512)) {
        GTEST_SKIP() << "AVX-512 not available";
    }

    const int64_t M = 16, N = 32, K = 16;
    std::vector<float> A(M * K), B(K * N), C_avx512(M * N, 0.0f), C_scalar(M * N, 0.0f);

    fill_random(A);
    fill_random(B, 55);

    GemmParams params{};
    params.M = M; params.N = N; params.K = K;
    params.A = A.data(); params.B = B.data();

    params.C = C_scalar.data();
    gemm_scalar(params);

    params.C = C_avx512.data();
    gemm_avx512(params);

    for (int64_t i = 0; i < M * N; ++i) {
        EXPECT_NEAR(C_avx512[i], C_scalar[i], 1e-4f) << "at index " << i;
    }
}

TEST(GemmAVX512, OddDimensions) {
    const auto& info = CpuFeatureDetector::get();
    if (!(info.flags.avx512f && info.flags.os_avx512)) {
        GTEST_SKIP() << "AVX-512 not available";
    }

    const int64_t M = 7, N = 23, K = 11;
    std::vector<float> A(M * K), B(K * N), C_avx512(M * N, 0.0f), C_scalar(M * N, 0.0f);

    fill_sequential(A);
    fill_sequential(B);

    GemmParams params{};
    params.M = M; params.N = N; params.K = K;
    params.A = A.data(); params.B = B.data();

    params.C = C_scalar.data();
    gemm_scalar(params);

    params.C = C_avx512.data();
    gemm_avx512(params);

    for (int64_t i = 0; i < M * N; ++i) {
        EXPECT_NEAR(C_avx512[i], C_scalar[i], 1e-2f) << "at index " << i;
    }
}

TEST(GemmAVX512, GEMV_M1) {
    const auto& info = CpuFeatureDetector::get();
    if (!(info.flags.avx512f && info.flags.os_avx512)) {
        GTEST_SKIP() << "AVX-512 not available";
    }

    const int64_t M = 1, N = 32, K = 16;
    std::vector<float> A(M * K), B(K * N), C_avx512(M * N, 0.0f), C_scalar(M * N, 0.0f);

    fill_random(A);
    fill_random(B, 88);

    GemmParams params{};
    params.M = M; params.N = N; params.K = K;
    params.A = A.data(); params.B = B.data();

    params.C = C_scalar.data();
    gemm_scalar(params);

    params.C = C_avx512.data();
    gemm_avx512(params);

    for (int64_t i = 0; i < M * N; ++i) {
        EXPECT_NEAR(C_avx512[i], C_scalar[i], 1e-4f) << "at index " << i;
    }
}

TEST(GemmAVX512, AlphaBeta) {
    const auto& info = CpuFeatureDetector::get();
    if (!(info.flags.avx512f && info.flags.os_avx512)) {
        GTEST_SKIP() << "AVX-512 not available";
    }

    const int64_t M = 4, N = 16, K = 4;
    std::vector<float> A(M * K), B(K * N);
    std::vector<float> C_avx512(M * N), C_scalar(M * N);

    fill_random(A);
    fill_random(B, 66);
    for (int64_t i = 0; i < M * N; ++i) {
        C_avx512[i] = C_scalar[i] = static_cast<float>(i);
    }

    GemmParams params{};
    params.M = M; params.N = N; params.K = K;
    params.A = A.data(); params.B = B.data();
    params.alpha = 2.5f; params.beta = 0.5f;

    params.C = C_scalar.data();
    gemm_scalar(params);

    params.C = C_avx512.data();
    gemm_avx512(params);

    for (int64_t i = 0; i < M * N; ++i) {
        EXPECT_NEAR(C_avx512[i], C_scalar[i], 1e-3f) << "at index " << i;
    }
}

TEST(GemmAVX512, LargerRandom) {
    const auto& info = CpuFeatureDetector::get();
    if (!(info.flags.avx512f && info.flags.os_avx512)) {
        GTEST_SKIP() << "AVX-512 not available";
    }

    // Test with a larger matrix to exercise more of the tiling logic
    const int64_t M = 64, N = 64, K = 64;
    std::vector<float> A(M * K), B(K * N), C_avx512(M * N, 0.0f), C_scalar(M * N, 0.0f);

    fill_random(A, 42);
    fill_random(B, 99);

    GemmParams params{};
    params.M = M; params.N = N; params.K = K;
    params.A = A.data(); params.B = B.data();

    params.C = C_scalar.data();
    gemm_scalar(params);

    params.C = C_avx512.data();
    gemm_avx512(params);

    for (int64_t i = 0; i < M * N; ++i) {
        EXPECT_NEAR(C_avx512[i], C_scalar[i], 1e-3f) << "at index " << i;
    }
}

TEST(GemmAVX512, GEMV_N1) {
    const auto& info = CpuFeatureDetector::get();
    if (!(info.flags.avx512f && info.flags.os_avx512)) {
        GTEST_SKIP() << "AVX-512 not available";
    }

    // M=3, N=1, K=16: matrix * column vector
    const int64_t M = 3, N = 1, K = 16;
    std::vector<float> A(M * K), B(K), C_avx512(M, 0.0f), C_scalar(M, 0.0f);

    fill_random(A);
    fill_random(B, 77);

    GemmParams params{};
    params.M = M; params.N = N; params.K = K;
    params.A = A.data(); params.B = B.data();

    params.C = C_scalar.data();
    gemm_scalar(params);

    params.C = C_avx512.data();
    gemm_avx512(params);

    for (int64_t i = 0; i < M; ++i) {
        EXPECT_NEAR(C_avx512[i], C_scalar[i], 1e-4f) << "at index " << i;
    }
}

// ============================================================================
// AVX-512 BF16 GEMM tests (runtime-checked)
// ============================================================================

TEST(GemmBF16, MatchesScalar) {
    const auto& info = CpuFeatureDetector::get();
    if (!(info.flags.avx512bf16 && info.flags.avx512f && info.flags.os_avx512)) {
        GTEST_SKIP() << "AVX-512 BF16 not available";
    }

    const int64_t M = 16, N = 32, K = 16;
    std::vector<float> A_fp32(M * K), B_fp32(K * N);
    fill_random(A_fp32);
    fill_random(B_fp32, 33);

    // Convert to BF16
    std::vector<uint16_t> A_bf16(M * K), B_bf16(K * N);
    for (int64_t i = 0; i < M * K; ++i) A_bf16[i] = f32_to_bf16(A_fp32[i]);
    for (int64_t i = 0; i < K * N; ++i) B_bf16[i] = f32_to_bf16(B_fp32[i]);

    // Compute FP32 reference using BF16-converted values
    std::vector<float> A_ref(M * K), B_ref(K * N);
    for (int64_t i = 0; i < M * K; ++i) A_ref[i] = bf16_to_f32(A_bf16[i]);
    for (int64_t i = 0; i < K * N; ++i) B_ref[i] = bf16_to_f32(B_bf16[i]);

    std::vector<float> C_scalar(M * N, 0.0f);
    ref_gemm(M, N, K, A_ref.data(), K, B_ref.data(), N, C_scalar.data(), N, 1.0f, 0.0f);

    std::vector<float> C_bf16(M * N, 0.0f);
    GemmParamsBF16 params{};
    params.M = M; params.N = N; params.K = K;
    params.A = A_bf16.data(); params.B = B_bf16.data(); params.C = C_bf16.data();

    gemm_bf16_avx512(params);

    for (int64_t i = 0; i < M * N; ++i) {
        EXPECT_NEAR(C_bf16[i], C_scalar[i], 1e-1f) << "at index " << i;
    }
}

TEST(GemmBF16, OddDimensions) {
    const auto& info = CpuFeatureDetector::get();
    if (!(info.flags.avx512bf16 && info.flags.avx512f && info.flags.os_avx512)) {
        GTEST_SKIP() << "AVX-512 BF16 not available";
    }

    const int64_t M = 5, N = 21, K = 9;
    std::vector<float> A_fp32(M * K), B_fp32(K * N);
    fill_random(A_fp32);
    fill_random(B_fp32, 44);

    std::vector<uint16_t> A_bf16(M * K), B_bf16(K * N);
    for (int64_t i = 0; i < M * K; ++i) A_bf16[i] = f32_to_bf16(A_fp32[i]);
    for (int64_t i = 0; i < K * N; ++i) B_bf16[i] = f32_to_bf16(B_fp32[i]);

    std::vector<float> A_ref(M * K), B_ref(K * N);
    for (int64_t i = 0; i < M * K; ++i) A_ref[i] = bf16_to_f32(A_bf16[i]);
    for (int64_t i = 0; i < K * N; ++i) B_ref[i] = bf16_to_f32(B_bf16[i]);

    std::vector<float> C_scalar(M * N, 0.0f);
    ref_gemm(M, N, K, A_ref.data(), K, B_ref.data(), N, C_scalar.data(), N, 1.0f, 0.0f);

    std::vector<float> C_bf16(M * N, 0.0f);
    GemmParamsBF16 params{};
    params.M = M; params.N = N; params.K = K;
    params.A = A_bf16.data(); params.B = B_bf16.data(); params.C = C_bf16.data();

    gemm_bf16_avx512(params);

    for (int64_t i = 0; i < M * N; ++i) {
        EXPECT_NEAR(C_bf16[i], C_scalar[i], 1e-1f) << "at index " << i;
    }
}

TEST(GemmBF16, GEMV) {
    const auto& info = CpuFeatureDetector::get();
    if (!(info.flags.avx512bf16 && info.flags.avx512f && info.flags.os_avx512)) {
        GTEST_SKIP() << "AVX-512 BF16 not available";
    }

    const int64_t M = 1, N = 32, K = 16;
    std::vector<float> A_fp32(M * K), B_fp32(K * N);
    fill_random(A_fp32);
    fill_random(B_fp32, 55);

    std::vector<uint16_t> A_bf16(M * K), B_bf16(K * N);
    for (int64_t i = 0; i < M * K; ++i) A_bf16[i] = f32_to_bf16(A_fp32[i]);
    for (int64_t i = 0; i < K * N; ++i) B_bf16[i] = f32_to_bf16(B_fp32[i]);

    std::vector<float> A_ref(M * K), B_ref(K * N);
    for (int64_t i = 0; i < M * K; ++i) A_ref[i] = bf16_to_f32(A_bf16[i]);
    for (int64_t i = 0; i < K * N; ++i) B_ref[i] = bf16_to_f32(B_bf16[i]);

    std::vector<float> C_scalar(M * N, 0.0f);
    ref_gemm(M, N, K, A_ref.data(), K, B_ref.data(), N, C_scalar.data(), N, 1.0f, 0.0f);

    std::vector<float> C_bf16(M * N, 0.0f);
    GemmParamsBF16 params{};
    params.M = M; params.N = N; params.K = K;
    params.A = A_bf16.data(); params.B = B_bf16.data(); params.C = C_bf16.data();

    gemm_bf16_avx512(params);

    for (int64_t i = 0; i < M * N; ++i) {
        EXPECT_NEAR(C_bf16[i], C_scalar[i], 1e-1f) << "at index " << i;
    }
}

TEST(GemmBF16, OddK) {
    // Test with odd K dimension (tests remainder handling in BF16 dpbf16_ps loop)
    const auto& info = CpuFeatureDetector::get();
    if (!(info.flags.avx512bf16 && info.flags.avx512f && info.flags.os_avx512)) {
        GTEST_SKIP() << "AVX-512 BF16 not available";
    }

    const int64_t M = 4, N = 16, K = 7;  // Odd K tests the single-remainder path
    std::vector<float> A_fp32(M * K), B_fp32(K * N);
    fill_random(A_fp32, 111);
    fill_random(B_fp32, 222);

    std::vector<uint16_t> A_bf16(M * K), B_bf16(K * N);
    for (int64_t i = 0; i < M * K; ++i) A_bf16[i] = f32_to_bf16(A_fp32[i]);
    for (int64_t i = 0; i < K * N; ++i) B_bf16[i] = f32_to_bf16(B_fp32[i]);

    std::vector<float> A_ref(M * K), B_ref(K * N);
    for (int64_t i = 0; i < M * K; ++i) A_ref[i] = bf16_to_f32(A_bf16[i]);
    for (int64_t i = 0; i < K * N; ++i) B_ref[i] = bf16_to_f32(B_bf16[i]);

    std::vector<float> C_scalar(M * N, 0.0f);
    ref_gemm(M, N, K, A_ref.data(), K, B_ref.data(), N, C_scalar.data(), N, 1.0f, 0.0f);

    std::vector<float> C_bf16(M * N, 0.0f);
    GemmParamsBF16 params{};
    params.M = M; params.N = N; params.K = K;
    params.A = A_bf16.data(); params.B = B_bf16.data(); params.C = C_bf16.data();

    gemm_bf16_avx512(params);

    for (int64_t i = 0; i < M * N; ++i) {
        EXPECT_NEAR(C_bf16[i], C_scalar[i], 1e-1f) << "at index " << i;
    }
}

// ============================================================================
// AVX-512 VNNI GEMM tests (runtime-checked)
// ============================================================================

TEST(GemmVNNI, SmallKnown) {
    const auto& info = CpuFeatureDetector::get();
    if (!(info.flags.avx512vnni && info.flags.avx512f && info.flags.os_avx512)) {
        GTEST_SKIP() << "AVX-512 VNNI not available";
    }

    // A[2,4] uint8, B[4,16] int8, C[2,16] int32
    const int64_t M = 2, N = 16, K = 4;
    std::vector<uint8_t> A = {1, 2, 3, 4, 5, 6, 7, 8};
    std::vector<int8_t> B(K * N);
    for (int64_t i = 0; i < K * N; ++i) B[i] = static_cast<int8_t>((i % 7) - 3);

    std::vector<int32_t> C_vnni(M * N, 0);
    std::vector<int32_t> C_ref(M * N, 0);

    // Compute reference
    for (int64_t i = 0; i < M; ++i) {
        for (int64_t j = 0; j < N; ++j) {
            int32_t sum = 0;
            for (int64_t k = 0; k < K; ++k) {
                sum += static_cast<int32_t>(A[i * K + k]) * static_cast<int32_t>(B[k * N + j]);
            }
            C_ref[i * N + j] = sum;
        }
    }

    GemmParamsVNNI params{};
    params.M = M; params.N = N; params.K = K;
    params.A = A.data(); params.B = B.data(); params.C = C_vnni.data();

    gemm_vnni_avx512(params);

    for (int64_t i = 0; i < M * N; ++i) {
        EXPECT_EQ(C_vnni[i], C_ref[i]) << "at index " << i;
    }
}

TEST(GemmVNNI, MatchesScalar) {
    const auto& info = CpuFeatureDetector::get();
    if (!(info.flags.avx512vnni && info.flags.avx512f && info.flags.os_avx512)) {
        GTEST_SKIP() << "AVX-512 VNNI not available";
    }

    const int64_t M = 16, N = 32, K = 16;
    std::vector<uint8_t> A(M * K);
    std::vector<int8_t> B(K * N);
    fill_random_u8(A);
    fill_random_i8(B);

    std::vector<int32_t> C_vnni(M * N, 0);
    std::vector<int32_t> C_ref(M * N, 0);

    // Reference
    for (int64_t i = 0; i < M; ++i) {
        for (int64_t j = 0; j < N; ++j) {
            int32_t sum = 0;
            for (int64_t k = 0; k < K; ++k) {
                sum += static_cast<int32_t>(A[i * K + k]) * static_cast<int32_t>(B[k * N + j]);
            }
            C_ref[i * N + j] = sum;
        }
    }

    GemmParamsVNNI params{};
    params.M = M; params.N = N; params.K = K;
    params.A = A.data(); params.B = B.data(); params.C = C_vnni.data();

    gemm_vnni_avx512(params);

    for (int64_t i = 0; i < M * N; ++i) {
        EXPECT_EQ(C_vnni[i], C_ref[i]) << "at index " << i;
    }
}

TEST(GemmVNNI, OddDimensions) {
    const auto& info = CpuFeatureDetector::get();
    if (!(info.flags.avx512vnni && info.flags.avx512f && info.flags.os_avx512)) {
        GTEST_SKIP() << "AVX-512 VNNI not available";
    }

    const int64_t M = 5, N = 21, K = 7;
    std::vector<uint8_t> A(M * K);
    std::vector<int8_t> B(K * N);
    fill_random_u8(A, 111);
    fill_random_i8(B, 222);

    std::vector<int32_t> C_vnni(M * N, 0);
    std::vector<int32_t> C_ref(M * N, 0);

    for (int64_t i = 0; i < M; ++i) {
        for (int64_t j = 0; j < N; ++j) {
            int32_t sum = 0;
            for (int64_t k = 0; k < K; ++k) {
                sum += static_cast<int32_t>(A[i * K + k]) * static_cast<int32_t>(B[k * N + j]);
            }
            C_ref[i * N + j] = sum;
        }
    }

    GemmParamsVNNI params{};
    params.M = M; params.N = N; params.K = K;
    params.A = A.data(); params.B = B.data(); params.C = C_vnni.data();

    gemm_vnni_avx512(params);

    for (int64_t i = 0; i < M * N; ++i) {
        EXPECT_EQ(C_vnni[i], C_ref[i]) << "at index " << i;
    }
}

TEST(GemmVNNI, NonMultipleK) {
    // K not a multiple of 4 (tests scalar remainder path in VNNI kernel)
    const auto& info = CpuFeatureDetector::get();
    if (!(info.flags.avx512vnni && info.flags.avx512f && info.flags.os_avx512)) {
        GTEST_SKIP() << "AVX-512 VNNI not available";
    }

    const int64_t M = 4, N = 16, K = 7;  // K=7: 4+3 remainder
    std::vector<uint8_t> A(M * K);
    std::vector<int8_t> B(K * N);
    fill_random_u8(A, 333);
    fill_random_i8(B, 444);

    std::vector<int32_t> C_vnni(M * N, 0);
    std::vector<int32_t> C_ref(M * N, 0);

    for (int64_t i = 0; i < M; ++i) {
        for (int64_t j = 0; j < N; ++j) {
            int32_t sum = 0;
            for (int64_t k = 0; k < K; ++k) {
                sum += static_cast<int32_t>(A[i * K + k]) * static_cast<int32_t>(B[k * N + j]);
            }
            C_ref[i * N + j] = sum;
        }
    }

    GemmParamsVNNI params{};
    params.M = M; params.N = N; params.K = K;
    params.A = A.data(); params.B = B.data(); params.C = C_vnni.data();

    gemm_vnni_avx512(params);

    for (int64_t i = 0; i < M * N; ++i) {
        EXPECT_EQ(C_vnni[i], C_ref[i]) << "at index " << i;
    }
}

TEST(GemmVNNI, GEMV_M1) {
    // M=1: single row of A (tests mr=1 edge case)
    const auto& info = CpuFeatureDetector::get();
    if (!(info.flags.avx512vnni && info.flags.avx512f && info.flags.os_avx512)) {
        GTEST_SKIP() << "AVX-512 VNNI not available";
    }

    const int64_t M = 1, N = 16, K = 8;
    std::vector<uint8_t> A(M * K);
    std::vector<int8_t> B(K * N);
    fill_random_u8(A, 555);
    fill_random_i8(B, 666);

    std::vector<int32_t> C_vnni(M * N, 0);
    std::vector<int32_t> C_ref(M * N, 0);

    for (int64_t j = 0; j < N; ++j) {
        int32_t sum = 0;
        for (int64_t k = 0; k < K; ++k) {
            sum += static_cast<int32_t>(A[k]) * static_cast<int32_t>(B[k * N + j]);
        }
        C_ref[j] = sum;
    }

    GemmParamsVNNI params{};
    params.M = M; params.N = N; params.K = K;
    params.A = A.data(); params.B = B.data(); params.C = C_vnni.data();

    gemm_vnni_avx512(params);

    for (int64_t j = 0; j < N; ++j) {
        EXPECT_EQ(C_vnni[j], C_ref[j]) << "at index " << j;
    }
}

// ============================================================================
// Edge case tests
// ============================================================================

TEST(GemmScalar, EmptyM) {
    std::vector<float> A, B, C;
    GemmParams params{};
    params.M = 0; params.N = 4; params.K = 4;
    EXPECT_NO_THROW(gemm_scalar(params));
}

TEST(GemmScalar, EmptyN) {
    std::vector<float> A(4, 1.0f), B, C;
    GemmParams params{};
    params.M = 4; params.N = 0; params.K = 4;
    params.A = A.data();
    EXPECT_NO_THROW(gemm_scalar(params));
}

TEST(GemmScalar, EmptyK) {
    std::vector<float> A, B, C(4, 0.0f);
    GemmParams params{};
    params.M = 2; params.N = 2; params.K = 0;
    params.C = C.data();
    EXPECT_NO_THROW(gemm_scalar(params));
}

TEST(GemmScalar, M1N1K1) {
    // Smallest possible GEMM: 1x1 = 1x1 * 1x1
    std::vector<float> A = {3.0f};
    std::vector<float> B = {7.0f};
    std::vector<float> C = {0.0f};

    GemmParams params{};
    params.M = 1; params.N = 1; params.K = 1;
    params.A = A.data(); params.B = B.data(); params.C = C.data();

    gemm_scalar(params);

    EXPECT_FLOAT_EQ(C[0], 21.0f);
}

TEST(GemmDispatch, WithAlphaBeta) {
    // Test dispatch with non-trivial alpha/beta
    const int64_t M = 4, N = 8, K = 4;
    std::vector<float> A(M * K), B(K * N);
    std::vector<float> C_dispatch(M * N), C_scalar(M * N);

    fill_random(A);
    fill_random(B, 99);
    for (int64_t i = 0; i < M * N; ++i) {
        C_dispatch[i] = C_scalar[i] = static_cast<float>(i + 1);
    }

    GemmParams params{};
    params.M = M; params.N = N; params.K = K;
    params.A = A.data(); params.B = B.data();
    params.alpha = 0.5f; params.beta = 2.0f;

    params.C = C_scalar.data();
    gemm_scalar(params);

    params.C = C_dispatch.data();
    gemm_dispatch(params);

    for (int64_t i = 0; i < M * N; ++i) {
        EXPECT_NEAR(C_dispatch[i], C_scalar[i], 1e-3f) << "at index " << i;
    }
}
