#include <gtest/gtest.h>
#include <vector>
#include <cmath>
#include <cstring>
#include <random>
#include <cstdint>

#include "kaguya/kernels/quantize.h"
#include "kaguya/tensor.h"

using namespace kaguya;
using namespace kaguya::kernels;

// ============================================================================
// Helper utilities
// ============================================================================

/// Fill a vector with random floats in [-1, 1]
static void fill_random(std::vector<float>& v, unsigned seed = 42) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (auto& x : v) x = dist(rng);
}

/// Fill a vector with random floats in [lo, hi]
static void fill_random_range(std::vector<float>& v, float lo, float hi, unsigned seed = 42) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(lo, hi);
    for (auto& x : v) x = dist(rng);
}

/// Compute max absolute error between two vectors
static float max_abs_error(const std::vector<float>& a, const std::vector<float>& b) {
    float max_err = 0.0f;
    for (size_t i = 0; i < a.size(); ++i) {
        max_err = std::max(max_err, std::fabs(a[i] - b[i]));
    }
    return max_err;
}

/// Simple FP32 → FP16 conversion for test construction (same as quantize.cpp)
static inline uint16_t test_fp32_to_fp16(float f) {
    uint32_t x;
    std::memcpy(&x, &f, sizeof(x));
    uint32_t sign = (x >> 16) & 0x8000;
    int32_t exponent = static_cast<int32_t>((x >> 23) & 0xFF) - 127;
    uint32_t mantissa = x & 0x7FFFFF;
    if (exponent > 15) return static_cast<uint16_t>(sign | 0x7C00);
    if (exponent < -14) return static_cast<uint16_t>(sign);
    uint32_t new_exp = static_cast<uint32_t>(exponent + 15);
    uint32_t new_mant = mantissa >> 13;
    return static_cast<uint16_t>(sign | (new_exp << 10) | new_mant);
}

// ============================================================================
// Q4_0 Dequantization tests
// ============================================================================

TEST(QuantizeQ4_0, DequantizeSingleBlock) {
    // Construct a single Q4_0 block with known values
    // Block: 18 bytes = {uint16_t d, uint8_t qs[16]}
    // d = 1.0 (fp16), qs[0] = 0x98 (nibble 8=0, nibble 9=1 in -8..7)
    alignas(2) uint8_t block[18] = {};

    // Set d = 1.0 in fp16 = 0x3C00
    uint16_t d_fp16 = test_fp32_to_fp16(1.0f);
    block[0] = static_cast<uint8_t>(d_fp16 & 0xFF);
    block[1] = static_cast<uint8_t>((d_fp16 >> 8) & 0xFF);

    // Set all qs bytes so that both nibbles are 8 (=0 when dequantized: (8-8)*1.0=0)
    for (int i = 0; i < 16; ++i) {
        block[2 + i] = 0x88; // low=8, high=8 → both produce 0
    }

    float out[32] = {};
    dequantize_q4_0_block(block, out);

    for (int i = 0; i < 32; ++i) {
        EXPECT_NEAR(out[i], 0.0f, 1e-6f) << "at index " << i;
    }
}

TEST(QuantizeQ4_0, DequantizeKnownValues) {
    // d = 0.5, first byte qs[0] = 0x00 → low nibble 0 → (0-8)*0.5 = -4.0
    //                                    → high nibble 0 → (0-8)*0.5 = -4.0
    alignas(2) uint8_t block[18] = {};
    uint16_t d_fp16 = test_fp32_to_fp16(0.5f);
    block[0] = static_cast<uint8_t>(d_fp16 & 0xFF);
    block[1] = static_cast<uint8_t>((d_fp16 >> 8) & 0xFF);
    block[2] = 0x00; // qs[0]: low=0, high=0
    for (int i = 1; i < 16; ++i) block[2 + i] = 0x88;

    float out[32] = {};
    dequantize_q4_0_block(block, out);

    EXPECT_NEAR(out[0], -4.0f, 1e-5f);     // (0-8)*0.5
    EXPECT_NEAR(out[16], -4.0f, 1e-5f);    // (0-8)*0.5
}

TEST(QuantizeQ4_0, Roundtrip) {
    // Quantize random FP32 → Q4_0 → dequantize back, verify within tolerance
    const int64_t n_blocks = 8;
    const int64_t n_elements = n_blocks * 32;
    std::vector<float> input(n_elements);
    fill_random_range(input, -0.5f, 0.5f, 123);

    std::vector<uint8_t> quantized(static_cast<size_t>(n_blocks) * 18);
    std::vector<float> output(n_elements);

    quantize_q4_0(input.data(), quantized.data(), n_blocks);
    dequantize_q4_0(quantized.data(), output.data(), n_blocks);

    // Q4_0 has ~4-bit precision, tolerance ~0.1
    float max_err = max_abs_error(input, output);
    EXPECT_LT(max_err, 0.15f) << "Q4_0 roundtrip max error: " << max_err;
}

TEST(QuantizeQ4_0, RoundtripZeroInput) {
    const int64_t n_blocks = 2;
    const int64_t n_elements = n_blocks * 32;
    std::vector<float> input(n_elements, 0.0f);
    std::vector<uint8_t> quantized(static_cast<size_t>(n_blocks) * 18);
    std::vector<float> output(n_elements);

    quantize_q4_0(input.data(), quantized.data(), n_blocks);
    dequantize_q4_0(quantized.data(), output.data(), n_blocks);

    for (int64_t i = 0; i < n_elements; ++i) {
        EXPECT_NEAR(output[i], 0.0f, 1e-6f) << "at index " << i;
    }
}

// ============================================================================
// Q8_0 Dequantization tests
// ============================================================================

TEST(QuantizeQ8_0, DequantizeSingleBlock) {
    // Q8_0 block: 34 bytes = {uint16_t d, int8_t qs[32]}
    alignas(2) uint8_t block[34] = {};
    uint16_t d_fp16 = test_fp32_to_fp16(1.0f);
    block[0] = static_cast<uint8_t>(d_fp16 & 0xFF);
    block[1] = static_cast<uint8_t>((d_fp16 >> 8) & 0xFF);

    // Set qs[0] = 10, rest = 0
    block[2] = 10;

    float out[32] = {};
    dequantize_q8_0(block, out, 1);

    EXPECT_NEAR(out[0], 10.0f, 1e-5f);
    for (int i = 1; i < 32; ++i) {
        EXPECT_NEAR(out[i], 0.0f, 1e-5f) << "at index " << i;
    }
}

TEST(QuantizeQ8_0, Roundtrip) {
    const int64_t n_blocks = 8;
    const int64_t n_elements = n_blocks * 32;
    std::vector<float> input(n_elements);
    fill_random_range(input, -0.5f, 0.5f, 456);

    std::vector<uint8_t> quantized(static_cast<size_t>(n_blocks) * 34);
    std::vector<float> output(n_elements);

    quantize_q8_0(input.data(), quantized.data(), n_blocks);
    dequantize_q8_0(quantized.data(), output.data(), n_blocks);

    // Q8_0 has ~8-bit precision, tolerance ~0.01
    float max_err = max_abs_error(input, output);
    EXPECT_LT(max_err, 0.02f) << "Q8_0 roundtrip max error: " << max_err;
}

TEST(QuantizeQ8_0, RoundtripNegative) {
    const int64_t n_blocks = 4;
    const int64_t n_elements = n_blocks * 32;
    std::vector<float> input(n_elements, -0.25f);
    std::vector<uint8_t> quantized(static_cast<size_t>(n_blocks) * 34);
    std::vector<float> output(n_elements);

    quantize_q8_0(input.data(), quantized.data(), n_blocks);
    dequantize_q8_0(quantized.data(), output.data(), n_blocks);

    for (int64_t i = 0; i < n_elements; ++i) {
        EXPECT_NEAR(output[i], -0.25f, 0.01f) << "at index " << i;
    }
}

// ============================================================================
// Q5_0 Dequantization tests
// ============================================================================

TEST(QuantizeQ5_0, DequantizeKnownBlock) {
    // Q5_0 block: 22 bytes = {uint16_t d, uint8_t qh[4], uint8_t qs[16]}
    // Construct a block where d=1.0, all qs=0, all qh=0
    // Then every element: q_low=0, q_high_bit=0, q_val=0, value=(0-16)*1.0 = -16
    alignas(2) uint8_t block[22] = {};
    uint16_t d_fp16 = test_fp32_to_fp16(1.0f);
    block[0] = static_cast<uint8_t>(d_fp16 & 0xFF);
    block[1] = static_cast<uint8_t>((d_fp16 >> 8) & 0xFF);
    // qh[4] = 0, qs[16] = 0 (already zero)

    float out[32] = {};
    dequantize_q5_0(block, out, 1);

    for (int i = 0; i < 32; ++i) {
        EXPECT_NEAR(out[i], -16.0f, 1e-5f) << "at index " << i;
    }
}

TEST(QuantizeQ5_0, DequantizeWithHighBits) {
    // Q5_0 block: {uint16_t d, uint8_t qh[4], uint8_t qs[16]}
    // Layout: bytes 0-1=d, 2-5=qh, 6-21=qs
    // qh is loaded as uint32_t (little-endian).
    // qh bit (2*i) → 5th bit of element i, qh bit (2*i+1) → 5th bit of element i+16
    // Set qh[0] = 0x01 → qh_val bit 0 set → element 0 gets 5th bit = 1
    // All qs = 0, so q_low = 0 for all elements
    // Element 0: q_val = 0 | (1<<4) = 16, value = (16-16)*1.0 = 0
    // Element 16: 5th bit from qh bit 1 = 0, value = (0-16)*1.0 = -16
    alignas(2) uint8_t block[22] = {};
    uint16_t d_fp16 = test_fp32_to_fp16(1.0f);
    block[0] = static_cast<uint8_t>(d_fp16 & 0xFF);
    block[1] = static_cast<uint8_t>((d_fp16 >> 8) & 0xFF);
    block[2] = 0x01; // qh[0]: bit 0 set → element 0 has 5th bit = 1

    float out[32] = {};
    dequantize_q5_0(block, out, 1);

    // Element 0: q_val=16, value=(16-16)*1.0 = 0
    EXPECT_NEAR(out[0], 0.0f, 1e-5f);
    // Element 16: q_val=0, value=(0-16)*1.0 = -16
    EXPECT_NEAR(out[16], -16.0f, 1e-5f);
    // Element 1: q_low=0, 5th bit from qh bit 2 = 0, value=-16
    EXPECT_NEAR(out[1], -16.0f, 1e-5f);
}

TEST(QuantizeQ5_0, DequantizeMultipleBlocks) {
    const int64_t n_blocks = 4;
    // Manually construct blocks with d=2.0, all qs=0, all qh=0
    // Expected: each element = (0-16)*2.0 = -32.0
    std::vector<uint8_t> data(static_cast<size_t>(n_blocks) * 22, 0);
    uint16_t d_fp16 = test_fp32_to_fp16(2.0f);
    for (int64_t b = 0; b < n_blocks; ++b) {
        data[static_cast<size_t>(b) * 22] = static_cast<uint8_t>(d_fp16 & 0xFF);
        data[static_cast<size_t>(b) * 22 + 1] = static_cast<uint8_t>((d_fp16 >> 8) & 0xFF);
    }

    const int64_t n_elements = n_blocks * 32;
    std::vector<float> output(n_elements);
    dequantize_q5_0(data.data(), output.data(), n_blocks);

    for (int64_t i = 0; i < n_elements; ++i) {
        EXPECT_NEAR(output[i], -32.0f, 1e-4f) << "at index " << i;
    }
}

// ============================================================================
// Q5_1 Dequantization tests
// ============================================================================

TEST(QuantizeQ5_1, DequantizeKnownBlock) {
    // Q5_1 block: 24 bytes = {uint16_t d, uint16_t m, uint8_t qh[4], uint8_t qs[16]}
    // d=1.0, m=16.0, all qs=0, all qh=0
    // value = q_val * d + m = 0 * 1.0 + 16.0 = 16.0
    alignas(2) uint8_t block[24] = {};
    uint16_t d_fp16 = test_fp32_to_fp16(1.0f);
    uint16_t m_fp16 = test_fp32_to_fp16(16.0f);
    block[0] = static_cast<uint8_t>(d_fp16 & 0xFF);
    block[1] = static_cast<uint8_t>((d_fp16 >> 8) & 0xFF);
    block[2] = static_cast<uint8_t>(m_fp16 & 0xFF);
    block[3] = static_cast<uint8_t>((m_fp16 >> 8) & 0xFF);

    float out[32] = {};
    dequantize_q5_1(block, out, 1);

    for (int i = 0; i < 32; ++i) {
        EXPECT_NEAR(out[i], 16.0f, 1e-4f) << "at index " << i;
    }
}

TEST(QuantizeQ5_1, DequantizeWithScaleAndOffset) {
    // Q5_1 block: {uint16_t d, uint16_t m, uint8_t qh[4], uint8_t qs[16]}
    // Layout: bytes 0-1=d, 2-3=m, 4-7=qh, 8-23=qs
    // d=0.5, m=8.0
    // qs[0] = 0xF0 → element 0: low nibble=0, element 16: high nibble=15
    // qh = 0 → no 5th bits
    // element 0: value = 0*0.5 + 8.0 = 8.0
    // element 16: value = 15*0.5 + 8.0 = 15.5
    alignas(2) uint8_t block[24] = {};
    uint16_t d_fp16 = test_fp32_to_fp16(0.5f);
    uint16_t m_fp16 = test_fp32_to_fp16(8.0f);
    block[0] = static_cast<uint8_t>(d_fp16 & 0xFF);
    block[1] = static_cast<uint8_t>((d_fp16 >> 8) & 0xFF);
    block[2] = static_cast<uint8_t>(m_fp16 & 0xFF);
    block[3] = static_cast<uint8_t>((m_fp16 >> 8) & 0xFF);
    // qh at bytes 4-7 (all zeros)
    // qs[0] at byte 8
    block[8] = 0xF0; // qs[0]: low nibble=0, high nibble=15

    float out[32] = {};
    dequantize_q5_1(block, out, 1);

    EXPECT_NEAR(out[0], 8.0f, 1e-4f);      // 0*0.5 + 8.0
    EXPECT_NEAR(out[16], 15.5f, 1e-4f);    // 15*0.5 + 8.0
}

// ============================================================================
// dequantize_dispatch tests
// ============================================================================

TEST(QuantizeDispatch, Q4_0Dispatch) {
    const int64_t n_blocks = 2;
    const int64_t n_elements = n_blocks * 32;
    std::vector<float> input(n_elements, 0.5f);
    std::vector<uint8_t> quantized(static_cast<size_t>(n_blocks) * 18);
    std::vector<float> output(n_elements);

    quantize_q4_0(input.data(), quantized.data(), n_blocks);
    dequantize_dispatch(DataType::Q4_0, quantized.data(), output.data(), n_elements);

    for (int64_t i = 0; i < n_elements; ++i) {
        EXPECT_NEAR(output[i], 0.5f, 0.15f) << "at index " << i;
    }
}

TEST(QuantizeDispatch, Q8_0Dispatch) {
    const int64_t n_blocks = 2;
    const int64_t n_elements = n_blocks * 32;
    std::vector<float> input(n_elements, 0.25f);
    std::vector<uint8_t> quantized(static_cast<size_t>(n_blocks) * 34);
    std::vector<float> output(n_elements);

    quantize_q8_0(input.data(), quantized.data(), n_blocks);
    dequantize_dispatch(DataType::Q8_0, quantized.data(), output.data(), n_elements);

    for (int64_t i = 0; i < n_elements; ++i) {
        EXPECT_NEAR(output[i], 0.25f, 0.01f) << "at index " << i;
    }
}

TEST(QuantizeDispatch, F32Dispatch) {
    const int64_t n_elements = 64;
    std::vector<float> input(n_elements);
    for (int64_t i = 0; i < n_elements; ++i) input[i] = static_cast<float>(i);
    std::vector<float> output(n_elements);

    dequantize_dispatch(DataType::F32, input.data(), output.data(), n_elements);

    for (int64_t i = 0; i < n_elements; ++i) {
        EXPECT_FLOAT_EQ(output[i], input[i]) << "at index " << i;
    }
}

TEST(QuantizeDispatch, UnsupportedTypeZerosOutput) {
    const int64_t n_elements = 32;
    std::vector<float> output(n_elements, 999.0f);
    uint8_t dummy = 0;

    // IQ1_S is truly unsupported — should produce zero output
    dequantize_dispatch(DataType::IQ1_S, &dummy, output.data(), n_elements);

    for (int64_t i = 0; i < n_elements; ++i) {
        EXPECT_FLOAT_EQ(output[i], 0.0f) << "at index " << i;
    }
}

// ============================================================================
// Fused Q4_0 dequantize + GEMM tests
// ============================================================================

TEST(QuantizeFused, Q4_0FusedMatchesSeparate) {
    // Compare fused Q4_0 dequantize+GEMM with separate dequantize then GEMM
    const int64_t M = 4;
    const int64_t N = 8;
    const int64_t K = 64; // Must be multiple of 32
    const int64_t blocks_per_row = K / 32;
    const int64_t n_a_blocks = M * blocks_per_row;

    // Generate random input and quantize
    std::vector<float> A_fp32(static_cast<size_t>(M * K));
    fill_random_range(A_fp32, -0.5f, 0.5f, 777);

    std::vector<uint8_t> A_q4(static_cast<size_t>(n_a_blocks) * 18);
    quantize_q4_0(A_fp32.data(), A_q4.data(), n_a_blocks);

    // Generate random B (transposed: [N,K] row-major)
    std::vector<float> B(static_cast<size_t>(N * K));
    fill_random_range(B, -1.0f, 1.0f, 888);

    // Method 1: Fused
    std::vector<float> C_fused(static_cast<size_t>(M * N), 0.0f);
    gemm_dequantize_q4_0_fused(A_q4.data(), n_a_blocks, B.data(), N, K, C_fused.data(), N);

    // Method 2: Separate dequantize + GEMM
    std::vector<float> A_deq(static_cast<size_t>(M * K));
    dequantize_q4_0(A_q4.data(), A_deq.data(), n_a_blocks);

    // Compute C[i,j] = sum_k A_deq[i,k] * B[j,k] (B transposed)
    std::vector<float> C_separate(static_cast<size_t>(M * N), 0.0f);
    for (int64_t i = 0; i < M; ++i) {
        for (int64_t j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int64_t k = 0; k < K; ++k) {
                sum += A_deq[static_cast<size_t>(i * K + k)] * B[static_cast<size_t>(j * K + k)];
            }
            C_separate[static_cast<size_t>(i * N + j)] = sum;
        }
    }

    // Results should be identical (same dequantized values, same math)
    for (int64_t i = 0; i < M * N; ++i) {
        EXPECT_NEAR(C_fused[i], C_separate[i], 1e-5f)
            << "at index " << i << " fused=" << C_fused[i] << " separate=" << C_separate[i];
    }
}

TEST(QuantizeFused, Q4_0FusedSingleRow) {
    // M=1 (GEMV-like)
    const int64_t M = 1;
    const int64_t N = 4;
    const int64_t K = 32; // Single block
    const int64_t n_a_blocks = M * (K / 32);

    std::vector<float> A_fp32 = {0.1f, -0.2f, 0.3f, -0.4f, 0.5f, -0.6f, 0.7f, -0.8f,
                                  0.1f, -0.2f, 0.3f, -0.4f, 0.5f, -0.6f, 0.7f, -0.8f,
                                  0.1f, -0.2f, 0.3f, -0.4f, 0.5f, -0.6f, 0.7f, -0.8f,
                                  0.1f, -0.2f, 0.3f, -0.4f, 0.5f, -0.6f, 0.7f, -0.8f};

    std::vector<uint8_t> A_q4(static_cast<size_t>(n_a_blocks) * 18);
    quantize_q4_0(A_fp32.data(), A_q4.data(), n_a_blocks);

    std::vector<float> B(static_cast<size_t>(N * K));
    fill_random_range(B, -1.0f, 1.0f, 999);

    std::vector<float> C_fused(static_cast<size_t>(M * N), 0.0f);
    gemm_dequantize_q4_0_fused(A_q4.data(), n_a_blocks, B.data(), N, K, C_fused.data(), N);

    // Verify manually: dequantize, then dot product
    std::vector<float> A_deq(static_cast<size_t>(K));
    dequantize_q4_0(A_q4.data(), A_deq.data(), n_a_blocks);

    for (int64_t j = 0; j < N; ++j) {
        float expected = 0.0f;
        for (int64_t k = 0; k < K; ++k) {
            expected += A_deq[k] * B[static_cast<size_t>(j * K + k)];
        }
        EXPECT_NEAR(C_fused[j], expected, 1e-4f) << "at column " << j;
    }
}

// ============================================================================
// Fused Q8_0 dequantize + GEMM tests
// ============================================================================

TEST(QuantizeFused, Q8_0FusedMatchesSeparate) {
    const int64_t M = 4;
    const int64_t N = 8;
    const int64_t K = 64;
    const int64_t blocks_per_row = K / 32;
    const int64_t n_a_blocks = M * blocks_per_row;

    std::vector<float> A_fp32(static_cast<size_t>(M * K));
    fill_random_range(A_fp32, -0.5f, 0.5f, 111);

    std::vector<uint8_t> A_q8(static_cast<size_t>(n_a_blocks) * 34);
    quantize_q8_0(A_fp32.data(), A_q8.data(), n_a_blocks);

    std::vector<float> B(static_cast<size_t>(N * K));
    fill_random_range(B, -1.0f, 1.0f, 222);

    // Fused
    std::vector<float> C_fused(static_cast<size_t>(M * N), 0.0f);
    gemm_dequantize_q8_0_fused(A_q8.data(), n_a_blocks, B.data(), N, K, C_fused.data(), N);

    // Separate
    std::vector<float> A_deq(static_cast<size_t>(M * K));
    dequantize_q8_0(A_q8.data(), A_deq.data(), n_a_blocks);

    std::vector<float> C_separate(static_cast<size_t>(M * N), 0.0f);
    for (int64_t i = 0; i < M; ++i) {
        for (int64_t j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int64_t k = 0; k < K; ++k) {
                sum += A_deq[static_cast<size_t>(i * K + k)] * B[static_cast<size_t>(j * K + k)];
            }
            C_separate[static_cast<size_t>(i * N + j)] = sum;
        }
    }

    for (int64_t i = 0; i < M * N; ++i) {
        EXPECT_NEAR(C_fused[i], C_separate[i], 1e-5f)
            << "at index " << i << " fused=" << C_fused[i] << " separate=" << C_separate[i];
    }
}

TEST(QuantizeFused, Q8_0FusedLargerMatrix) {
    // Larger matrix to test multi-row iteration
    const int64_t M = 8;
    const int64_t N = 16;
    const int64_t K = 96; // 3 blocks per row
    const int64_t blocks_per_row = K / 32;
    const int64_t n_a_blocks = M * blocks_per_row;

    std::vector<float> A_fp32(static_cast<size_t>(M * K));
    fill_random_range(A_fp32, -1.0f, 1.0f, 333);

    std::vector<uint8_t> A_q8(static_cast<size_t>(n_a_blocks) * 34);
    quantize_q8_0(A_fp32.data(), A_q8.data(), n_a_blocks);

    std::vector<float> B(static_cast<size_t>(N * K));
    fill_random_range(B, -1.0f, 1.0f, 444);

    std::vector<float> C_fused(static_cast<size_t>(M * N), 0.0f);
    gemm_dequantize_q8_0_fused(A_q8.data(), n_a_blocks, B.data(), N, K, C_fused.data(), N);

    // Verify with separate computation
    std::vector<float> A_deq(static_cast<size_t>(M * K));
    dequantize_q8_0(A_q8.data(), A_deq.data(), n_a_blocks);

    std::vector<float> C_separate(static_cast<size_t>(M * N), 0.0f);
    for (int64_t i = 0; i < M; ++i) {
        for (int64_t j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int64_t k = 0; k < K; ++k) {
                sum += A_deq[static_cast<size_t>(i * K + k)] * B[static_cast<size_t>(j * K + k)];
            }
            C_separate[static_cast<size_t>(i * N + j)] = sum;
        }
    }

    for (int64_t i = 0; i < M * N; ++i) {
        EXPECT_NEAR(C_fused[i], C_separate[i], 1e-4f) << "at index " << i;
    }
}

// ============================================================================
// Block size / element size consistency tests
// ============================================================================

TEST(QuantizeMeta, BlockSizesMatchTensor) {
    EXPECT_EQ(data_type_element_size(DataType::Q4_0), 18u);
    EXPECT_EQ(data_type_element_size(DataType::Q8_0), 34u);
    EXPECT_EQ(data_type_element_size(DataType::Q5_0), 22u);
    EXPECT_EQ(data_type_element_size(DataType::Q5_1), 24u);
}

TEST(QuantizeMeta, BlockElementCounts) {
    EXPECT_EQ(data_type_block_size(DataType::Q4_0), 32);
    EXPECT_EQ(data_type_block_size(DataType::Q8_0), 32);
    EXPECT_EQ(data_type_block_size(DataType::Q5_0), 32);
    EXPECT_EQ(data_type_block_size(DataType::Q5_1), 32);
}

// ============================================================================
// Edge case tests
// ============================================================================

TEST(QuantizeEdge, Q4_0EmptyBlocks) {
    // Zero blocks should not crash
    std::vector<float> out(32, 999.0f);
    dequantize_q4_0(nullptr, out.data(), 0);
    // Output should be unchanged (no blocks processed)
    EXPECT_FLOAT_EQ(out[0], 999.0f);
}

TEST(QuantizeEdge, Q8_0EmptyBlocks) {
    std::vector<float> out(32, 777.0f);
    dequantize_q8_0(nullptr, out.data(), 0);
    EXPECT_FLOAT_EQ(out[0], 777.0f);
}

TEST(QuantizeEdge, Q4_0AllSameValue) {
    const int64_t n_blocks = 2;
    const int64_t n_elements = n_blocks * 32;
    std::vector<float> input(n_elements, 0.42f);
    std::vector<uint8_t> quantized(static_cast<size_t>(n_blocks) * 18);
    std::vector<float> output(n_elements);

    quantize_q4_0(input.data(), quantized.data(), n_blocks);
    dequantize_q4_0(quantized.data(), output.data(), n_blocks);

    // All values should be approximately equal
    float first = output[0];
    for (int64_t i = 1; i < n_elements; ++i) {
        EXPECT_NEAR(output[i], first, 1e-6f) << "at index " << i;
    }
}

TEST(QuantizeEdge, Q8_0AllSameValue) {
    const int64_t n_blocks = 2;
    const int64_t n_elements = n_blocks * 32;
    std::vector<float> input(n_elements, -0.73f);
    std::vector<uint8_t> quantized(static_cast<size_t>(n_blocks) * 34);
    std::vector<float> output(n_elements);

    quantize_q8_0(input.data(), quantized.data(), n_blocks);
    dequantize_q8_0(quantized.data(), output.data(), n_blocks);

    float first = output[0];
    for (int64_t i = 1; i < n_elements; ++i) {
        EXPECT_NEAR(output[i], first, 1e-6f) << "at index " << i;
    }
}
