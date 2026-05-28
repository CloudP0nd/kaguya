#include "kaguya/kernels/quantize.h"
#include "kaguya/kernels/gemm.h"

#include <cstring>
#include <cmath>
#include <algorithm>
#include <limits>

namespace kaguya::kernels {

// ============================================================================
// FP16 ↔ FP32 conversion helpers
// ============================================================================

/// Simple FP16 → FP32 conversion (IEEE 754 half-precision)
static inline float fp16_to_fp32(uint16_t h) {
    uint32_t sign = (h >> 15) & 0x1;
    uint32_t exponent = (h >> 10) & 0x1F;
    uint32_t mantissa = h & 0x3FF;
    float result;
    if (exponent == 0) {
        if (mantissa == 0) {
            result = 0.0f;
        } else {
            // Denormalized
            result = std::ldexp(static_cast<float>(mantissa) / 1024.0f, -14);
        }
    } else if (exponent == 31) {
        result = mantissa ? std::numeric_limits<float>::quiet_NaN()
                          : std::numeric_limits<float>::infinity();
    } else {
        result = std::ldexp(1.0f + static_cast<float>(mantissa) / 1024.0f,
                            static_cast<int32_t>(exponent) - 15);
    }
    return sign ? -result : result;
}

/// Simple FP32 → FP16 conversion (IEEE 754 half-precision)
static inline uint16_t fp32_to_fp16(float f) {
    uint32_t x;
    std::memcpy(&x, &f, sizeof(x));
    uint32_t sign = (x >> 16) & 0x8000;
    int32_t exponent = static_cast<int32_t>((x >> 23) & 0xFF) - 127;
    uint32_t mantissa = x & 0x7FFFFF;
    if (exponent > 15) {
        return static_cast<uint16_t>(sign | 0x7C00); // overflow → inf
    }
    if (exponent < -14) {
        return static_cast<uint16_t>(sign); // underflow → 0
    }
    uint32_t new_exp = static_cast<uint32_t>(exponent + 15);
    uint32_t new_mant = mantissa >> 13;
    return static_cast<uint16_t>(sign | (new_exp << 10) | new_mant);
}

// ============================================================================
// Block structure definitions (ggml-compatible, packed)
// ============================================================================

struct Q4_0Block {
    uint16_t d;       // FP16 scale
    uint8_t qs[16];   // 4-bit quantized values (2 per byte)
};
static_assert(sizeof(Q4_0Block) == 18, "Q4_0 block must be 18 bytes");

struct Q8_0Block {
    uint16_t d;       // FP16 scale
    int8_t qs[32];    // 8-bit quantized values
};
static_assert(sizeof(Q8_0Block) == 34, "Q8_0 block must be 34 bytes");

struct Q5_0Block {
    uint16_t d;       // FP16 scale
    uint8_t qh[4];    // High bits (5th bit for each of 32 elements)
    uint8_t qs[16];   // Low 4 bits (2 per byte, same as Q4_0)
};
static_assert(sizeof(Q5_0Block) == 22, "Q5_0 block must be 22 bytes");

struct Q5_1Block {
    uint16_t d;       // FP16 scale
    uint16_t m;       // FP16 minimum (offset)
    uint8_t qh[4];    // High bits (5th bit for each of 32 elements)
    uint8_t qs[16];   // Low 4 bits (2 per byte, same as Q4_0)
};
static_assert(sizeof(Q5_1Block) == 24, "Q5_1 block must be 24 bytes");

// ============================================================================
// Dequantization kernels
// ============================================================================

void dequantize_q4_0_block(const void* block, float* out) {
    const auto* b = static_cast<const Q4_0Block*>(block);
    const float d = fp16_to_fp32(b->d);

    for (int i = 0; i < 16; ++i) {
        // Low nibble → element i, high nibble → element i+16
        const int q_low  = static_cast<int>(b->qs[i] & 0x0F);
        const int q_high = static_cast<int>((b->qs[i] >> 4) & 0x0F);
        out[i]      = static_cast<float>(q_low  - 8) * d;
        out[i + 16] = static_cast<float>(q_high - 8) * d;
    }
}

void dequantize_q4_0(const void* data, float* out, int64_t n_blocks) {
    const auto* blocks = static_cast<const Q4_0Block*>(data);
    for (int64_t b = 0; b < n_blocks; ++b) {
        dequantize_q4_0_block(&blocks[b], out + b * 32);
    }
}

void dequantize_q8_0(const void* data, float* out, int64_t n_blocks) {
    const auto* blocks = static_cast<const Q8_0Block*>(data);
    for (int64_t b = 0; b < n_blocks; ++b) {
        const float d = fp16_to_fp32(blocks[b].d);
        for (int i = 0; i < 32; ++i) {
            out[b * 32 + i] = static_cast<float>(blocks[b].qs[i]) * d;
        }
    }
}

void dequantize_q5_0(const void* data, float* out, int64_t n_blocks) {
    const auto* blocks = static_cast<const Q5_0Block*>(data);
    for (int64_t b = 0; b < n_blocks; ++b) {
        const float d = fp16_to_fp32(blocks[b].d);

        // Load qh as a 32-bit value (little-endian)
        uint32_t qh_val;
        std::memcpy(&qh_val, blocks[b].qh, 4);

        // Process pairs: qs[i] low nibble → element i, high nibble → element i+16
        // qh bit (2*i) → 5th bit of element i, qh bit (2*i+1) → 5th bit of element i+16
        for (int i = 0; i < 16; ++i) {
            // Element i: low nibble of qs[i] + 5th bit from qh
            const int q_low_0 = static_cast<int>(blocks[b].qs[i] & 0x0F);
            const int q_high_bit_0 = static_cast<int>((qh_val >> (2 * i)) & 1);
            const int q_val_0 = q_low_0 | (q_high_bit_0 << 4);
            out[b * 32 + i] = static_cast<float>(q_val_0 - 16) * d;

            // Element i+16: high nibble of qs[i] + 5th bit from qh
            const int q_low_1 = static_cast<int>((blocks[b].qs[i] >> 4) & 0x0F);
            const int q_high_bit_1 = static_cast<int>((qh_val >> (2 * i + 1)) & 1);
            const int q_val_1 = q_low_1 | (q_high_bit_1 << 4);
            out[b * 32 + i + 16] = static_cast<float>(q_val_1 - 16) * d;
        }
    }
}

void dequantize_q5_1(const void* data, float* out, int64_t n_blocks) {
    const auto* blocks = static_cast<const Q5_1Block*>(data);
    for (int64_t b = 0; b < n_blocks; ++b) {
        const float d = fp16_to_fp32(blocks[b].d);
        const float m = fp16_to_fp32(blocks[b].m);

        // Load qh as a 32-bit value (little-endian)
        uint32_t qh_val;
        std::memcpy(&qh_val, blocks[b].qh, 4);

        // Process pairs: same nibble layout as Q5_0
        for (int i = 0; i < 16; ++i) {
            // Element i: low nibble of qs[i] + 5th bit from qh
            const int q_low_0 = static_cast<int>(blocks[b].qs[i] & 0x0F);
            const int q_high_bit_0 = static_cast<int>((qh_val >> (2 * i)) & 1);
            const int q_val_0 = q_low_0 | (q_high_bit_0 << 4);
            out[b * 32 + i] = static_cast<float>(q_val_0) * d + m;

            // Element i+16: high nibble of qs[i] + 5th bit from qh
            const int q_low_1 = static_cast<int>((blocks[b].qs[i] >> 4) & 0x0F);
            const int q_high_bit_1 = static_cast<int>((qh_val >> (2 * i + 1)) & 1);
            const int q_val_1 = q_low_1 | (q_high_bit_1 << 4);
            out[b * 32 + i + 16] = static_cast<float>(q_val_1) * d + m;
        }
    }
}

void dequantize_dispatch(DataType dtype, const void* data, float* out, int64_t n_elements) {
    const int bs = data_type_block_size(dtype);
    const int64_t n_blocks = (n_elements + bs - 1) / bs;

    switch (dtype) {
        case DataType::Q4_0:
            dequantize_q4_0(data, out, n_blocks);
            break;
        case DataType::Q8_0:
            dequantize_q8_0(data, out, n_blocks);
            break;
        case DataType::Q5_0:
            dequantize_q5_0(data, out, n_blocks);
            break;
        case DataType::Q5_1:
            dequantize_q5_1(data, out, n_blocks);
            break;
        case DataType::F32:
            // No conversion needed
            std::memcpy(out, data, static_cast<size_t>(n_elements) * sizeof(float));
            break;
        default:
            // Unsupported quantization type — zero output
            std::memset(out, 0, static_cast<size_t>(n_elements) * sizeof(float));
            break;
    }
}

// ============================================================================
// Quantization kernels (FP32 → quantized, for testing)
// ============================================================================

void quantize_q4_0(const float* in, void* out, int64_t n_blocks) {
    auto* blocks = static_cast<Q4_0Block*>(out);

    for (int64_t b = 0; b < n_blocks; ++b) {
        const float* block_in = in + b * 32;

        // Find max absolute value in the block
        float max_abs = 0.0f;
        for (int i = 0; i < 32; ++i) {
            max_abs = std::max(max_abs, std::fabs(block_in[i]));
        }

        // Compute scale: d = max_abs / 8 (range is -8*d .. 7*d)
        const float d = max_abs / 8.0f;
        blocks[b].d = fp32_to_fp16(d);

        if (d == 0.0f) {
            std::memset(blocks[b].qs, 8, 16); // All zeros → nibble 8 (= 0 when -8)
            continue;
        }

        const float inv_d = 1.0f / d;
        for (int i = 0; i < 16; ++i) {
            // Quantize element i (low nibble)
            int q_low = static_cast<int>(std::round(block_in[i] * inv_d)) + 8;
            q_low = std::clamp(q_low, 0, 15);

            // Quantize element i+16 (high nibble)
            int q_high = static_cast<int>(std::round(block_in[i + 16] * inv_d)) + 8;
            q_high = std::clamp(q_high, 0, 15);

            blocks[b].qs[i] = static_cast<uint8_t>(q_low | (q_high << 4));
        }
    }
}

void quantize_q8_0(const float* in, void* out, int64_t n_blocks) {
    auto* blocks = static_cast<Q8_0Block*>(out);

    for (int64_t b = 0; b < n_blocks; ++b) {
        const float* block_in = in + b * 32;

        // Find max absolute value in the block
        float max_abs = 0.0f;
        for (int i = 0; i < 32; ++i) {
            max_abs = std::max(max_abs, std::fabs(block_in[i]));
        }

        // Compute scale: d = max_abs / 128 (range is -128*d .. 127*d)
        const float d = max_abs / 128.0f;
        blocks[b].d = fp32_to_fp16(d);

        if (d == 0.0f) {
            std::memset(blocks[b].qs, 0, 32);
            continue;
        }

        const float inv_d = 1.0f / d;
        for (int i = 0; i < 32; ++i) {
            int q = static_cast<int>(std::round(block_in[i] * inv_d));
            q = std::clamp(q, -128, 127);
            blocks[b].qs[i] = static_cast<int8_t>(q);
        }
    }
}

// ============================================================================
// Fused dequantize + GEMM kernels
// ============================================================================

/// Fused Q4_0 dequantize + GEMM: C[M,N] = A_deq[M,K] * B[N,K]^T
/// B is stored row-major as [N,K] (transposed).
/// Process one row of A at a time: dequantize row → dot product with each row of B.
void gemm_dequantize_q4_0_fused(
    const void* a_quantized, int64_t n_a_blocks,
    const float* B, int64_t N, int64_t K,
    float* C, int64_t ldc)
{
    const int64_t block_size = 32;
    const int64_t blocks_per_row = K / block_size;
    const int64_t M = (blocks_per_row > 0) ? n_a_blocks / blocks_per_row : 0;

    // Temporary buffer for one dequantized row
    std::vector<float> row_buf(static_cast<size_t>(K));

    for (int64_t i = 0; i < M; ++i) {
        // Dequantize row i of A
        const auto* row_blocks = static_cast<const Q4_0Block*>(a_quantized) + i * blocks_per_row;
        dequantize_q4_0(row_blocks, row_buf.data(), blocks_per_row);

        // Compute C[i, j] = dot(row_buf, B[j, :]) for all j
        for (int64_t j = 0; j < N; ++j) {
            float sum = 0.0f;
            const float* b_row = B + j * K;
            for (int64_t k = 0; k < K; ++k) {
                sum += row_buf[static_cast<size_t>(k)] * b_row[static_cast<size_t>(k)];
            }
            C[i * ldc + j] = sum;
        }
    }
}

/// Fused Q8_0 dequantize + GEMM: same layout as Q4_0 version
void gemm_dequantize_q8_0_fused(
    const void* a_quantized, int64_t n_a_blocks,
    const float* B, int64_t N, int64_t K,
    float* C, int64_t ldc)
{
    const int64_t block_size = 32;
    const int64_t blocks_per_row = K / block_size;
    const int64_t M = (blocks_per_row > 0) ? n_a_blocks / blocks_per_row : 0;

    // Temporary buffer for one dequantized row
    std::vector<float> row_buf(static_cast<size_t>(K));

    for (int64_t i = 0; i < M; ++i) {
        // Dequantize row i of A
        const auto* row_blocks = static_cast<const Q8_0Block*>(a_quantized) + i * blocks_per_row;
        dequantize_q8_0(row_blocks, row_buf.data(), blocks_per_row);

        // Compute C[i, j] = dot(row_buf, B[j, :]) for all j
        for (int64_t j = 0; j < N; ++j) {
            float sum = 0.0f;
            const float* b_row = B + j * K;
            for (int64_t k = 0; k < K; ++k) {
                sum += row_buf[static_cast<size_t>(k)] * b_row[static_cast<size_t>(k)];
            }
            C[i * ldc + j] = sum;
        }
    }
}

} // namespace kaguya::kernels
