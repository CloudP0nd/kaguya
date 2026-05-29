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
// K-quant block structure definitions (ggml-compatible, QK_K = 256)
// ============================================================================

static constexpr int QK_K = 256;       // Number of elements per K-quant super-block
static constexpr int K_SCALE_SIZE = 12; // Size of packed scales array for Q4_K/Q5_K

// Q2_K: 2-bit quantization with per-group scale and min
// weight = d * scale_4bit * q_2bit - dmin * min_4bit
// 16 groups of 16 elements, scales and mins quantized with 4 bits
// Effectively 2.625 bpw
struct Q2_KBlock {
    uint8_t scales[QK_K / 16]; // 16 bytes: scales and mins, quantized with 4 bits
    uint8_t qs[QK_K / 4];      // 64 bytes: 2-bit quantized values (4 per byte)
    uint16_t d;                 // FP16 super-block scale for quantized scales
    uint16_t dmin;              // FP16 super-block scale for quantized mins
};
static_assert(sizeof(Q2_KBlock) == 84, "Q2_K block must be 84 bytes");

// Q3_K: 3-bit quantization with per-group scale (no min)
// weight = d * (scale_6bit - 32) * (q_2bit - hmask_offset)
// 16 groups of 16 elements, scales quantized with 6 bits
// Effectively 3.4375 bpw
struct Q3_KBlock {
    uint8_t hmask[QK_K / 8];   // 32 bytes: sign mask (1 bit per element)
    uint8_t qs[QK_K / 4];      // 64 bytes: low 2 bits per element
    uint8_t scales[12];         // 12 bytes: scales, quantized with 6 bits
    uint16_t d;                 // FP16 super-block scale
};
static_assert(sizeof(Q3_KBlock) == 110, "Q3_K block must be 110 bytes");

// Q4_K: 4-bit quantization with per-group scale and min
// weight = d * scale_6bit * q_4bit - dmin * min_6bit
// 8 groups of 32 elements, scales/mins quantized with 6 bits
// Effectively 4.5 bpw
struct Q4_KBlock {
    uint16_t d;                 // FP16 super-block scale for quantized scales
    uint16_t dmin;              // FP16 super-block scale for quantized mins
    uint8_t scales[K_SCALE_SIZE]; // 12 bytes: scales and mins, quantized with 6 bits
    uint8_t qs[QK_K / 2];      // 128 bytes: 4-bit quantized values
};
static_assert(sizeof(Q4_KBlock) == 144, "Q4_K block must be 144 bytes");

// Q5_K: 5-bit quantization with per-group scale and min
// weight = d * scale_6bit * q_5bit - dmin * min_6bit
// 8 groups of 32 elements, scales/mins quantized with 6 bits
// Effectively 5.5 bpw
struct Q5_KBlock {
    uint16_t d;                 // FP16 super-block scale for quantized scales
    uint16_t dmin;              // FP16 super-block scale for quantized mins
    uint8_t scales[K_SCALE_SIZE]; // 12 bytes: scales and mins, quantized with 6 bits
    uint8_t qh[QK_K / 8];      // 32 bytes: 5th bit for each element
    uint8_t qs[QK_K / 2];      // 128 bytes: low 4 bits per element
};
static_assert(sizeof(Q5_KBlock) == 176, "Q5_K block must be 176 bytes");

// Q6_K: 6-bit quantization with per-group scale (no min)
// weight = d * scale_8bit * (q_6bit - 32)
// 16 groups of 16 elements, scales stored as int8_t
// Effectively 6.5625 bpw
struct Q6_KBlock {
    uint8_t ql[QK_K / 2];      // 128 bytes: lower 4 bits per element
    uint8_t qh[QK_K / 4];      // 64 bytes: upper 2 bits per element
    int8_t scales[QK_K / 16];  // 16 bytes: per-group scales, quantized with 8 bits
    uint16_t d;                 // FP16 super-block scale
};
static_assert(sizeof(Q6_KBlock) == 210, "Q6_K block must be 210 bytes");

// Q8_K: 8-bit quantization with per-block scale (no min)
// weight = d * q_8bit
// Intermediate quantization type used for dot products
struct Q8_KBlock {
    float d;                    // FP32 super-block scale
    int8_t qs[QK_K];           // 256 bytes: 8-bit quantized values
    int16_t bsums[QK_K / 16];  // 32 bytes: sum of quants in groups of 16
};
static_assert(sizeof(Q8_KBlock) == 292, "Q8_K block must be 292 bytes");

// ============================================================================
// K-quant helper functions
// ============================================================================

/// Decode scale and min from the packed scales array for Q4_K/Q5_K.
/// The 12-byte scales array encodes 8 scale+min pairs using 6 bits each.
/// For j < 4:  scale = scales[j] & 63, min = scales[j+4] & 63
/// For j >= 4: upper bits are packed from earlier bytes
static inline void get_scale_min_k4(int j, const uint8_t* q, uint8_t* d, uint8_t* m) {
    if (j < 4) {
        *d = q[j] & 63;
        *m = q[j + 4] & 63;
    } else {
        *d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        *m = (q[j + 4] >> 4) | ((q[j - 0] >> 6) << 4);
    }
}

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

// ============================================================================
// K-quant dequantization kernels
// ============================================================================

void dequantize_q2_k_block(const void* block, float* out) {
    const auto* b = static_cast<const Q2_KBlock*>(block);
    const float d   = fp16_to_fp32(b->d);
    const float min = fp16_to_fp32(b->dmin);
    const uint8_t* q = b->qs;

    int is = 0;
    // Process 2 super-groups of 128 elements each
    for (int n = 0; n < QK_K; n += 128) {
        int shift = 0;
        // 4 sub-groups per super-group, each producing 32 elements
        for (int j = 0; j < 4; ++j) {
            // First 16 elements of sub-group
            uint8_t sc = b->scales[is++];
            float dl = d * (sc & 0xF);
            float ml = min * (sc >> 4);
            for (int l = 0; l < 16; ++l) {
                out[n + j * 32 + l] = dl * static_cast<float>(static_cast<int8_t>((q[l] >> shift) & 3)) - ml;
            }

            // Second 16 elements of sub-group
            sc = b->scales[is++];
            dl = d * (sc & 0xF);
            ml = min * (sc >> 4);
            for (int l = 0; l < 16; ++l) {
                out[n + j * 32 + 16 + l] = dl * static_cast<float>(static_cast<int8_t>((q[l + 16] >> shift) & 3)) - ml;
            }

            shift += 2;
        }
        q += 32;
    }
}

void dequantize_q2_k(const void* data, float* out, int64_t n_blocks) {
    const auto* blocks = static_cast<const Q2_KBlock*>(data);
    for (int64_t b = 0; b < n_blocks; ++b) {
        dequantize_q2_k_block(&blocks[b], out + b * QK_K);
    }
}

void dequantize_q3_k_block(const void* block, float* out) {
    const auto* b = static_cast<const Q3_KBlock*>(block);
    const float d_all = fp16_to_fp32(b->d);
    const uint8_t* q = b->qs;
    const uint8_t* hm = b->hmask;

    // Unpack the 12-byte scales into 16 6-bit values
    // The packing uses 3 6-bit values per pair of bytes
    const uint32_t kmask1 = 0x03030303;
    const uint32_t kmask2 = 0x0f0f0f0f;
    uint32_t aux[4];
    const int8_t* scales = reinterpret_cast<const int8_t*>(aux);

    std::memcpy(aux, b->scales, 12);
    uint32_t tmp = aux[2];
    aux[2] = ((aux[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
    aux[3] = ((aux[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
    aux[0] = (aux[0] & kmask2) | (((tmp >> 0) & kmask1) << 4);
    aux[1] = (aux[1] & kmask2) | (((tmp >> 2) & kmask1) << 4);

    int is = 0;
    uint8_t m = 1;
    // Process 2 super-groups of 128 elements each
    for (int n = 0; n < QK_K; n += 128) {
        int shift = 0;
        // 4 sub-groups per super-group, each producing 32 elements
        for (int j = 0; j < 4; ++j) {
            // First 16 elements of sub-group
            float dl = d_all * (scales[is++] - 32);
            for (int l = 0; l < 16; ++l) {
                out[n + j * 32 + l] = dl * static_cast<float>(
                    static_cast<int8_t>((q[l] >> shift) & 3) - ((hm[l] & m) ? 0 : 4));
            }

            // Second 16 elements of sub-group
            dl = d_all * (scales[is++] - 32);
            for (int l = 0; l < 16; ++l) {
                out[n + j * 32 + 16 + l] = dl * static_cast<float>(
                    static_cast<int8_t>((q[l + 16] >> shift) & 3) - ((hm[l + 16] & m) ? 0 : 4));
            }

            shift += 2;
            m <<= 1;
        }
        q += 32;
    }
}

void dequantize_q3_k(const void* data, float* out, int64_t n_blocks) {
    const auto* blocks = static_cast<const Q3_KBlock*>(data);
    for (int64_t b = 0; b < n_blocks; ++b) {
        dequantize_q3_k_block(&blocks[b], out + b * QK_K);
    }
}

void dequantize_q4_k_block(const void* block, float* out) {
    const auto* b = static_cast<const Q4_KBlock*>(block);
    const uint8_t* q = b->qs;
    const float d   = fp16_to_fp32(b->d);
    const float min = fp16_to_fp32(b->dmin);

    int is = 0;
    uint8_t sc, m;
    // 8 groups of 32 elements each (256 / 32 = 8 groups)
    for (int j = 0; j < QK_K; j += 64) {
        get_scale_min_k4(is + 0, b->scales, &sc, &m);
        const float d1 = d * sc; const float m1 = min * m;
        get_scale_min_k4(is + 1, b->scales, &sc, &m);
        const float d2 = d * sc; const float m2 = min * m;
        for (int l = 0; l < 32; ++l) out[j + l]      = d1 * (q[l] & 0xF) - m1;
        for (int l = 0; l < 32; ++l) out[j + 32 + l]  = d2 * (q[l] >> 4)  - m2;
        q += 32;
        is += 2;
    }
}

void dequantize_q4_k(const void* data, float* out, int64_t n_blocks) {
    const auto* blocks = static_cast<const Q4_KBlock*>(data);
    for (int64_t b = 0; b < n_blocks; ++b) {
        dequantize_q4_k_block(&blocks[b], out + b * QK_K);
    }
}

void dequantize_q5_k_block(const void* block, float* out) {
    const auto* b = static_cast<const Q5_KBlock*>(block);
    const uint8_t* ql = b->qs;
    const uint8_t* qh = b->qh;
    const float d   = fp16_to_fp32(b->d);
    const float min = fp16_to_fp32(b->dmin);

    int is = 0;
    uint8_t sc, m;
    uint8_t u1 = 1, u2 = 2;
    // 8 groups of 32 elements each (256 / 32 = 8 groups)
    for (int j = 0; j < QK_K; j += 64) {
        get_scale_min_k4(is + 0, b->scales, &sc, &m);
        const float d1 = d * sc; const float m1 = min * m;
        get_scale_min_k4(is + 1, b->scales, &sc, &m);
        const float d2 = d * sc; const float m2 = min * m;
        for (int l = 0; l < 32; ++l) {
            out[j + l]      = d1 * static_cast<float>((ql[l] & 0xF) + (qh[l] & u1 ? 16 : 0)) - m1;
        }
        for (int l = 0; l < 32; ++l) {
            out[j + 32 + l]  = d2 * static_cast<float>((ql[l] >> 4) + (qh[l] & u2 ? 16 : 0)) - m2;
        }
        ql += 32;
        is += 2;
        u1 <<= 2;
        u2 <<= 2;
    }
}

void dequantize_q5_k(const void* data, float* out, int64_t n_blocks) {
    const auto* blocks = static_cast<const Q5_KBlock*>(data);
    for (int64_t b = 0; b < n_blocks; ++b) {
        dequantize_q5_k_block(&blocks[b], out + b * QK_K);
    }
}

void dequantize_q6_k_block(const void* block, float* out) {
    const auto* b = static_cast<const Q6_KBlock*>(block);
    const float d = fp16_to_fp32(b->d);
    const uint8_t* ql = b->ql;
    const uint8_t* qh = b->qh;
    const int8_t* sc = b->scales;

    // Process 2 super-groups of 128 elements each
    for (int n = 0; n < QK_K; n += 128) {
        for (int l = 0; l < 32; ++l) {
            const int is_idx = l / 16;
            const int8_t q1 = static_cast<int8_t>((ql[l]      & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
            const int8_t q2 = static_cast<int8_t>((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
            const int8_t q3 = static_cast<int8_t>((ql[l]      >> 4)  | (((qh[l] >> 4) & 3) << 4)) - 32;
            const int8_t q4 = static_cast<int8_t>((ql[l + 32] >> 4)  | (((qh[l] >> 6) & 3) << 4)) - 32;
            out[l]      = d * sc[is_idx + 0] * q1;
            out[l + 32] = d * sc[is_idx + 2] * q2;
            out[l + 64] = d * sc[is_idx + 4] * q3;
            out[l + 96] = d * sc[is_idx + 6] * q4;
        }
        out += 128;
        ql += 64;
        qh += 32;
        sc += 8;
    }
}

void dequantize_q6_k(const void* data, float* out, int64_t n_blocks) {
    const auto* blocks = static_cast<const Q6_KBlock*>(data);
    for (int64_t b = 0; b < n_blocks; ++b) {
        dequantize_q6_k_block(&blocks[b], out + b * QK_K);
    }
}

void dequantize_q8_k_block(const void* block, float* out) {
    const auto* b = static_cast<const Q8_KBlock*>(block);
    for (int j = 0; j < QK_K; ++j) {
        out[j] = b->d * static_cast<float>(b->qs[j]);
    }
}

void dequantize_q8_k(const void* data, float* out, int64_t n_blocks) {
    const auto* blocks = static_cast<const Q8_KBlock*>(data);
    for (int64_t b = 0; b < n_blocks; ++b) {
        dequantize_q8_k_block(&blocks[b], out + b * QK_K);
    }
}

void dequantize_bf16(const void* data, float* out, int64_t n_elements) {
    const auto* src = static_cast<const uint16_t*>(data);
    for (int64_t i = 0; i < n_elements; ++i) {
        // BF16 → FP32: just shift left by 16 bits
        uint32_t bits = static_cast<uint32_t>(src[i]) << 16;
        std::memcpy(&out[i], &bits, sizeof(float));
    }
}

void dequantize_f16(const void* data, float* out, int64_t n_elements) {
    const auto* src = static_cast<const uint16_t*>(data);
    for (int64_t i = 0; i < n_elements; ++i) {
        out[i] = fp16_to_fp32(src[i]);
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
        case DataType::BF16:
            dequantize_bf16(data, out, n_elements);
            break;
        case DataType::F16:
            dequantize_f16(data, out, n_elements);
            break;
        case DataType::F32:
            // No conversion needed
            std::memcpy(out, data, static_cast<size_t>(n_elements) * sizeof(float));
            break;
        // ---- K-quant types (256 elements per block) ----
        case DataType::Q2_K: {
            const int64_t full_blocks = n_elements / bs;
            const int64_t remaining = n_elements % bs;
            if (full_blocks > 0) {
                dequantize_q2_k(data, out, full_blocks);
            }
            if (remaining > 0) {
                // Partial last block: dequantize into temp buffer, copy only needed elements
                float tmp[QK_K];
                const auto* bdata = static_cast<const uint8_t*>(data);
                dequantize_q2_k_block(bdata + full_blocks * sizeof(Q2_KBlock), tmp);
                std::memcpy(out + full_blocks * bs, tmp, static_cast<size_t>(remaining) * sizeof(float));
            }
            break;
        }
        case DataType::Q3_K: {
            const int64_t full_blocks = n_elements / bs;
            const int64_t remaining = n_elements % bs;
            if (full_blocks > 0) {
                dequantize_q3_k(data, out, full_blocks);
            }
            if (remaining > 0) {
                float tmp[QK_K];
                const auto* bdata = static_cast<const uint8_t*>(data);
                dequantize_q3_k_block(bdata + full_blocks * sizeof(Q3_KBlock), tmp);
                std::memcpy(out + full_blocks * bs, tmp, static_cast<size_t>(remaining) * sizeof(float));
            }
            break;
        }
        case DataType::Q4_K: {
            const int64_t full_blocks = n_elements / bs;
            const int64_t remaining = n_elements % bs;
            if (full_blocks > 0) {
                dequantize_q4_k(data, out, full_blocks);
            }
            if (remaining > 0) {
                float tmp[QK_K];
                const auto* bdata = static_cast<const uint8_t*>(data);
                dequantize_q4_k_block(bdata + full_blocks * sizeof(Q4_KBlock), tmp);
                std::memcpy(out + full_blocks * bs, tmp, static_cast<size_t>(remaining) * sizeof(float));
            }
            break;
        }
        case DataType::Q5_K: {
            const int64_t full_blocks = n_elements / bs;
            const int64_t remaining = n_elements % bs;
            if (full_blocks > 0) {
                dequantize_q5_k(data, out, full_blocks);
            }
            if (remaining > 0) {
                float tmp[QK_K];
                const auto* bdata = static_cast<const uint8_t*>(data);
                dequantize_q5_k_block(bdata + full_blocks * sizeof(Q5_KBlock), tmp);
                std::memcpy(out + full_blocks * bs, tmp, static_cast<size_t>(remaining) * sizeof(float));
            }
            break;
        }
        case DataType::Q6_K: {
            const int64_t full_blocks = n_elements / bs;
            const int64_t remaining = n_elements % bs;
            if (full_blocks > 0) {
                dequantize_q6_k(data, out, full_blocks);
            }
            if (remaining > 0) {
                float tmp[QK_K];
                const auto* bdata = static_cast<const uint8_t*>(data);
                dequantize_q6_k_block(bdata + full_blocks * sizeof(Q6_KBlock), tmp);
                std::memcpy(out + full_blocks * bs, tmp, static_cast<size_t>(remaining) * sizeof(float));
            }
            break;
        }
        case DataType::Q8_K: {
            const int64_t full_blocks = n_elements / bs;
            const int64_t remaining = n_elements % bs;
            if (full_blocks > 0) {
                dequantize_q8_k(data, out, full_blocks);
            }
            if (remaining > 0) {
                float tmp[QK_K];
                const auto* bdata = static_cast<const uint8_t*>(data);
                dequantize_q8_k_block(bdata + full_blocks * sizeof(Q8_KBlock), tmp);
                std::memcpy(out + full_blocks * bs, tmp, static_cast<size_t>(remaining) * sizeof(float));
            }
            break;
        }
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
