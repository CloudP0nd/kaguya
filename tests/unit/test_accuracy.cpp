#include <gtest/gtest.h>
#include <vector>
#include <cmath>
#include <cstring>
#include <random>
#include <cstdint>
#include <algorithm>
#include <limits>

#include "kaguya/kernels/quantize.h"
#include "kaguya/kernels/special_ops.h"
#include "kaguya/kernels/gemm.h"
#include "kaguya/attention.h"
#include "kaguya/model.h"
#include "kaguya/pipeline.h"
#include "kaguya/sampling.h"
#include "kaguya/kv_cache.h"

using namespace kaguya;
using namespace kaguya::kernels;

// ============================================================================
// Helper utilities
// ============================================================================

static void fill_random(std::vector<float>& v, float lo, float hi, unsigned seed = 42) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(lo, hi);
    for (auto& x : v) x = dist(rng);
}

static float max_abs_error(const std::vector<float>& a, const std::vector<float>& b) {
    float max_err = 0.0f;
    for (size_t i = 0; i < a.size(); ++i) {
        max_err = std::max(max_err, std::fabs(a[i] - b[i]));
    }
    return max_err;
}

static float mean_abs_error(const std::vector<float>& a, const std::vector<float>& b) {
    double sum = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        sum += std::fabs(static_cast<double>(a[i]) - static_cast<double>(b[i]));
    }
    return static_cast<float>(sum / static_cast<double>(a.size()));
}

/// Simple FP32 → FP16 conversion
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

/// Local Q5_0 quantization (for testing)
static void quantize_q5_0_local(const float* in, void* out, int64_t n_blocks) {
    // Q5_0 block: 22 bytes = {uint16_t d, uint8_t qh[4], uint8_t qs[16]}
    auto* blocks = static_cast<uint8_t*>(out);

    for (int64_t b = 0; b < n_blocks; ++b) {
        const float* block_in = in + b * 32;

        // Find max absolute value
        float max_abs = 0.0f;
        for (int i = 0; i < 32; ++i) {
            max_abs = std::max(max_abs, std::fabs(block_in[i]));
        }

        // Scale: d = max_abs / 16 (5-bit unsigned range 0-31, offset by 16)
        const float d = max_abs / 16.0f;
        uint16_t d_fp16 = test_fp32_to_fp16(d);
        blocks[b * 22] = static_cast<uint8_t>(d_fp16 & 0xFF);
        blocks[b * 22 + 1] = static_cast<uint8_t>((d_fp16 >> 8) & 0xFF);

        // qh[4] = 0 initially
        blocks[b * 22 + 2] = 0;
        blocks[b * 22 + 3] = 0;
        blocks[b * 22 + 4] = 0;
        blocks[b * 22 + 5] = 0;

        if (d == 0.0f) {
            // All zeros → q_val = 16 (=0 when -16 offset), qs nibbles = 0
            for (int i = 0; i < 16; ++i) {
                blocks[b * 22 + 6 + i] = 0;
            }
            continue;
        }

        const float inv_d = 1.0f / d;
        uint32_t qh_val = 0;

        for (int i = 0; i < 16; ++i) {
            // Quantize element i (low nibble + potential 5th bit)
            int q0 = static_cast<int>(std::round(block_in[i] * inv_d)) + 16;
            q0 = std::clamp(q0, 0, 31);
            int low0 = q0 & 0x0F;
            int high0 = (q0 >> 4) & 1;
            if (high0) qh_val |= (1u << (2 * i));

            // Quantize element i+16 (high nibble + potential 5th bit)
            int q1 = static_cast<int>(std::round(block_in[i + 16] * inv_d)) + 16;
            q1 = std::clamp(q1, 0, 31);
            int low1 = q1 & 0x0F;
            int high1 = (q1 >> 4) & 1;
            if (high1) qh_val |= (1u << (2 * i + 1));

            blocks[b * 22 + 6 + i] = static_cast<uint8_t>(low0 | (low1 << 4));
        }

        // Store qh
        blocks[b * 22 + 2] = static_cast<uint8_t>(qh_val & 0xFF);
        blocks[b * 22 + 3] = static_cast<uint8_t>((qh_val >> 8) & 0xFF);
        blocks[b * 22 + 4] = static_cast<uint8_t>((qh_val >> 16) & 0xFF);
        blocks[b * 22 + 5] = static_cast<uint8_t>((qh_val >> 24) & 0xFF);
    }
}

/// Local Q5_1 quantization (for testing)
static void quantize_q5_1_local(const float* in, void* out, int64_t n_blocks) {
    // Q5_1 block: 24 bytes = {uint16_t d, uint16_t m, uint8_t qh[4], uint8_t qs[16]}
    auto* blocks = static_cast<uint8_t*>(out);

    for (int64_t b = 0; b < n_blocks; ++b) {
        const float* block_in = in + b * 32;

        // Find min and max
        float min_val = block_in[0];
        float max_val = block_in[0];
        for (int i = 1; i < 32; ++i) {
            min_val = std::min(min_val, block_in[i]);
            max_val = std::max(max_val, block_in[i]);
        }

        // Scale: d = (max - min) / 31, offset m = min
        const float d = (max_val - min_val) / 31.0f;
        const float m = min_val;
        uint16_t d_fp16 = test_fp32_to_fp16(d);
        uint16_t m_fp16 = test_fp32_to_fp16(m);
        blocks[b * 24] = static_cast<uint8_t>(d_fp16 & 0xFF);
        blocks[b * 24 + 1] = static_cast<uint8_t>((d_fp16 >> 8) & 0xFF);
        blocks[b * 24 + 2] = static_cast<uint8_t>(m_fp16 & 0xFF);
        blocks[b * 24 + 3] = static_cast<uint8_t>((m_fp16 >> 8) & 0xFF);

        // qh[4] = 0 initially
        blocks[b * 24 + 4] = 0;
        blocks[b * 24 + 5] = 0;
        blocks[b * 24 + 6] = 0;
        blocks[b * 24 + 7] = 0;

        if (d == 0.0f) {
            // All values are the same as min
            for (int i = 0; i < 16; ++i) {
                blocks[b * 24 + 8 + i] = 0;
            }
            continue;
        }

        const float inv_d = 1.0f / d;
        uint32_t qh_val = 0;

        for (int i = 0; i < 16; ++i) {
            // Quantize element i: q_val = round((x - m) / d), clamped to [0, 31]
            int q0 = static_cast<int>(std::round((block_in[i] - m) * inv_d));
            q0 = std::clamp(q0, 0, 31);
            int low0 = q0 & 0x0F;
            int high0 = (q0 >> 4) & 1;
            if (high0) qh_val |= (1u << (2 * i));

            // Quantize element i+16
            int q1 = static_cast<int>(std::round((block_in[i + 16] - m) * inv_d));
            q1 = std::clamp(q1, 0, 31);
            int low1 = q1 & 0x0F;
            int high1 = (q1 >> 4) & 1;
            if (high1) qh_val |= (1u << (2 * i + 1));

            blocks[b * 24 + 8 + i] = static_cast<uint8_t>(low0 | (low1 << 4));
        }

        // Store qh
        blocks[b * 24 + 4] = static_cast<uint8_t>(qh_val & 0xFF);
        blocks[b * 24 + 5] = static_cast<uint8_t>((qh_val >> 8) & 0xFF);
        blocks[b * 24 + 6] = static_cast<uint8_t>((qh_val >> 16) & 0xFF);
        blocks[b * 24 + 7] = static_cast<uint8_t>((qh_val >> 24) & 0xFF);
    }
}

/// Local BF16 → FP32 conversion
static float bf16_to_fp32(uint16_t h) {
    uint32_t bits = static_cast<uint32_t>(h) << 16;
    float result;
    std::memcpy(&result, &bits, sizeof(float));
    return result;
}

/// Local FP32 → BF16 conversion (round to nearest even)
static uint16_t fp32_to_bf16(float f) {
    uint32_t x;
    std::memcpy(&x, &f, sizeof(x));
    // Round to nearest even for the lower 16 bits
    uint32_t lsb = (x >> 16) & 1;
    uint32_t rounding = x & 0xFFFF;
    uint32_t bias = 0x7FFF + lsb; // round to nearest even
    uint32_t result = (x >> 16) + (rounding + bias > 0xFFFF ? 1 : 0);
    return static_cast<uint16_t>(result & 0xFFFF);
}

// ============================================================================
// 1. Q4_0 Dequantize Error Bound
// ============================================================================

TEST(Accuracy, Q4_0DequantizeErrorBound) {
    const int64_t n_blocks = 64;
    const int64_t n_elements = n_blocks * 32;
    std::vector<float> input(n_elements);
    fill_random(input, -0.5f, 0.5f, 123);

    std::vector<uint8_t> quantized(static_cast<size_t>(n_blocks) * 18);
    std::vector<float> output(n_elements);

    quantize_q4_0(input.data(), quantized.data(), n_blocks);
    dequantize_q4_0(quantized.data(), output.data(), n_blocks);

    float max_err = max_abs_error(input, output);
    float mean_err = mean_abs_error(input, output);

    EXPECT_LT(max_err, 0.1f) << "Q4_0 max absolute error: " << max_err;
    EXPECT_LT(mean_err, 0.05f) << "Q4_0 mean absolute error: " << mean_err;
}

// ============================================================================
// 2. Q8_0 Dequantize Error Bound
// ============================================================================

TEST(Accuracy, Q8_0DequantizeErrorBound) {
    const int64_t n_blocks = 64;
    const int64_t n_elements = n_blocks * 32;
    std::vector<float> input(n_elements);
    fill_random(input, -1.0f, 1.0f, 456);

    std::vector<uint8_t> quantized(static_cast<size_t>(n_blocks) * 34);
    std::vector<float> output(n_elements);

    quantize_q8_0(input.data(), quantized.data(), n_blocks);
    dequantize_q8_0(quantized.data(), output.data(), n_blocks);

    float max_err = max_abs_error(input, output);
    float mean_err = mean_abs_error(input, output);

    EXPECT_LT(max_err, 0.01f) << "Q8_0 max absolute error: " << max_err;
    EXPECT_LT(mean_err, 0.005f) << "Q8_0 mean absolute error: " << mean_err;
}

// ============================================================================
// 3. Q5_0 Dequantize Error Bound
// ============================================================================

TEST(Accuracy, Q5_0DequantizeErrorBound) {
    const int64_t n_blocks = 64;
    const int64_t n_elements = n_blocks * 32;
    std::vector<float> input(n_elements);
    fill_random(input, -0.5f, 0.5f, 789);

    std::vector<uint8_t> quantized(static_cast<size_t>(n_blocks) * 22);
    std::vector<float> output(n_elements);

    quantize_q5_0_local(input.data(), quantized.data(), n_blocks);
    dequantize_q5_0(quantized.data(), output.data(), n_blocks);

    float max_err = max_abs_error(input, output);
    float mean_err = mean_abs_error(input, output);

    EXPECT_LT(max_err, 0.05f) << "Q5_0 max absolute error: " << max_err;
    EXPECT_LT(mean_err, 0.03f) << "Q5_0 mean absolute error: " << mean_err;
}

// ============================================================================
// 4. Q5_1 Dequantize Error Bound
// ============================================================================

TEST(Accuracy, Q5_1DequantizeErrorBound) {
    const int64_t n_blocks = 64;
    const int64_t n_elements = n_blocks * 32;
    std::vector<float> input(n_elements);
    fill_random(input, -1.0f, 1.0f, 1011);

    std::vector<uint8_t> quantized(static_cast<size_t>(n_blocks) * 24);
    std::vector<float> output(n_elements);

    quantize_q5_1_local(input.data(), quantized.data(), n_blocks);
    dequantize_q5_1(quantized.data(), output.data(), n_blocks);

    float max_err = max_abs_error(input, output);
    float mean_err = mean_abs_error(input, output);

    EXPECT_LT(max_err, 0.05f) << "Q5_1 max absolute error: " << max_err;
    EXPECT_LT(mean_err, 0.03f) << "Q5_1 mean absolute error: " << mean_err;
}

// ============================================================================
// 5. BF16 Dequantize Error Bound (roundtrip)
// ============================================================================

TEST(Accuracy, BF16DequantizeErrorBound) {
    const int64_t n_elements = 1024;
    std::vector<float> input(n_elements);
    fill_random(input, -100.0f, 100.0f, 2022);

    // FP32 → BF16 → FP32
    std::vector<uint16_t> bf16_data(n_elements);
    for (int64_t i = 0; i < n_elements; ++i) {
        bf16_data[i] = fp32_to_bf16(input[i]);
    }

    std::vector<float> output(n_elements);
    dequantize_bf16(bf16_data.data(), output.data(), n_elements);

    // Verify that BF16 roundtrip produces exact match for values in BF16 range
    // BF16 has 7-bit mantissa (vs 23-bit for FP32), so the roundtrip should
    // reproduce the value exactly when re-quantized
    for (int64_t i = 0; i < n_elements; ++i) {
        float roundtrip = output[i];
        // Re-quantize to BF16 — should be identical
        uint16_t bf16_again = fp32_to_bf16(roundtrip);
        float roundtrip2 = bf16_to_fp32(bf16_again);
        EXPECT_FLOAT_EQ(roundtrip, roundtrip2) << "BF16 roundtrip mismatch at index " << i;
    }
}

// ============================================================================
// 6. Softmax Sum To One
// ============================================================================

TEST(Accuracy, SoftmaxSumToOne) {
    const int64_t n = 256;
    std::vector<float> data(n);
    fill_random(data, -5.0f, 5.0f, 303);

    softmax_dispatch(data.data(), n);

    double sum = 0.0;
    for (int64_t i = 0; i < n; ++i) {
        sum += static_cast<double>(data[i]);
    }

    EXPECT_NEAR(sum, 1.0, 1e-5) << "Softmax output sum: " << sum;

    // All values should be in [0, 1]
    for (int64_t i = 0; i < n; ++i) {
        EXPECT_GE(data[i], 0.0f) << "Negative softmax output at index " << i;
        EXPECT_LE(data[i], 1.0f) << "Softmax output > 1 at index " << i;
    }
}

// ============================================================================
// 7. RMSNorm Preserves Scale
// ============================================================================

TEST(Accuracy, RMSNormPreservesScale) {
    const int64_t n = 256;
    std::vector<float> x(n, 1.0f);        // Unit input
    std::vector<float> weight(n, 1.0f);    // Unit weight
    std::vector<float> out(n, 0.0f);

    rmsnorm_dispatch(out.data(), x.data(), weight.data(), n);

    // With unit input and unit weight, RMSNorm should not amplify:
    // RMS of [1,1,...,1] = 1.0
    // out[i] = (x[i] / RMS) * weight[i] = (1.0 / 1.0) * 1.0 = 1.0
    // So output norm should be ~ sqrt(n) (same as input norm for unit vectors)

    double input_norm_sq = 0.0;
    double output_norm_sq = 0.0;
    for (int64_t i = 0; i < n; ++i) {
        input_norm_sq += static_cast<double>(x[i]) * static_cast<double>(x[i]);
        output_norm_sq += static_cast<double>(out[i]) * static_cast<double>(out[i]);
    }

    // Output norm should be approximately equal to input norm for unit input/weight
    double ratio = output_norm_sq / input_norm_sq;
    EXPECT_NEAR(ratio, 1.0, 0.01) << "RMSNorm amplification ratio: " << ratio;
}

// ============================================================================
// 8. GEMM Accuracy Vs Scalar
// ============================================================================

TEST(Accuracy, GEMMAccuracyVsScalar) {
    const int64_t M = 8;
    const int64_t N = 8;
    const int64_t K = 16;

    std::vector<float> A(static_cast<size_t>(M * K));
    std::vector<float> B(static_cast<size_t>(K * N));
    fill_random(A, -1.0f, 1.0f, 404);
    fill_random(B, -1.0f, 1.0f, 505);

    // Scalar GEMM result
    std::vector<float> C_scalar(static_cast<size_t>(M * N), 0.0f);
    GemmParams params_scalar;
    params_scalar.M = M;
    params_scalar.N = N;
    params_scalar.K = K;
    params_scalar.A = A.data();
    params_scalar.B = B.data();
    params_scalar.C = C_scalar.data();
    params_scalar.lda = K;
    params_scalar.ldb = N;
    params_scalar.ldc = N;
    params_scalar.alpha = 1.0f;
    params_scalar.beta = 0.0f;
    gemm_scalar(params_scalar);

    // Dispatched GEMM result
    std::vector<float> C_dispatch(static_cast<size_t>(M * N), 0.0f);
    GemmParams params_dispatch = params_scalar;
    params_dispatch.C = C_dispatch.data();
    gemm_dispatch(params_dispatch);

    // Verify max error < 1e-4
    float max_err = max_abs_error(C_scalar, C_dispatch);
    EXPECT_LT(max_err, 1e-4f) << "GEMM dispatch vs scalar max error: " << max_err;
}

// ============================================================================
// 9. Attention Output Finite
// ============================================================================

TEST(Accuracy, AttentionOutputFinite) {
    const int64_t n_heads = 2, n_kv_heads = 2, head_dim = 8, seq_len = 4, n_rep = 1;

    std::vector<float> q(static_cast<size_t>(n_heads * head_dim));
    std::vector<float> k_cache(static_cast<size_t>(n_kv_heads * seq_len * head_dim));
    std::vector<float> v_cache(static_cast<size_t>(n_kv_heads * seq_len * head_dim));
    std::vector<float> out(static_cast<size_t>(n_heads * head_dim), 0.0f);
    std::vector<float> scores(static_cast<size_t>(n_heads * seq_len), 0.0f);

    fill_random(q, -1.0f, 1.0f, 606);
    fill_random(k_cache, -1.0f, 1.0f, 707);
    fill_random(v_cache, -1.0f, 1.0f, 808);

    AttentionParams params;
    params.q = q.data();
    params.k_cache = k_cache.data();
    params.v_cache = v_cache.data();
    params.out = out.data();
    params.scores = scores.data();
    params.n_heads = n_heads;
    params.n_kv_heads = n_kv_heads;
    params.head_dim = head_dim;
    params.seq_len = seq_len;
    params.n_rep = n_rep;
    params.kv_stride = seq_len * head_dim;

    compute_attention(params);

    // Verify all output elements are finite (no NaN, no Inf)
    for (int64_t i = 0; i < n_heads * head_dim; ++i) {
        EXPECT_TRUE(std::isfinite(out[i])) << "Non-finite attention output at index " << i
                                            << " value=" << out[i];
    }
}

// ============================================================================
// 10. Pipeline Logits Finite
// ============================================================================

namespace {

/// Create a minimal synthetic model for accuracy testing
Model create_accuracy_model() {
    Model model;
    ModelWeights weights;
    HyperParams hp;
    hp.vocab_size = 32;
    hp.context_length = 64;
    hp.emb_dim = 16;
    hp.num_layers = 2;
    hp.num_heads = 2;
    hp.num_kv_heads = 2;
    hp.head_dim = 8;
    hp.ffn_dim = 32;
    hp.rope_freq_base = 10000.0f;
    hp.rope_freq_scale = 1.0f;
    hp.norm_eps = 1e-5f;
    hp.n_rot = 8;
    hp.n_embd_head_k = 8;
    hp.n_embd_head_v = 8;
    hp.use_gqa = false;
    hp.n_rep = 1;

    weights.hparams = hp;
    weights.arch = ModelArch::LLAMA;

    static std::vector<float> tok_emb_data(static_cast<size_t>(hp.vocab_size * hp.emb_dim), 0.01f);
    weights.tok_emb = tok_emb_data.data();
    weights.tok_emb_bytes = tok_emb_data.size() * sizeof(float);
    weights.tok_emb_ne0 = hp.emb_dim;
    weights.tok_emb_ne1 = hp.vocab_size;
    weights.tok_emb_dtype = DataType::F32;

    static std::vector<float> output_norm_data(static_cast<size_t>(hp.emb_dim), 1.0f);
    weights.output_norm = output_norm_data.data();
    weights.output_norm_bytes = output_norm_data.size() * sizeof(float);
    weights.output_norm_dtype = DataType::F32;

    static std::vector<float> output_proj_data(static_cast<size_t>(hp.vocab_size * hp.emb_dim), 0.01f);
    weights.output_proj = output_proj_data.data();
    weights.output_proj_bytes = output_proj_data.size() * sizeof(float);
    weights.output_proj_ne0 = hp.emb_dim;
    weights.output_proj_ne1 = hp.vocab_size;
    weights.output_proj_dtype = DataType::F32;

    weights.layers.resize(hp.num_layers);
    static std::vector<std::vector<float>> layer_storage;
    layer_storage.clear();

    for (int64_t l = 0; l < hp.num_layers; ++l) {
        auto& lw = weights.layers[static_cast<size_t>(l)];

        auto alloc = [&](int64_t n, float val) -> const float* {
            layer_storage.emplace_back(static_cast<size_t>(n), val);
            return layer_storage.back().data();
        };

        lw.attn_norm = alloc(hp.emb_dim, 1.0f);
        lw.attn_norm_bytes = hp.emb_dim * sizeof(float);
        lw.attn_norm_dtype = DataType::F32;

        lw.ffn_norm = alloc(hp.emb_dim, 1.0f);
        lw.ffn_norm_bytes = hp.emb_dim * sizeof(float);
        lw.ffn_norm_dtype = DataType::F32;

        int64_t kv_dim = hp.num_kv_heads * hp.head_dim;

        lw.wq = alloc(hp.num_heads * hp.head_dim * hp.emb_dim, 0.01f);
        lw.wq_bytes = hp.num_heads * hp.head_dim * hp.emb_dim * sizeof(float);
        lw.wq_ne0 = hp.emb_dim; lw.wq_ne1 = hp.num_heads * hp.head_dim;
        lw.wq_dtype = DataType::F32;

        lw.wk = alloc(kv_dim * hp.emb_dim, 0.01f);
        lw.wk_bytes = kv_dim * hp.emb_dim * sizeof(float);
        lw.wk_ne0 = hp.emb_dim; lw.wk_ne1 = kv_dim;
        lw.wk_dtype = DataType::F32;

        lw.wv = alloc(kv_dim * hp.emb_dim, 0.01f);
        lw.wv_bytes = kv_dim * hp.emb_dim * sizeof(float);
        lw.wv_ne0 = hp.emb_dim; lw.wv_ne1 = kv_dim;
        lw.wv_dtype = DataType::F32;

        lw.wo = alloc(hp.emb_dim * hp.num_heads * hp.head_dim, 0.01f);
        lw.wo_bytes = hp.emb_dim * hp.num_heads * hp.head_dim * sizeof(float);
        lw.wo_ne0 = hp.num_heads * hp.head_dim; lw.wo_ne1 = hp.emb_dim;
        lw.wo_dtype = DataType::F32;

        lw.w_gate = alloc(hp.ffn_dim * hp.emb_dim, 0.01f);
        lw.w_gate_bytes = hp.ffn_dim * hp.emb_dim * sizeof(float);
        lw.w_gate_ne0 = hp.emb_dim; lw.w_gate_ne1 = hp.ffn_dim;
        lw.w_gate_dtype = DataType::F32;

        lw.w_up = alloc(hp.ffn_dim * hp.emb_dim, 0.01f);
        lw.w_up_bytes = hp.ffn_dim * hp.emb_dim * sizeof(float);
        lw.w_up_ne0 = hp.emb_dim; lw.w_up_ne1 = hp.ffn_dim;
        lw.w_up_dtype = DataType::F32;

        lw.w_down = alloc(hp.emb_dim * hp.ffn_dim, 0.01f);
        lw.w_down_bytes = hp.emb_dim * hp.ffn_dim * sizeof(float);
        lw.w_down_ne0 = hp.ffn_dim; lw.w_down_ne1 = hp.emb_dim;
        lw.w_down_dtype = DataType::F32;
    }

    model.set_weights(std::move(weights));
    return model;
}

} // anonymous namespace

TEST(Accuracy, PipelineLogitsFinite) {
    auto model = create_accuracy_model();
    Pipeline pipeline(model);

    // Prefill with a few tokens
    pipeline.prefill({0, 1, 2, 3});

    const auto& logits = pipeline.logits();

    // Verify all logits are finite (no NaN, no Inf)
    for (size_t i = 0; i < logits.size(); ++i) {
        EXPECT_TRUE(std::isfinite(logits[i])) << "Non-finite logit at index " << i
                                                << " value=" << logits[i];
    }
}
