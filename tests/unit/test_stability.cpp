#include <gtest/gtest.h>
#include <vector>
#include <cmath>
#include <cstring>
#include <random>
#include <cstdint>
#include <stdexcept>
#include <limits>

#include "kaguya/kernels/gemm.h"
#include "kaguya/kernels/special_ops.h"
#include "kaguya/kernels/quantize.h"
#include "kaguya/kv_cache.h"
#include "kaguya/attention.h"
#include "kaguya/model.h"
#include "kaguya/pipeline.h"
#include "kaguya/sampling.h"

using namespace kaguya;
using namespace kaguya::kernels;

// ============================================================================
// Helper: create a minimal synthetic model for pipeline tests
// ============================================================================

namespace {

Model create_stability_model(int64_t context_length = 64) {
    Model model;
    ModelWeights weights;
    HyperParams hp;
    hp.vocab_size = 32;
    hp.context_length = context_length;
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

    static std::vector<float> tok_emb_data;
    tok_emb_data.assign(static_cast<size_t>(hp.vocab_size * hp.emb_dim), 0.01f);
    weights.tok_emb = tok_emb_data.data();
    weights.tok_emb_bytes = tok_emb_data.size() * sizeof(float);
    weights.tok_emb_ne0 = hp.emb_dim;
    weights.tok_emb_ne1 = hp.vocab_size;
    weights.tok_emb_dtype = DataType::F32;

    static std::vector<float> output_norm_data;
    output_norm_data.assign(static_cast<size_t>(hp.emb_dim), 1.0f);
    weights.output_norm = output_norm_data.data();
    weights.output_norm_bytes = output_norm_data.size() * sizeof(float);
    weights.output_norm_dtype = DataType::F32;

    static std::vector<float> output_proj_data;
    output_proj_data.assign(static_cast<size_t>(hp.vocab_size * hp.emb_dim), 0.01f);
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

// ============================================================================
// 1. GEMMZeroDimensions
// ============================================================================

TEST(Stability, GEMMZeroDimensions) {
    // M=0
    {
        GemmParams params{};
        params.M = 0; params.N = 4; params.K = 4;
        EXPECT_NO_THROW(gemm_scalar(params));
        EXPECT_NO_THROW(gemm_dispatch(params));
    }
    // N=0
    {
        std::vector<float> A(4, 1.0f);
        GemmParams params{};
        params.M = 4; params.N = 0; params.K = 4;
        params.A = A.data();
        EXPECT_NO_THROW(gemm_scalar(params));
        EXPECT_NO_THROW(gemm_dispatch(params));
    }
    // K=0
    {
        std::vector<float> C(4, 0.0f);
        GemmParams params{};
        params.M = 2; params.N = 2; params.K = 0;
        params.C = C.data();
        EXPECT_NO_THROW(gemm_scalar(params));
        EXPECT_NO_THROW(gemm_dispatch(params));
    }
}

// ============================================================================
// 2. GEMMSingleElement
// ============================================================================

TEST(Stability, GEMMSingleElement) {
    std::vector<float> A = {3.0f};
    std::vector<float> B = {7.0f};
    std::vector<float> C_scalar = {0.0f};
    std::vector<float> C_dispatch = {0.0f};

    GemmParams params{};
    params.M = 1; params.N = 1; params.K = 1;
    params.A = A.data(); params.B = B.data();

    params.C = C_scalar.data();
    gemm_scalar(params);
    EXPECT_FLOAT_EQ(C_scalar[0], 21.0f);

    params.C = C_dispatch.data();
    gemm_dispatch(params);
    EXPECT_NEAR(C_dispatch[0], 21.0f, 1e-4f);
}

// ============================================================================
// 3. GEMMLargeK
// ============================================================================

TEST(Stability, GEMMLargeK) {
    // GEMV with K=2048, M=1, N=1
    const int64_t M = 1, N = 1, K = 2048;
    std::vector<float> A(M * K, 0.001f);
    std::vector<float> B(K * N, 0.001f);
    std::vector<float> C_scalar(M * N, 0.0f);
    std::vector<float> C_dispatch(M * N, 0.0f);

    GemmParams params{};
    params.M = M; params.N = N; params.K = K;
    params.A = A.data(); params.B = B.data();

    params.C = C_scalar.data();
    gemm_scalar(params);

    params.C = C_dispatch.data();
    gemm_dispatch(params);

    // Both should produce the same result
    EXPECT_NEAR(C_dispatch[0], C_scalar[0], 1e-2f);

    // The result should be finite
    EXPECT_TRUE(std::isfinite(C_scalar[0]));
    EXPECT_TRUE(std::isfinite(C_dispatch[0]));
}

// ============================================================================
// 4. SoftmaxSingleElement
// ============================================================================

TEST(Stability, SoftmaxSingleElement) {
    std::vector<float> x = {42.0f};
    softmax_dispatch(x.data(), 1);
    EXPECT_FLOAT_EQ(x[0], 1.0f);
}

// ============================================================================
// 5. SoftmaxTwoElements
// ============================================================================

TEST(Stability, SoftmaxTwoElements) {
    std::vector<float> x = {1.0f, 2.0f};
    softmax_dispatch(x.data(), 2);

    // Sum should be 1
    EXPECT_NEAR(x[0] + x[1], 1.0f, 1e-6f);
    // Second element should be larger
    EXPECT_GT(x[1], x[0]);
    // Both should be positive
    EXPECT_GT(x[0], 0.0f);
    EXPECT_GT(x[1], 0.0f);
}

// ============================================================================
// 6. RMSNormZeroInput
// ============================================================================

TEST(Stability, RMSNormZeroInput) {
    const int64_t n = 16;
    std::vector<float> x(n, 0.0f);
    std::vector<float> weight(n, 1.0f);
    std::vector<float> out(n, -1.0f);

    // Should not crash or produce NaN
    EXPECT_NO_THROW(rmsnorm_dispatch(out.data(), x.data(), weight.data(), n));

    for (int64_t i = 0; i < n; ++i) {
        EXPECT_TRUE(std::isfinite(out[i])) << "Non-finite RMSNorm output at index " << i;
        // With zero input, output should be zero (or very close)
        EXPECT_NEAR(out[i], 0.0f, 1e-3f);
    }
}

// ============================================================================
// 7. LayerNormZeroInput
// ============================================================================

TEST(Stability, LayerNormZeroInput) {
    const int64_t n = 16;
    std::vector<float> x(n, 0.0f);
    std::vector<float> weight(n, 1.0f);
    std::vector<float> bias(n, 0.0f);
    std::vector<float> out(n, -1.0f);

    // Should not crash or produce NaN
    EXPECT_NO_THROW(layernorm_dispatch(out.data(), x.data(), weight.data(), bias.data(), n));

    // With zero input and zero bias, output should be close to zero
    for (int64_t i = 0; i < n; ++i) {
        EXPECT_TRUE(std::isfinite(out[i])) << "Non-finite LayerNorm output at index " << i;
    }
}

// ============================================================================
// 8. RoPEZeroPosition
// ============================================================================

TEST(Stability, RoPEZeroPosition) {
    // At position 0, RoPE should be identity (no rotation)
    const int64_t n_heads = 2, head_dim = 8;
    std::vector<float> x = {1.0f, 0.0f, 0.0f, 1.0f, 0.5f, 0.5f, -0.5f, 0.5f,
                             1.0f, 0.0f, 0.0f, 1.0f, 0.5f, 0.5f, -0.5f, 0.5f};
    std::vector<float> x_copy = x;

    rope_dispatch(x.data(), n_heads, head_dim, 0);

    // At pos=0, all theta values are 0, so cos=1, sin=0 → identity
    for (int64_t i = 0; i < n_heads * head_dim; ++i) {
        EXPECT_NEAR(x[i], x_copy[i], 1e-5f) << "RoPE at pos=0 should be identity, index " << i;
    }
}

// ============================================================================
// 9. KVCacheSinglePosition
// ============================================================================

TEST(Stability, KVCacheSinglePosition) {
    KVCache cache(1, 1, 4, 8);

    // Store data at position 0
    float data[] = {1.0f, 2.0f, 3.0f, 4.0f};
    std::memcpy(cache.key(0, 0, 0), data, sizeof(data));
    std::memcpy(cache.value(0, 0, 0), data, sizeof(data));
    cache.advance(1);

    EXPECT_EQ(cache.n_positions(), 1);

    // Verify data is intact
    const float* k = cache.key(0, 0, 0);
    const float* v = cache.value(0, 0, 0);
    for (int i = 0; i < 4; ++i) {
        EXPECT_FLOAT_EQ(k[i], data[i]);
        EXPECT_FLOAT_EQ(v[i], data[i]);
    }
}

// ============================================================================
// 10. KVCacheMaxPositions
// ============================================================================

TEST(Stability, KVCacheMaxPositions) {
    const int64_t max_seq_len = 16;
    KVCache cache(1, 1, 4, max_seq_len);

    // Fill all positions
    for (int64_t pos = 0; pos < max_seq_len; ++pos) {
        float* k = cache.key(0, 0, pos);
        float* v = cache.value(0, 0, pos);
        for (int d = 0; d < 4; ++d) {
            k[d] = static_cast<float>(pos * 4 + d);
            v[d] = static_cast<float>(pos * 4 + d + 100);
        }
    }
    cache.advance(max_seq_len);

    EXPECT_EQ(cache.n_positions(), max_seq_len);

    // Verify all data
    for (int64_t pos = 0; pos < max_seq_len; ++pos) {
        const float* k = cache.key(0, 0, pos);
        const float* v = cache.value(0, 0, pos);
        for (int d = 0; d < 4; ++d) {
            EXPECT_FLOAT_EQ(k[d], static_cast<float>(pos * 4 + d));
            EXPECT_FLOAT_EQ(v[d], static_cast<float>(pos * 4 + d + 100));
        }
    }
}

// ============================================================================
// 11. PipelineEmptyPrompt
// ============================================================================

TEST(Stability, PipelineEmptyPrompt) {
    auto model = create_stability_model();
    Pipeline pipeline(model);

    // Empty prompt should not crash
    EXPECT_NO_THROW(pipeline.prefill({}));
    EXPECT_EQ(pipeline.current_pos(), 0);
}

// ============================================================================
// 12. PipelineSingleToken
// ============================================================================

TEST(Stability, PipelineSingleToken) {
    auto model = create_stability_model();
    Pipeline pipeline(model);

    // Single token prefill
    EXPECT_NO_THROW(pipeline.prefill({0}));
    EXPECT_EQ(pipeline.current_pos(), 1);

    // Logits should be finite
    const auto& logits = pipeline.logits();
    for (size_t i = 0; i < logits.size(); ++i) {
        EXPECT_TRUE(std::isfinite(logits[i])) << "Non-finite logit at index " << i;
    }
}

// ============================================================================
// 13. AttentionSinglePosition
// ============================================================================

TEST(Stability, AttentionSinglePosition) {
    const int64_t n_heads = 1, n_kv_heads = 1, head_dim = 4, seq_len = 1, n_rep = 1;

    float q[] = {1.0f, 0.0f, 0.0f, 0.0f};
    float k_cache[] = {1.0f, 0.0f, 0.0f, 0.0f};
    float v_cache[] = {5.0f, 6.0f, 7.0f, 8.0f};
    float out[4] = {};
    float scores[1] = {};

    AttentionParams params;
    params.q = q;
    params.k_cache = k_cache;
    params.v_cache = v_cache;
    params.out = out;
    params.scores = scores;
    params.n_heads = n_heads;
    params.n_kv_heads = n_kv_heads;
    params.head_dim = head_dim;
    params.seq_len = seq_len;
    params.n_rep = n_rep;
    params.kv_stride = seq_len * head_dim;

    EXPECT_NO_THROW(compute_attention(params));

    // Output should be finite and non-zero
    float sum = 0.0f;
    for (int i = 0; i < 4; ++i) {
        EXPECT_TRUE(std::isfinite(out[i]));
        sum += std::fabs(out[i]);
    }
    EXPECT_GT(sum, 0.0f);
}

// ============================================================================
// 14. DequantizeEmptyBlocks
// ============================================================================

TEST(Stability, DequantizeEmptyBlocks) {
    // n_blocks=0 should not crash
    EXPECT_NO_THROW(dequantize_q4_0(nullptr, nullptr, 0));
    EXPECT_NO_THROW(dequantize_q8_0(nullptr, nullptr, 0));
    EXPECT_NO_THROW(dequantize_bf16(nullptr, nullptr, 0));
}

// ============================================================================
// 15. SamplerAllNegativeLogits
// ============================================================================

TEST(Stability, SamplerAllNegativeLogits) {
    const int64_t vocab_size = 16;
    std::vector<float> logits(vocab_size, -1000.0f);

    SamplingParams sp;
    sp.temperature = 1.0f;
    sp.top_k = 0;
    sp.top_p = 1.0f;
    sp.seed = 42;
    Sampler sampler(sp);

    // Should not crash, should return a valid token
    int32_t token = sampler.sample(logits.data(), vocab_size, {});
    EXPECT_GE(token, 0);
    EXPECT_LT(token, vocab_size);
}

// ============================================================================
// 16. SamplerAllEqualLogits
// ============================================================================

TEST(Stability, SamplerAllEqualLogits) {
    const int64_t vocab_size = 16;
    std::vector<float> logits(vocab_size, 5.0f);

    SamplingParams sp;
    sp.temperature = 1.0f;
    sp.top_k = 0;
    sp.top_p = 1.0f;
    sp.seed = 42;
    Sampler sampler(sp);

    // All equal logits → uniform distribution, should return a valid token
    int32_t token = sampler.sample(logits.data(), vocab_size, {});
    EXPECT_GE(token, 0);
    EXPECT_LT(token, vocab_size);
}

// ============================================================================
// 17. GEMMWithAlphaBeta
// ============================================================================

TEST(Stability, GEMMWithAlphaBeta) {
    const int64_t M = 4, N = 8, K = 4;
    std::vector<float> A(M * K), B(K * N);
    std::vector<float> C_scalar(M * N), C_dispatch(M * N);

    // Fill with random data
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (auto& x : A) x = dist(rng);
    for (auto& x : B) x = dist(rng);

    // Initialize C with non-zero values
    for (int64_t i = 0; i < M * N; ++i) {
        C_scalar[i] = C_dispatch[i] = static_cast<float>(i + 1);
    }

    GemmParams params{};
    params.M = M; params.N = N; params.K = K;
    params.A = A.data(); params.B = B.data();
    params.alpha = 0.5f; params.beta = 2.0f;

    params.C = C_scalar.data();
    gemm_scalar(params);

    params.C = C_dispatch.data();
    gemm_dispatch(params);

    // Results should match
    for (int64_t i = 0; i < M * N; ++i) {
        EXPECT_NEAR(C_dispatch[i], C_scalar[i], 1e-3f) << "at index " << i;
    }

    // Results should be finite
    for (int64_t i = 0; i < M * N; ++i) {
        EXPECT_TRUE(std::isfinite(C_scalar[i]));
        EXPECT_TRUE(std::isfinite(C_dispatch[i]));
    }
}

// ============================================================================
// 18. KVCacheResetAndReuse
// ============================================================================

TEST(Stability, KVCacheResetAndReuse) {
    KVCache cache(1, 1, 4, 8);

    // First use: store data
    float data1[] = {1.0f, 2.0f, 3.0f, 4.0f};
    std::memcpy(cache.key(0, 0, 0), data1, sizeof(data1));
    cache.advance(2);
    EXPECT_EQ(cache.n_positions(), 2);

    // Reset
    cache.reset();
    EXPECT_EQ(cache.n_positions(), 0);

    // Verify data is zeroed
    const float* k = cache.key(0, 0, 0);
    for (int i = 0; i < 4; ++i) {
        EXPECT_FLOAT_EQ(k[i], 0.0f);
    }

    // Reuse: store different data
    float data2[] = {10.0f, 20.0f, 30.0f, 40.0f};
    std::memcpy(cache.key(0, 0, 0), data2, sizeof(data2));
    cache.advance(1);
    EXPECT_EQ(cache.n_positions(), 1);

    // Verify new data
    k = cache.key(0, 0, 0);
    for (int i = 0; i < 4; ++i) {
        EXPECT_FLOAT_EQ(k[i], data2[i]);
    }
}

// ============================================================================
// 19. PipelineMultipleResets
// ============================================================================

TEST(Stability, PipelineMultipleResets) {
    auto model = create_stability_model();
    Pipeline pipeline(model);

    SamplingParams sp;
    sp.temperature = 1.0f;
    sp.top_k = 0;
    sp.top_p = 1.0f;
    sp.seed = 42;

    for (int cycle = 0; cycle < 3; ++cycle) {
        pipeline.reset();
        EXPECT_EQ(pipeline.current_pos(), 0);

        pipeline.prefill({0, 1, 2});
        EXPECT_EQ(pipeline.current_pos(), 3);

        Sampler sampler(sp);
        int32_t token = pipeline.decode(sampler);
        EXPECT_GE(token, 0);
        EXPECT_LT(token, model.hparams().vocab_size);
    }
}

// ============================================================================
// 20. KVCacheMoveSemantics
// ============================================================================

TEST(Stability, KVCacheMoveSemantics) {
    // Move constructor
    {
        KVCache cache1(2, 4, 32, 64);
        float data[] = {42.0f, 43.0f, 44.0f, 45.0f};
        std::memcpy(cache1.key(0, 0, 0), data, sizeof(data));
        cache1.advance(3);

        KVCache cache2(std::move(cache1));
        EXPECT_EQ(cache2.n_layers(), 2);
        EXPECT_EQ(cache2.n_kv_heads(), 4);
        EXPECT_EQ(cache2.head_dim(), 32);
        EXPECT_EQ(cache2.max_seq_len(), 64);
        EXPECT_EQ(cache2.n_positions(), 3);

        const float* k = cache2.key(0, 0, 0);
        EXPECT_FLOAT_EQ(k[0], 42.0f);
        EXPECT_FLOAT_EQ(k[1], 43.0f);
        EXPECT_FLOAT_EQ(k[2], 44.0f);
        EXPECT_FLOAT_EQ(k[3], 45.0f);
    }

    // Move assignment
    {
        KVCache cache1(2, 4, 32, 64);
        float data[] = {10.0f, 20.0f, 30.0f, 40.0f};
        std::memcpy(cache1.key(0, 0, 0), data, sizeof(data));
        cache1.advance(5);

        KVCache cache2(1, 1, 4, 8);
        cache2 = std::move(cache1);

        EXPECT_EQ(cache2.n_layers(), 2);
        EXPECT_EQ(cache2.n_kv_heads(), 4);
        EXPECT_EQ(cache2.head_dim(), 32);
        EXPECT_EQ(cache2.max_seq_len(), 64);
        EXPECT_EQ(cache2.n_positions(), 5);

        const float* k = cache2.key(0, 0, 0);
        EXPECT_FLOAT_EQ(k[0], 10.0f);
        EXPECT_FLOAT_EQ(k[1], 20.0f);
        EXPECT_FLOAT_EQ(k[2], 30.0f);
        EXPECT_FLOAT_EQ(k[3], 40.0f);
    }
}
