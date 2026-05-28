#include <gtest/gtest.h>
#include "kaguya/model.h"
#include "kaguya/pipeline.h"
#include "kaguya/sampling.h"
#include "kaguya/kv_cache.h"
#include "kaguya/attention.h"
#include "kaguya/batch.h"

#include <vector>
#include <cmath>

using namespace kaguya;

// ============================================================================
// Pipeline construction (synthetic model)
// ============================================================================

namespace {

/// Create a minimal synthetic model for testing
/// The model won't produce meaningful output, but tests the pipeline mechanics
Model create_synthetic_model() {
    Model model;

    // We need to manually construct ModelWeights with F32 data
    // Since we can't easily create a GGUF file, we'll set up the model directly
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

    // Token embedding: [vocab_size, emb_dim] = [32, 16]
    static std::vector<float> tok_emb_data(static_cast<size_t>(hp.vocab_size * hp.emb_dim), 0.01f);
    weights.tok_emb = tok_emb_data.data();
    weights.tok_emb_bytes = tok_emb_data.size() * sizeof(float);
    weights.tok_emb_ne0 = hp.emb_dim;
    weights.tok_emb_ne1 = hp.vocab_size;
    weights.tok_emb_dtype = DataType::F32;

    // Output norm: [emb_dim] = [16]
    static std::vector<float> output_norm_data(static_cast<size_t>(hp.emb_dim), 1.0f);
    weights.output_norm = output_norm_data.data();
    weights.output_norm_bytes = output_norm_data.size() * sizeof(float);
    weights.output_norm_dtype = DataType::F32;

    // Output projection: [vocab_size, emb_dim] = [32, 16]
    static std::vector<float> output_proj_data(static_cast<size_t>(hp.vocab_size * hp.emb_dim), 0.01f);
    weights.output_proj = output_proj_data.data();
    weights.output_proj_bytes = output_proj_data.size() * sizeof(float);
    weights.output_proj_ne0 = hp.emb_dim;
    weights.output_proj_ne1 = hp.vocab_size;
    weights.output_proj_dtype = DataType::F32;

    // Per-layer weights
    weights.layers.resize(hp.num_layers);

    for (int64_t l = 0; l < hp.num_layers; ++l) {
        auto& lw = weights.layers[static_cast<size_t>(l)];

        // Attention norm: [emb_dim]
        static std::vector<float> attn_norm_data[2];
        attn_norm_data[l].resize(static_cast<size_t>(hp.emb_dim), 1.0f);
        lw.attn_norm = attn_norm_data[l].data();
        lw.attn_norm_bytes = attn_norm_data[l].size() * sizeof(float);
        lw.attn_norm_dtype = DataType::F32;

        // FFN norm: [emb_dim]
        static std::vector<float> ffn_norm_data[2];
        ffn_norm_data[l].resize(static_cast<size_t>(hp.emb_dim), 1.0f);
        lw.ffn_norm = ffn_norm_data[l].data();
        lw.ffn_norm_bytes = ffn_norm_data[l].size() * sizeof(float);
        lw.ffn_norm_dtype = DataType::F32;

        // Wq: [n_heads * head_dim, emb_dim] = [16, 16]
        static std::vector<float> wq_data[2];
        wq_data[l].assign(static_cast<size_t>(hp.num_heads * hp.head_dim * hp.emb_dim), 0.01f);
        lw.wq = wq_data[l].data();
        lw.wq_bytes = wq_data[l].size() * sizeof(float);
        lw.wq_ne0 = hp.emb_dim;
        lw.wq_ne1 = hp.num_heads * hp.head_dim;
        lw.wq_dtype = DataType::F32;

        // Wk: [n_kv_heads * head_dim, emb_dim] = [16, 16]
        static std::vector<float> wk_data[2];
        wk_data[l].assign(static_cast<size_t>(hp.num_kv_heads * hp.head_dim * hp.emb_dim), 0.01f);
        lw.wk = wk_data[l].data();
        lw.wk_bytes = wk_data[l].size() * sizeof(float);
        lw.wk_ne0 = hp.emb_dim;
        lw.wk_ne1 = hp.num_kv_heads * hp.head_dim;
        lw.wk_dtype = DataType::F32;

        // Wv: [n_kv_heads * head_dim, emb_dim] = [16, 16]
        static std::vector<float> wv_data[2];
        wv_data[l].assign(static_cast<size_t>(hp.num_kv_heads * hp.head_dim * hp.emb_dim), 0.01f);
        lw.wv = wv_data[l].data();
        lw.wv_bytes = wv_data[l].size() * sizeof(float);
        lw.wv_ne0 = hp.emb_dim;
        lw.wv_ne1 = hp.num_kv_heads * hp.head_dim;
        lw.wv_dtype = DataType::F32;

        // Wo: [emb_dim, n_heads * head_dim] = [16, 16]
        static std::vector<float> wo_data[2];
        wo_data[l].assign(static_cast<size_t>(hp.emb_dim * hp.num_heads * hp.head_dim), 0.01f);
        lw.wo = wo_data[l].data();
        lw.wo_bytes = wo_data[l].size() * sizeof(float);
        lw.wo_ne0 = hp.num_heads * hp.head_dim;
        lw.wo_ne1 = hp.emb_dim;
        lw.wo_dtype = DataType::F32;

        // W_gate: [ffn_dim, emb_dim] = [32, 16]
        static std::vector<float> w_gate_data[2];
        w_gate_data[l].assign(static_cast<size_t>(hp.ffn_dim * hp.emb_dim), 0.01f);
        lw.w_gate = w_gate_data[l].data();
        lw.w_gate_bytes = w_gate_data[l].size() * sizeof(float);
        lw.w_gate_ne0 = hp.emb_dim;
        lw.w_gate_ne1 = hp.ffn_dim;
        lw.w_gate_dtype = DataType::F32;

        // W_up: [ffn_dim, emb_dim] = [32, 16]
        static std::vector<float> w_up_data[2];
        w_up_data[l].assign(static_cast<size_t>(hp.ffn_dim * hp.emb_dim), 0.01f);
        lw.w_up = w_up_data[l].data();
        lw.w_up_bytes = w_up_data[l].size() * sizeof(float);
        lw.w_up_ne0 = hp.emb_dim;
        lw.w_up_ne1 = hp.ffn_dim;
        lw.w_up_dtype = DataType::F32;

        // W_down: [emb_dim, ffn_dim] = [16, 32]
        static std::vector<float> w_down_data[2];
        w_down_data[l].assign(static_cast<size_t>(hp.emb_dim * hp.ffn_dim), 0.01f);
        lw.w_down = w_down_data[l].data();
        lw.w_down_bytes = w_down_data[l].size() * sizeof(float);
        lw.w_down_ne0 = hp.ffn_dim;
        lw.w_down_ne1 = hp.emb_dim;
        lw.w_down_dtype = DataType::F32;
    }

    model.set_weights(std::move(weights));
    return model;
}

} // anonymous namespace

// ============================================================================
// Pipeline basic tests
// ============================================================================

TEST(Inference, PipelineConstruction) {
    auto model = create_synthetic_model();
    Pipeline pipeline(model);
    EXPECT_EQ(pipeline.current_pos(), 0);
}

TEST(Inference, PipelineReset) {
    auto model = create_synthetic_model();
    Pipeline pipeline(model);

    pipeline.prefill({1, 2, 3});
    EXPECT_GT(pipeline.current_pos(), 0);

    pipeline.reset();
    EXPECT_EQ(pipeline.current_pos(), 0);
}

TEST(Inference, PipelinePrefillAdvancesPosition) {
    auto model = create_synthetic_model();
    Pipeline pipeline(model);

    pipeline.prefill({0, 1, 2});
    EXPECT_EQ(pipeline.current_pos(), 3);
}

TEST(Inference, PipelineDecodeProducesToken) {
    auto model = create_synthetic_model();
    Pipeline pipeline(model);

    pipeline.prefill({0});

    SamplingParams sp;
    sp.temperature = 1.0f;
    sp.top_k = 0; // disable top-k to avoid filtering all tokens
    sp.top_p = 1.0f; // disable top-p
    Sampler sampler(sp);

    int32_t token = pipeline.decode(sampler);
    EXPECT_GE(token, 0);
    EXPECT_LT(token, model.hparams().vocab_size);
    EXPECT_EQ(pipeline.current_pos(), 2);
}

TEST(Inference, PipelineGenerateReturnsTokens) {
    auto model = create_synthetic_model();
    Pipeline pipeline(model);

    SamplingParams sp;
    sp.temperature = 1.0f;
    sp.top_k = 0;
    sp.top_p = 1.0f;
    sp.seed = 42;
    Sampler sampler(sp);

    auto tokens = pipeline.generate({0, 1}, 5, sampler);
    EXPECT_EQ(tokens.size(), 5u);

    // All tokens should be valid
    for (auto t : tokens) {
        EXPECT_GE(t, 0);
        EXPECT_LT(t, model.hparams().vocab_size);
    }
}

TEST(Inference, PipelineGenerateDeterministicWithSeed) {
    auto model = create_synthetic_model();

    SamplingParams sp;
    sp.temperature = 1.0f;
    sp.top_k = 0;
    sp.top_p = 1.0f;
    sp.seed = 42;

    Pipeline p1(model);
    Sampler s1(sp);
    auto tokens1 = p1.generate({0}, 10, s1);

    Pipeline p2(model);
    Sampler s2(sp);
    auto tokens2 = p2.generate({0}, 10, s2);

    EXPECT_EQ(tokens1, tokens2);
}

TEST(Inference, PipelineLogitsSizeMatchesVocab) {
    auto model = create_synthetic_model();
    Pipeline pipeline(model);

    pipeline.prefill({0});

    const auto& logits = pipeline.logits();
    EXPECT_EQ(static_cast<int64_t>(logits.size()), model.hparams().vocab_size);
}

// ============================================================================
// Batch inference tests
// ============================================================================

TEST(Inference, BatchCreation) {
    auto model = create_synthetic_model();
    BatchInference batch(model, 4);
    EXPECT_EQ(batch.num_sequences(), 0);
    EXPECT_EQ(batch.num_active(), 0);
}

TEST(Inference, BatchAddSequence) {
    auto model = create_synthetic_model();
    BatchInference batch(model, 4);

    int id = batch.add_sequence({0, 1, 2}, 5);
    EXPECT_EQ(id, 0);
    EXPECT_EQ(batch.num_sequences(), 1);
    EXPECT_EQ(batch.num_active(), 1);
    EXPECT_FALSE(batch.all_finished());
}

TEST(Inference, BatchStepGeneratesToken) {
    auto model = create_synthetic_model();
    BatchInference batch(model, 4);

    batch.add_sequence({0, 1}, 3);

    SamplingParams sp;
    sp.temperature = 1.0f;
    sp.top_k = 0;
    sp.top_p = 1.0f;
    sp.seed = 42;
    Sampler sampler(sp);

    batch.step(sampler);

    EXPECT_EQ(batch.sequence(0).output.size(), 1u);
}

TEST(Inference, BatchRunAllFinishes) {
    auto model = create_synthetic_model();
    BatchInference batch(model, 4);

    batch.add_sequence({0}, 3);
    batch.add_sequence({1}, 3);

    SamplingParams sp;
    sp.temperature = 1.0f;
    sp.top_k = 0;
    sp.top_p = 1.0f;
    sp.seed = 42;
    Sampler sampler(sp);

    batch.run_all(sampler);

    EXPECT_TRUE(batch.all_finished());
    EXPECT_EQ(batch.sequence(0).output.size(), 3u);
    EXPECT_EQ(batch.sequence(1).output.size(), 3u);
}
