#pragma once
/// @file pipeline.h
/// @brief Phase 4 Transformer inference pipeline.
///
/// Orchestrates the full transformer forward pass:
///   token → embedding → [layer: norm→QKV→RoPE→attn→residual→norm→FFN→residual] → output_norm → logits → sample
///
/// Supports quantized weights via on-the-fly dequantization.

#include <vector>
#include <cstdint>
#include <memory>
#include "kaguya/model.h"
#include "kaguya/kv_cache.h"
#include "kaguya/sampling.h"

namespace kaguya {

/// Transformer inference pipeline
class Pipeline {
public:
    /// Create a pipeline for the given model
    explicit Pipeline(const Model& model);

    ~Pipeline() = default;

    /// Reset for a new sequence (clears KV cache and position)
    void reset();

    /// Process prompt tokens (prefill — processes tokens one at a time)
    void prefill(const std::vector<int32_t>& tokens);

    /// Generate one token using the given sampler
    int32_t decode(Sampler& sampler);

    /// Convenience: generate n_predict tokens from a prompt
    std::vector<int32_t> generate(const std::vector<int32_t>& prompt,
                                   int32_t n_predict,
                                   Sampler& sampler);

    const Model& model() const { return model_; }
    const KVCache& kv_cache() const { return kv_cache_; }
    int64_t current_pos() const { return pos_; }

    /// Get the logits from the last forward pass
    const std::vector<float>& logits() const { return logits_; }

private:
    /// Forward pass for a single token at given position
    void forward(int64_t pos, int32_t token);

    /// Process one transformer layer
    void forward_layer(int64_t layer, float* x, int64_t pos);

    /// Dequantize a weight tensor to the internal FP32 buffer
    /// Returns pointer to the dequantized data (valid until next call)
    float* dequantize_weight(const void* data, DataType dtype, int64_t n_elements);

    /// Perform GEMV: y = W * x^T where W is [M, K] and x is [K]
    /// Handles dequantization internally if weight is quantized
    void weighted_project(float* y,
                          const void* w_data, DataType w_dtype,
                          int64_t w_ne0, int64_t w_ne1,
                          const float* x);

    const Model& model_;
    KVCache kv_cache_;
    int64_t pos_ = 0;

    // Working buffers (pre-allocated to avoid per-step allocation)
    std::vector<float> logits_;        // [vocab_size]
    std::vector<float> xb_;            // [emb_dim] — after norm
    std::vector<float> xb2_;           // [emb_dim] — after attention
    std::vector<float> q_;             // [n_heads * head_dim]
    std::vector<float> key_tmp_;       // [n_kv_heads * head_dim]
    std::vector<float> val_tmp_;       // [n_kv_heads * head_dim]
    std::vector<float> att_scores_;    // [n_heads, max_seq_len]
    std::vector<float> hb_;            // [ffn_dim] — gate
    std::vector<float> hb2_;           // [ffn_dim] — up
    std::vector<float> weight_buf_;    // scratch for dequantized weights
    std::vector<int32_t> recent_tokens_; // for repetition penalty
};

} // namespace kaguya
