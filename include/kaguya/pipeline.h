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
#include <optional>
#include "kaguya/model.h"
#include "kaguya/kv_cache.h"
#include "kaguya/sampling.h"

namespace kaguya {

/// Generation stop condition
struct GenerationStop {
    int32_t eos_token_id = -1;  ///< EOS token ID to stop on (-1 = disabled)
    bool stop_on_eos = true;    ///< Whether to stop when EOS is generated
};

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
    /// Returns -1 if context is full (pos >= context_length)
    int32_t decode(Sampler& sampler);

    /// Convenience: generate n_predict tokens from a prompt
    std::vector<int32_t> generate(const std::vector<int32_t>& prompt,
                                   int32_t n_predict,
                                   Sampler& sampler);

    /// Convenience: generate n_predict tokens with EOS stopping
    std::vector<int32_t> generate(const std::vector<int32_t>& prompt,
                                   int32_t n_predict,
                                   Sampler& sampler,
                                   const GenerationStop& stop);

    const Model& model() const { return model_; }
    const KVCache& kv_cache() const { return kv_cache_; }
    int64_t current_pos() const { return pos_; }

    /// Get the logits from the last forward pass
    const std::vector<float>& logits() const { return logits_; }

    /// Check if the context window is full
    bool is_context_full() const { return pos_ >= model_.hparams().context_length; }

    /// Set EOS token ID for stopping
    void set_eos_token_id(int32_t eos_id) { eos_token_id_ = eos_id; }

    /// Get EOS token ID
    int32_t eos_token_id() const { return eos_token_id_; }

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
    std::vector<float> attn_proj_;     // [emb_dim] — attention output projection
    std::vector<float> ffn_down_;      // [emb_dim] — FFN down projection
    std::vector<float> weight_buf_;    // scratch for dequantized weights
    std::vector<int32_t> recent_tokens_; // for repetition penalty
    int32_t eos_token_id_ = -1;       // EOS token ID for stopping
};

} // namespace kaguya
