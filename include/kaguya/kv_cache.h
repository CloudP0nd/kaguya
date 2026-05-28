#pragma once
/// @file kv_cache.h
/// @brief Phase 4 KV-Cache Manager for transformer inference.
///
/// Stores key and value tensors for all layers in contiguous memory.
/// Layout: [n_layers, n_kv_heads, max_seq_len, head_dim]
/// This layout ensures that for a given (layer, kv_head), all positions
/// are contiguous in memory, enabling efficient GEMM for attention.

#include <cstdint>
#include <vector>
#include "kaguya/memory_manager.h"

namespace kaguya {

/// KV Cache for transformer inference
class KVCache {
public:
    /// Create a KV cache
    /// @param n_layers Number of transformer layers
    /// @param n_kv_heads Number of KV heads (may differ from query heads for GQA)
    /// @param head_dim Dimension of each head
    /// @param max_seq_len Maximum sequence length (context window)
    KVCache(int64_t n_layers, int64_t n_kv_heads, int64_t head_dim, int64_t max_seq_len);

    ~KVCache() = default;

    // Non-copyable, movable
    KVCache(const KVCache&) = delete;
    KVCache& operator=(const KVCache&) = delete;
    KVCache(KVCache&&) noexcept = default;
    KVCache& operator=(KVCache&&) noexcept = default;

    // --- Per-position access (for storing new K/V during inference) ---

    /// Get pointer to key vector at (layer, kv_head, pos) — length head_dim
    float* key(int64_t layer, int64_t kv_head, int64_t pos);
    const float* key(int64_t layer, int64_t kv_head, int64_t pos) const;

    /// Get pointer to value vector at (layer, kv_head, pos) — length head_dim
    float* value(int64_t layer, int64_t kv_head, int64_t pos);
    const float* value(int64_t layer, int64_t kv_head, int64_t pos) const;

    // --- Full head access (for attention computation) ---

    /// Get key matrix for (layer, kv_head): [max_seq_len, head_dim]
    float* key_head(int64_t layer, int64_t kv_head);
    const float* key_head(int64_t layer, int64_t kv_head) const;

    /// Get value matrix for (layer, kv_head): [max_seq_len, head_dim]
    float* value_head(int64_t layer, int64_t kv_head);
    const float* value_head(int64_t layer, int64_t kv_head) const;

    // --- State management ---

    /// Current number of valid positions in the cache
    int64_t n_positions() const { return n_positions_; }

    /// Advance the position counter by n
    void advance(int64_t n);

    /// Reset the cache (start a new sequence)
    void reset();

    // --- Dimensions ---

    int64_t n_layers() const { return n_layers_; }
    int64_t n_kv_heads() const { return n_kv_heads_; }
    int64_t head_dim() const { return head_dim_; }
    int64_t max_seq_len() const { return max_seq_len_; }
    int64_t kv_dim() const { return n_kv_heads_ * head_dim_; }

    /// Total elements per layer's key or value cache
    int64_t layer_elements() const { return n_kv_heads_ * max_seq_len_ * head_dim_; }

    /// Total allocated size in bytes (for one of key or value)
    size_t allocated_bytes() const { return key_buf_.size() * sizeof(float); }

private:
    int64_t n_layers_;
    int64_t n_kv_heads_;
    int64_t head_dim_;
    int64_t max_seq_len_;
    int64_t n_positions_ = 0;

    // Strides (in elements)
    int64_t layer_stride_;   // n_kv_heads * max_seq_len * head_dim
    int64_t head_stride_;    // max_seq_len * head_dim
    int64_t pos_stride_;     // head_dim

    // Key cache: [n_layers, n_kv_heads, max_seq_len, head_dim]
    std::vector<float> key_buf_;
    // Value cache: [n_layers, n_kv_heads, max_seq_len, head_dim]
    std::vector<float> value_buf_;
};

} // namespace kaguya
