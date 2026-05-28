#pragma once
/// @file attention.h
/// @brief Phase 4 Attention computation for transformer inference.
///
/// Computes multi-head attention with support for GQA (Grouped-Query Attention).
/// Uses the existing kernel primitives (softmax_dispatch, gemm_dispatch) for computation.

#include <cstdint>

namespace kaguya {

/// Parameters for attention computation
struct AttentionParams {
    const float* q;          ///< Query: [n_heads, head_dim]
    const float* k_cache;    ///< Key cache: [n_kv_heads, kv_stride] — per-layer KV cache
    const float* v_cache;    ///< Value cache: [n_kv_heads, kv_stride] — per-layer KV cache
    float* out;              ///< Output: [n_heads, head_dim]
    float* scores;           ///< Scratch: [n_heads, seq_len] (attention weights)

    int64_t n_heads;         ///< Number of query heads
    int64_t n_kv_heads;      ///< Number of KV heads
    int64_t head_dim;        ///< Dimension of each head
    int64_t seq_len;         ///< Current sequence length (positions in cache)
    int64_t n_rep;           ///< n_heads / n_kv_heads (GQA repetition factor)
    int64_t kv_stride;       ///< Stride between KV heads in elements (= max_seq_len * head_dim for our cache)
};

/// Compute multi-head attention with GQA support
/// For each query head h, maps to kv_head = h / n_rep
/// Computes: scores = q_h @ k_cache^T, softmax, out_h = scores @ v_cache
void compute_attention(const AttentionParams& params);

} // namespace kaguya
