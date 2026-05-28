#include <gtest/gtest.h>
#include "kaguya/attention.h"
#include "kaguya/kernels/special_ops.h"

#include <cmath>
#include <vector>
#include <numeric>

using namespace kaguya;

// ============================================================================
// Basic attention: single head, single position (self-attention)
// ============================================================================

TEST(Attention, SingleHeadSinglePosition) {
    // Q = K = V with 1 position, 1 head, head_dim=4
    const int64_t n_heads = 1, n_kv_heads = 1, head_dim = 4, seq_len = 1, n_rep = 1;

    float q[] = {1.0f, 0.0f, 0.0f, 0.0f};
    float k_cache[] = {1.0f, 0.0f, 0.0f, 0.0f}; // [1, 4]
    float v_cache[] = {5.0f, 6.0f, 7.0f, 8.0f}; // [1, 4]

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

    compute_attention(params);

    // With Q=K and softmax, the single attention weight is 1.0
    // Output should be the value vector
    float expected_scale = 1.0f / std::sqrt(4.0f); // 1/sqrt(head_dim)
    // Score = Q*K / sqrt(d) = (1.0*1.0) / 2.0 = 0.5
    // After softmax with single element: exp(0.5)/exp(0.5) = 1.0
    // Output = 1.0 * V = [5, 6, 7, 8]
    for (int i = 0; i < 4; ++i) {
        EXPECT_NEAR(out[i], v_cache[i], 1e-5f);
    }
}

// ============================================================================
// Two positions: attend more to similar vector
// ============================================================================

TEST(Attention, TwoPositionsPrefersSimilar) {
    const int64_t n_heads = 1, n_kv_heads = 1, head_dim = 4, seq_len = 2, n_rep = 1;

    float q[] = {1.0f, 0.0f, 0.0f, 0.0f}; // Similar to position 0

    // K for 2 positions
    float k_cache[] = {
        1.0f, 0.0f, 0.0f, 0.0f,  // pos 0: similar to Q
        0.0f, 0.0f, 0.0f, 1.0f,  // pos 1: orthogonal to Q
    };

    // V for 2 positions
    float v_cache[] = {
        10.0f, 20.0f, 30.0f, 40.0f,  // pos 0
        50.0f, 60.0f, 70.0f, 80.0f,  // pos 1
    };

    float out[4] = {};
    float scores[2] = {};

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

    compute_attention(params);

    // Output should be closer to position 0's value since Q is more similar to K[0]
    // out[0] should be closer to 10 than to 50
    EXPECT_LT(out[0], 30.0f); // Closer to 10 (pos 0) than 50 (pos 1)
}

// ============================================================================
// GQA: 2 query heads, 1 KV head
// ============================================================================

TEST(Attention, GQA_TwoQueryHeadsOneKVHead) {
    const int64_t n_heads = 2, n_kv_heads = 1, head_dim = 4, seq_len = 1, n_rep = 2;

    float q[] = {
        1.0f, 0.0f, 0.0f, 0.0f,  // head 0
        0.0f, 1.0f, 0.0f, 0.0f,  // head 1
    };

    float k_cache[] = {
        1.0f, 0.0f, 0.0f, 0.0f,  // single KV head, pos 0
    };

    float v_cache[] = {
        5.0f, 6.0f, 7.0f, 8.0f,
    };

    float out[8] = {};
    float scores[2] = {};

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

    compute_attention(params);

    // Both heads attend to the same KV head
    // Head 0: Q similar to K, should get V
    // Head 1: Q partially similar, different score but still gets V weighted
    // Both outputs should be non-zero (softmax with single position = weight 1)
    for (int h = 0; h < 2; ++h) {
        float sum = 0.0f;
        for (int d = 0; d < 4; ++d) {
            sum += std::fabs(out[h * 4 + d]);
        }
        EXPECT_GT(sum, 0.0f);
    }
}

// ============================================================================
// Attention weights sum to 1
// ============================================================================

TEST(Attention, AttentionWeightsSumToOne) {
    const int64_t n_heads = 1, n_kv_heads = 1, head_dim = 8, seq_len = 5, n_rep = 1;

    std::vector<float> q(static_cast<size_t>(head_dim));
    std::vector<float> k_cache(static_cast<size_t>(seq_len * head_dim));
    std::vector<float> v_cache(static_cast<size_t>(seq_len * head_dim));

    // Fill with random-ish values
    for (int64_t i = 0; i < head_dim; ++i) q[i] = static_cast<float>(i + 1);
    for (int64_t p = 0; p < seq_len; ++p) {
        for (int64_t d = 0; d < head_dim; ++d) {
            k_cache[static_cast<size_t>(p * head_dim + d)] = static_cast<float>((p + 1) * (d + 1));
            v_cache[static_cast<size_t>(p * head_dim + d)] = static_cast<float>(p + d);
        }
    }

    float out[8] = {};
    std::vector<float> scores(static_cast<size_t>(seq_len));

    AttentionParams params;
    params.q = q.data();
    params.k_cache = k_cache.data();
    params.v_cache = v_cache.data();
    params.out = out;
    params.scores = scores.data();
    params.n_heads = n_heads;
    params.n_kv_heads = n_kv_heads;
    params.head_dim = head_dim;
    params.seq_len = seq_len;
    params.n_rep = n_rep;

    compute_attention(params);

    // Softmax weights should sum to ~1.0
    float weight_sum = 0.0f;
    for (int64_t i = 0; i < seq_len; ++i) {
        weight_sum += scores[static_cast<size_t>(i)];
    }
    EXPECT_NEAR(weight_sum, 1.0f, 1e-5f);
}
