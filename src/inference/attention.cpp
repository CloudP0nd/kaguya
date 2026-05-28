#include "kaguya/attention.h"
#include "kaguya/kernels/special_ops.h"
#include "kaguya/kernels/gemm.h"

#include <cstring>
#include <cmath>

namespace kaguya {

void compute_attention(const AttentionParams& params) {
    const int64_t n_heads = params.n_heads;
    const int64_t head_dim = params.head_dim;
    const int64_t seq_len = params.seq_len;
    const int64_t n_rep = params.n_rep;

    for (int64_t h = 0; h < n_heads; ++h) {
        const int64_t kv_h = h / n_rep; // GQA: map query head to KV head

        const float* q_h = params.q + h * head_dim;         // [head_dim]
        const float* k_h = params.k_cache + kv_h * params.kv_stride; // [max_seq_len, head_dim] — only first seq_len rows are valid
        const float* v_h = params.v_cache + kv_h * params.kv_stride; // [max_seq_len, head_dim] — only first seq_len rows are valid
        float* scores_h = params.scores + h * seq_len;       // [seq_len]
        float* out_h = params.out + h * head_dim;            // [head_dim]

        // Step 1: Compute attention scores = k_h @ q_h^T (GEMV)
        // scores^T = k_h * q_h^T: A=k_h[seq_len, head_dim], B=q_h^T[head_dim, 1], C=scores^T[seq_len, 1]
        if (seq_len > 0) {
            kernels::GemmParams gp;
            gp.M = seq_len;
            gp.K = head_dim;
            gp.N = 1;
            gp.A = k_h;
            gp.lda = head_dim;
            gp.B = q_h;     // [head_dim] treated as [head_dim, 1] with ldb=1
            gp.ldb = 1;
            gp.C = scores_h; // [seq_len] treated as [seq_len, 1] with ldc=1
            gp.ldc = 1;
            gp.alpha = 1.0f / std::sqrt(static_cast<float>(head_dim)); // Scale by 1/sqrt(d)
            gp.beta = 0.0f;
            kernels::gemm_dispatch(gp);
        }

        // Step 2: Softmax over attention scores
        kernels::softmax_dispatch(scores_h, seq_len);

        // Step 3: Weighted sum = scores @ v_h (GEMV)
        // out_h = scores * v_h: A=scores[1, seq_len], B=v_h[seq_len, head_dim], C=out_h[1, head_dim]
        if (seq_len > 0) {
            kernels::GemmParams gp;
            gp.M = 1;
            gp.K = seq_len;
            gp.N = head_dim;
            gp.A = scores_h;
            gp.lda = seq_len;
            gp.B = v_h;
            gp.ldb = head_dim;
            gp.C = out_h;
            gp.ldc = head_dim;
            gp.alpha = 1.0f;
            gp.beta = 0.0f;
            kernels::gemm_dispatch(gp);
        }
    }
}

} // namespace kaguya
