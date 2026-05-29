#include "kaguya/pipeline.h"
#include "kaguya/attention.h"
#include "kaguya/kernels/special_ops.h"
#include "kaguya/kernels/quantize.h"
#include "kaguya/kernels/gemm.h"
#include "kaguya/cpu_features.h"
#include "kaguya/memory_manager.h"

#include <cstring>
#include <cmath>
#include <algorithm>

namespace kaguya {

Pipeline::Pipeline(const Model& model)
    : model_(model),
      kv_cache_(model.hparams().num_layers,
                model.hparams().num_kv_heads,
                model.hparams().head_dim,
                model.hparams().context_length),
      pos_(0)
{
    const auto& hp = model.hparams();

    // Pre-allocate working buffers
    logits_.resize(static_cast<size_t>(hp.vocab_size), 0.0f);
    xb_.resize(static_cast<size_t>(hp.emb_dim), 0.0f);
    xb2_.resize(static_cast<size_t>(hp.emb_dim), 0.0f);
    q_.resize(static_cast<size_t>(hp.num_heads * hp.head_dim), 0.0f);
    key_tmp_.resize(static_cast<size_t>(hp.num_kv_heads * hp.head_dim), 0.0f);
    val_tmp_.resize(static_cast<size_t>(hp.num_kv_heads * hp.head_dim), 0.0f);
    att_scores_.resize(static_cast<size_t>(hp.num_heads * hp.context_length), 0.0f);
    hb_.resize(static_cast<size_t>(hp.ffn_dim), 0.0f);
    hb2_.resize(static_cast<size_t>(hp.ffn_dim), 0.0f);
    attn_proj_.resize(static_cast<size_t>(hp.emb_dim), 0.0f);
    ffn_down_.resize(static_cast<size_t>(hp.emb_dim), 0.0f);

    // Weight buffer: largest possible weight (FFN up/gate: ffn_dim * emb_dim)
    const int64_t max_weight_elements = std::max(hp.ffn_dim * hp.emb_dim, hp.vocab_size * hp.emb_dim);
    weight_buf_.resize(static_cast<size_t>(max_weight_elements), 0.0f);
}

void Pipeline::reset() {
    pos_ = 0;
    kv_cache_.reset();
    recent_tokens_.clear();
    std::fill(logits_.begin(), logits_.end(), 0.0f);
}

float* Pipeline::dequantize_weight(const void* data, DataType dtype, int64_t n_elements) {
    if (dtype == DataType::F32) {
        return const_cast<float*>(static_cast<const float*>(data));
    }
    // Dequantize to scratch buffer
    if (n_elements > static_cast<int64_t>(weight_buf_.size())) {
        weight_buf_.resize(static_cast<size_t>(n_elements), 0.0f);
    }
    kernels::dequantize_dispatch(dtype, data, weight_buf_.data(), n_elements);
    return weight_buf_.data();
}

void Pipeline::weighted_project(float* y,
                                 const void* w_data, DataType w_dtype,
                                 int64_t w_ne0, int64_t w_ne1,
                                 const float* x) {
    // GEMV: y = W * x^T where W is [w_ne1, w_ne0] in row-major
    // ne0 = input dim (K), ne1 = output dim (M)
    const int64_t M = w_ne1;
    const int64_t K = w_ne0;

    if (M == 0 || K == 0) return;

    float* w_fp32 = dequantize_weight(w_data, w_dtype, M * K);

    kernels::GemmParams params;
    params.M = M;
    params.K = K;
    params.N = 1;
    params.A = w_fp32;
    params.lda = K;
    params.B = x;   // [K] treated as [K, 1]
    params.ldb = 1;
    params.C = y;   // [M] treated as [M, 1]
    params.ldc = 1;
    params.alpha = 1.0f;
    params.beta = 0.0f;
    kernels::gemm_dispatch(params);
}

void Pipeline::forward_layer(int64_t layer, float* x, int64_t pos) {
    const auto& hp = model_.hparams();
    const auto& lw = model_.layer(layer);

    // ---- Attention ----

    // RMSNorm on input
    kernels::rmsnorm_dispatch(xb_.data(), x,
                               static_cast<const float*>(lw.attn_norm),
                               hp.emb_dim, hp.norm_eps);

    // QKV projections
    weighted_project(q_.data(), lw.wq, lw.wq_dtype, lw.wq_ne0, lw.wq_ne1, xb_.data());
    weighted_project(key_tmp_.data(), lw.wk, lw.wk_dtype, lw.wk_ne0, lw.wk_ne1, xb_.data());
    weighted_project(val_tmp_.data(), lw.wv, lw.wv_dtype, lw.wv_ne0, lw.wv_ne1, xb_.data());

    // Apply RoPE to Q and K
    kernels::rope_dispatch(q_.data(), hp.num_heads, hp.head_dim, pos,
                            hp.rope_freq_base, hp.rope_freq_scale);
    kernels::rope_dispatch(key_tmp_.data(), hp.num_kv_heads, hp.head_dim, pos,
                            hp.rope_freq_base, hp.rope_freq_scale);

    // Store K and V into cache
    for (int64_t h = 0; h < hp.num_kv_heads; ++h) {
        const float* k_src = key_tmp_.data() + h * hp.head_dim;
        float* k_dst = kv_cache_.key(layer, h, pos);
        std::memcpy(k_dst, k_src, static_cast<size_t>(hp.head_dim) * sizeof(float));

        const float* v_src = val_tmp_.data() + h * hp.head_dim;
        float* v_dst = kv_cache_.value(layer, h, pos);
        std::memcpy(v_dst, v_src, static_cast<size_t>(hp.head_dim) * sizeof(float));
    }

    // Compute attention
    const int64_t seq_len = pos + 1; // number of valid positions in cache

    AttentionParams attn_params;
    attn_params.q = q_.data();
    attn_params.out = xb2_.data();
    attn_params.scores = att_scores_.data();
    attn_params.n_heads = hp.num_heads;
    attn_params.n_kv_heads = hp.num_kv_heads;
    attn_params.head_dim = hp.head_dim;
    attn_params.seq_len = seq_len;
    attn_params.n_rep = hp.n_rep;

    // Stride-aware attention: pass KV cache directly without copying.
    // The KV cache layout is [n_kv_heads, max_seq_len, head_dim] per layer,
    // with head_stride = max_seq_len * head_dim between KV heads.
    // compute_attention uses kv_stride to index heads correctly.
    attn_params.k_cache = kv_cache_.key_head(layer, 0);
    attn_params.v_cache = kv_cache_.value_head(layer, 0);
    attn_params.kv_stride = kv_cache_.head_stride(); // max_seq_len * head_dim

    compute_attention(attn_params);

    // Output projection — compute wo * attn_out into pre-allocated buffer, then add residual
    std::fill(attn_proj_.begin(), attn_proj_.end(), 0.0f);
    weighted_project(attn_proj_.data(), lw.wo, lw.wo_dtype, lw.wo_ne0, lw.wo_ne1, xb2_.data());

    // Residual connection: x = x + wo * attn_out
    for (int64_t i = 0; i < hp.emb_dim; ++i) {
        x[i] = x[i] + attn_proj_[static_cast<size_t>(i)];
    }

    // ---- FFN ----

    // RMSNorm
    kernels::rmsnorm_dispatch(xb_.data(), x,
                               static_cast<const float*>(lw.ffn_norm),
                               hp.emb_dim, hp.norm_eps);

    // Gate and Up projections
    weighted_project(hb_.data(), lw.w_gate, lw.w_gate_dtype, lw.w_gate_ne0, lw.w_gate_ne1, xb_.data());
    weighted_project(hb2_.data(), lw.w_up, lw.w_up_dtype, lw.w_up_ne0, lw.w_up_ne1, xb_.data());

    // SiLU activation on gate
    kernels::silu_dispatch(hb_.data(), hp.ffn_dim);

    // Element-wise multiply: gate * up
    for (int64_t i = 0; i < hp.ffn_dim; ++i) {
        hb_[static_cast<size_t>(i)] *= hb2_[static_cast<size_t>(i)];
    }

    // Down projection — use pre-allocated buffer
    std::fill(ffn_down_.begin(), ffn_down_.end(), 0.0f);
    weighted_project(ffn_down_.data(), lw.w_down, lw.w_down_dtype, lw.w_down_ne0, lw.w_down_ne1, hb_.data());

    // Residual connection: x = x + ffn_down
    for (int64_t i = 0; i < hp.emb_dim; ++i) {
        x[i] = x[i] + ffn_down_[static_cast<size_t>(i)];
    }
}

void Pipeline::forward(int64_t pos, int32_t token) {
    const auto& hp = model_.hparams();
    const auto& weights = model_.weights();

    // Step 1: Token embedding lookup
    // tok_emb is [vocab_size, emb_dim] in row-major (ne0=emb_dim, ne1=vocab_size)
    // Embedding for token i: tok_emb + i * emb_dim
    if (weights.tok_emb) {
        float* emb = dequantize_weight(weights.tok_emb, weights.tok_emb_dtype,
                                        hp.vocab_size * hp.emb_dim);
        const int64_t emb_stride = weights.tok_emb_ne0; // emb_dim
        if (token >= 0 && token < hp.vocab_size) {
            std::memcpy(xb_.data(), emb + token * emb_stride,
                        static_cast<size_t>(hp.emb_dim) * sizeof(float));
        } else {
            std::fill(xb_.begin(), xb_.end(), 0.0f);
        }
    } else {
        std::fill(xb_.begin(), xb_.end(), 0.0f);
    }

    // Copy embedding to working buffer (x)
    std::memcpy(xb2_.data(), xb_.data(), static_cast<size_t>(hp.emb_dim) * sizeof(float));

    // Step 2: Process each transformer layer
    for (int64_t layer = 0; layer < hp.num_layers; ++layer) {
        // Swap: x is the input to this layer, becomes output
        forward_layer(layer, xb2_.data(), pos);
    }

    // Step 3: Output RMSNorm
    kernels::rmsnorm_dispatch(xb_.data(), xb2_.data(),
                               static_cast<const float*>(weights.output_norm),
                               hp.emb_dim, hp.norm_eps);

    // Step 4: Output projection (logits)
    if (weights.output_proj) {
        weighted_project(logits_.data(), weights.output_proj, weights.output_proj_dtype,
                          weights.output_proj_ne0, weights.output_proj_ne1, xb_.data());
    } else if (weights.tok_emb) {
        // Tied embeddings: logits = x @ tok_emb^T
        // tok_emb is [vocab_size, emb_dim], so tok_emb^T is [emb_dim, vocab_size]
        // But we can use weighted_project with tok_emb as the weight
        // weighted_project does y = W * x^T where W is [ne1, ne0]
        // For tied embeddings: W = tok_emb [vocab_size, emb_dim], so ne0=emb_dim, ne1=vocab_size
        weighted_project(logits_.data(), weights.tok_emb, weights.tok_emb_dtype,
                          weights.tok_emb_ne0, weights.tok_emb_ne1, xb_.data());
    } else {
        std::fill(logits_.begin(), logits_.end(), 0.0f);
    }
}

void Pipeline::prefill(const std::vector<int32_t>& tokens) {
    if (tokens.empty()) return;  // Handle empty token list gracefully
    for (int32_t token : tokens) {
        if (pos_ >= model_.hparams().context_length) break;
        forward(pos_, token);
        kv_cache_.advance(1);
        pos_++;
        recent_tokens_.push_back(token);
    }
}

int32_t Pipeline::decode(Sampler& sampler) {
    // Check if context is full
    if (pos_ >= model_.hparams().context_length) {
        return -1;  // End of context
    }

    // Forward pass at current position (the last token is already in recent_tokens_)
    int32_t last_token = recent_tokens_.empty() ? 0 : recent_tokens_.back();
    forward(pos_, last_token);

    // Sample next token
    int32_t next_token = sampler.sample(logits_.data(),
                                         static_cast<int64_t>(logits_.size()),
                                         recent_tokens_);

    kv_cache_.advance(1);
    pos_++;
    recent_tokens_.push_back(next_token);

    return next_token;
}

std::vector<int32_t> Pipeline::generate(const std::vector<int32_t>& prompt,
                                          int32_t n_predict,
                                          Sampler& sampler) {
    reset();
    prefill(prompt);

    std::vector<int32_t> output;
    output.reserve(static_cast<size_t>(n_predict));

    for (int32_t i = 0; i < n_predict; ++i) {
        if (pos_ >= model_.hparams().context_length) break;

        int32_t token = decode(sampler);
        output.push_back(token);

        // Check for EOS
        if (eos_token_id_ >= 0 && token == eos_token_id_) break;
    }

    return output;
}

std::vector<int32_t> Pipeline::generate(const std::vector<int32_t>& prompt,
                                          int32_t n_predict,
                                          Sampler& sampler,
                                          const GenerationStop& stop) {
    // Set EOS token from stop config
    int32_t prev_eos = eos_token_id_;
    if (stop.eos_token_id >= 0) {
        eos_token_id_ = stop.eos_token_id;
    }

    auto result = generate(prompt, n_predict, sampler);

    // Restore previous EOS setting
    eos_token_id_ = prev_eos;
    return result;
}

} // namespace kaguya
