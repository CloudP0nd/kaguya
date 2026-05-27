#pragma once
// Kaguya — weights
// Model weight references (non-owning pointers into GGUF tensor data)

#include <cstdint>
#include <string>
#include <vector>
#include "kaguya/model/hparams.h"

namespace kaguya {

struct TensorInfo;

/// A single transformer layer's weight references
struct LayerWeights {
    // Attention
    const void* attn_norm     = nullptr; size_t attn_norm_bytes = 0;
    const void* wq            = nullptr; size_t wq_bytes = 0;
    const void* wk            = nullptr; size_t wk_bytes = 0;
    const void* wv            = nullptr; size_t wv_bytes = 0;
    const void* wo            = nullptr; size_t wo_bytes = 0;

    // FFN
    const void* ffn_norm      = nullptr; size_t ffn_norm_bytes = 0;
    const void* w_gate        = nullptr; size_t w_gate_bytes = 0;
    const void* w_up          = nullptr; size_t w_up_bytes = 0;
    const void* w_down        = nullptr; size_t w_down_bytes = 0;

    // MoE specific
    const void* moe_gate      = nullptr; size_t moe_gate_bytes = 0;
    std::vector<const void*> expert_w_gate;
    std::vector<const void*> expert_w_up;
    std::vector<const void*> expert_w_down;

    // Tensor shapes for dimension checking
    int64_t wq_ne0 = 0, wq_ne1 = 0;
    int64_t wk_ne0 = 0, wk_ne1 = 0;
    int64_t wv_ne0 = 0, wv_ne1 = 0;
    int64_t wo_ne0 = 0, wo_ne1 = 0;
    int64_t w_gate_ne0 = 0, w_gate_ne1 = 0;
    int64_t w_up_ne0 = 0, w_up_ne1 = 0;
    int64_t w_down_ne0 = 0, w_down_ne1 = 0;
};

/// All model weights
struct ModelWeights {
    // Global
    const void* tok_emb       = nullptr; size_t tok_emb_bytes = 0;
    const void* output_norm   = nullptr; size_t output_norm_bytes = 0;
    const void* output_proj   = nullptr; size_t output_proj_bytes = 0;

    int64_t tok_emb_ne0 = 0, tok_emb_ne1 = 0;
    int64_t output_proj_ne0 = 0, output_proj_ne1 = 0;

    // Per-layer
    std::vector<LayerWeights> layers;

    // Architecture info
    ModelArch arch = ModelArch::UNKNOWN;
    HyperParams hparams;
};

} // namespace kaguya
