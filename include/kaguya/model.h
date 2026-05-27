#pragma once

#include <string>
#include <vector>
#include <unordered_map>
#include <memory>
#include "kaguya/tensor.h"

namespace kaguya {

// Forward declaration
struct GgufTensorInfo;

/// Supported model architectures
enum class ModelArch {
    LLAMA,
    QWEN2,
    MISTRAL,
    MIXTRAL,
    PHI3,
    GEMMA,
    FALCON,
    DEEPSEEK,
    COMMAND_R,
    UNKNOWN,
};

/// Get architecture enum from string name
ModelArch arch_from_string(const std::string& name);

/// Get human-readable architecture name
const char* arch_to_string(ModelArch arch);

/// Transformer hyperparameters
struct HyperParams {
    int64_t vocab_size         = 0;
    int64_t context_length     = 0;
    int64_t emb_dim            = 0;
    int64_t num_layers         = 0;
    int64_t num_heads          = 0;
    int64_t num_kv_heads       = 0;
    int64_t head_dim           = 0;
    int64_t ffn_dim            = 0;
    int64_t num_experts        = 0;     ///< 0 = dense, >0 = MoE
    int64_t num_experts_per_tok = 0;
    float   rope_freq_base     = 10000.0f;
    float   rope_freq_scale    = 1.0f;
    float   norm_eps           = 1e-5f;
    int64_t n_rot              = 0;     ///< Rotary dimensions (usually head_dim)
    int64_t n_embd_head_k      = 0;     ///< Per-head key dim
    int64_t n_embd_head_v      = 0;     ///< Per-head value dim
    bool    use_gqa            = false;  ///< Grouped-query attention
    int64_t n_rep              = 1;     ///< KV head repetition factor

    /// Validate hyperparameters
    bool valid() const {
        return vocab_size > 0 && context_length > 0 && emb_dim > 0 &&
               num_layers > 0 && num_heads > 0;
    }
};

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

/// All model weights (non-owning pointers into GGUF tensor data)
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

/// Loaded model
class Model {
public:
    Model();
    ~Model() = default;

    /// Load from GGUF file path
    bool load(const std::string& path);

    /// Model architecture
    ModelArch arch() const { return weights_.arch; }

    /// Hyperparameters
    const HyperParams& hparams() const { return weights_.hparams; }

    /// Model weights
    const ModelWeights& weights() const { return weights_; }

    /// Token embedding
    const void* tok_emb() const { return weights_.tok_emb; }

    /// Output norm
    const void* output_norm() const { return weights_.output_norm; }

    /// Output projection (may share with tok_emb)
    const void* output_proj() const { return weights_.output_proj; }

    /// Layer weights
    const LayerWeights& layer(int i) const { return weights_.layers[i]; }

    /// Number of layers
    int num_layers() const { return static_cast<int>(weights_.layers.size()); }

    /// Lookup tensor info by name (for GGUF loader compatibility)
    const GgufTensorInfo* tensor_info(const std::string& name) const;

    /// Set internal data (called by ModelLoader)
    void set_weights(ModelWeights&& weights) { weights_ = std::move(weights); }

private:
    ModelWeights weights_;
};

} // namespace kaguya
