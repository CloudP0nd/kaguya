#pragma once
// Kaguya — hparams
// Transformer hyperparameters extracted from GGUF metadata

#include <cstdint>
#include <string>
#include <vector>

namespace kaguya {

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

/// Get architecture enum from string name
ModelArch arch_from_string(const std::string& name);

/// Get human-readable architecture name
const char* arch_to_string(ModelArch arch);

} // namespace kaguya
