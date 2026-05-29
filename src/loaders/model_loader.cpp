// Kaguya — model_loader implementation
// High-level model loader: GGUF -> ModelWeights

#include "kaguya/model_loader.h"
#include "kaguya/gguf_loader.h"

#include <iostream>
#include <algorithm>

namespace kaguya {

ModelLoader::ModelLoader() : gguf_(std::make_unique<GgufLoader>()) {}
ModelLoader::~ModelLoader() = default;

bool ModelLoader::load(const std::string& path, bool mmap) {
    // Step 1: Parse GGUF file
    if (!gguf_->load(path, mmap)) {
        std::cerr << "Kaguya: Failed to load GGUF file: " << path << "\n";
        return false;
    }

    // Step 2: Extract hyperparameters
    if (!extract_hparams()) {
        std::cerr << "Kaguya: Failed to extract hyperparameters\n";
        return false;
    }

    // Step 3: Build weight references
    if (!build_weight_refs()) {
        std::cerr << "Kaguya: Failed to build weight references\n";
        return false;
    }

    // Step 4: Build tokenizer from GGUF metadata
    if (!build_tokenizer()) {
        // Non-fatal: tokenizer may not be present in all GGUF files
        std::cerr << "Kaguya: Warning — could not build BPE tokenizer (byte-level fallback will be used)\n";
    }

    // Transfer to Model
    model_.set_weights(std::move(weights_));

    const auto& hp = model_.hparams();
    std::cout << "Kaguya: Model loaded successfully — "
              << arch_name() << " "
              << hp.emb_dim << "d "
              << hp.num_layers << "L "
              << hp.num_heads << "H\n";

    return true;
}

std::string ModelLoader::arch_name() const {
    return arch_to_string(model_.arch());
}

bool ModelLoader::extract_hparams() {
    auto& hp = weights_.hparams;

    // Get architecture
    auto arch_str = gguf_->metadata_string("general.architecture");
    if (!arch_str) {
        std::cerr << "Kaguya: Missing general.architecture in metadata\n";
        return false;
    }
    weights_.arch = arch_from_string(*arch_str);
    std::string prefix = *arch_str; // e.g., "llama", "qwen2"

    // Extract hyperparameters using the architecture prefix
    // Common keys with fallbacks for different naming conventions
    auto get_int = [&](const std::vector<std::string>& keys) -> std::optional<int64_t> {
        for (const auto& k : keys) {
            if (auto v = gguf_->metadata_int(k)) return v;
        }
        return std::nullopt;
    };

    auto get_float = [&](const std::vector<std::string>& keys) -> std::optional<double> {
        for (const auto& k : keys) {
            if (auto v = gguf_->metadata_float(k)) return v;
        }
        // Also try as int (some GGUF writers store floats as int)
        for (const auto& k : keys) {
            if (auto v = gguf_->metadata_int(k)) return static_cast<double>(*v);
        }
        return std::nullopt;
    };

    // Vocabulary size
    hp.vocab_size = get_int({
        prefix + ".vocab_size",
        "tokenizer.ggml.tokens", // array length gives vocab size
    }).value_or(0);

    // Try to get vocab size from tokenizer array
    if (hp.vocab_size == 0) {
        if (auto* tokens = gguf_->metadata("tokenizer.ggml.tokens")) {
            if (tokens->is_array()) {
                hp.vocab_size = static_cast<int64_t>(tokens->as_array().size());
            }
        }
    }

    // Context length
    hp.context_length = get_int({
        prefix + ".context_length",
        prefix + ".max_position_embeddings",
    }).value_or(0);

    // Embedding dimension
    hp.emb_dim = get_int({
        prefix + ".embedding_length",
        prefix + ".hidden_size",
    }).value_or(0);

    // Number of layers
    hp.num_layers = get_int({
        prefix + ".block_count",
        prefix + ".num_hidden_layers",
    }).value_or(0);

    // Number of attention heads
    hp.num_heads = get_int({
        prefix + ".attention.head_count",
        prefix + ".num_attention_heads",
    }).value_or(0);

    // Number of KV heads
    hp.num_kv_heads = get_int({
        prefix + ".attention.head_count_kv",
        prefix + ".num_key_value_heads",
    }).value_or(hp.num_heads); // Default: same as num_heads (MHA)

    // FFN dimension
    hp.ffn_dim = get_int({
        prefix + ".feed_forward_length",
        prefix + ".intermediate_size",
    }).value_or(0);

    // RoPE base frequency
    hp.rope_freq_base = static_cast<float>(get_float({
        prefix + ".rope.freq_base",
        prefix + ".rope_theta",
    }).value_or(10000.0));

    // RoPE scale
    hp.rope_freq_scale = static_cast<float>(get_float({
        prefix + ".rope.scale_linear",
    }).value_or(1.0));

    // Norm epsilon
    hp.norm_eps = static_cast<float>(get_float({
        prefix + ".attention.layer_norm_rms_epsilon",
        prefix + ".layer_norm_epsilon",
        prefix + ".rms_norm_eps",
    }).value_or(1e-5));

    // MoE
    hp.num_experts = get_int({
        prefix + ".expert_count",
        prefix + ".num_local_experts",
    }).value_or(0);

    hp.num_experts_per_tok = get_int({
        prefix + ".expert_used_count",
        prefix + ".num_experts_per_tok",
    }).value_or(0);

    // Derived values
    hp.head_dim = hp.emb_dim / hp.num_heads;
    hp.n_rot = hp.head_dim;
    hp.n_embd_head_k = hp.head_dim;
    hp.n_embd_head_v = hp.head_dim;
    hp.use_gqa = (hp.num_kv_heads < hp.num_heads);
    hp.n_rep = hp.num_heads / hp.num_kv_heads;

    if (!hp.valid()) {
        std::cerr << "Kaguya: Invalid hyperparameters — "
                  << "vocab=" << hp.vocab_size
                  << " ctx=" << hp.context_length
                  << " emb=" << hp.emb_dim
                  << " layers=" << hp.num_layers
                  << " heads=" << hp.num_heads << "\n";
        return false;
    }

    return true;
}

const GgufTensorInfo* ModelLoader::find_tensor(const std::vector<std::string>& names) {
    for (const auto& name : names) {
        if (auto* ti = gguf_->tensor_info(name)) return ti;
    }
    return nullptr;
}

const void* ModelLoader::get_tensor(const std::string& name, size_t& out_bytes,
                                      int64_t& ne0, int64_t& ne1) {
    const GgufTensorInfo* ti = gguf_->tensor_info(name);
    if (!ti) return nullptr;

    out_bytes = ggml_nbytes(*ti);
    ne0 = (ti->n_dims > 0) ? static_cast<int64_t>(ti->dims[0]) : 0;
    ne1 = (ti->n_dims > 1) ? static_cast<int64_t>(ti->dims[1]) : 1;

    return gguf_->tensor_data(*ti);
}

bool ModelLoader::build_weight_refs() {
    const auto& hp = weights_.hparams;

    // --- Global weights ---

    // Token embedding
    if (auto* p = find_tensor({"token_embd.weight", "model.embed_tokens.weight"})) {
        weights_.tok_emb = gguf_->tensor_data(*p);
        weights_.tok_emb_bytes = ggml_nbytes(*p);
        weights_.tok_emb_ne0 = (p->n_dims > 0) ? p->dims[0] : 0;
        weights_.tok_emb_ne1 = (p->n_dims > 1) ? p->dims[1] : 1;
        weights_.tok_emb_dtype = ggml_to_data_type(p->type);
    }

    // Output norm
    if (auto* p = find_tensor({"output_norm.weight", "model.norm.weight"})) {
        weights_.output_norm = gguf_->tensor_data(*p);
        weights_.output_norm_bytes = ggml_nbytes(*p);
        weights_.output_norm_dtype = ggml_to_data_type(p->type);
    }

    // Output projection (may share with tok_emb for tied weights)
    if (auto* p = find_tensor({"output.weight", "lm_head.weight"})) {
        weights_.output_proj = gguf_->tensor_data(*p);
        weights_.output_proj_bytes = ggml_nbytes(*p);
        weights_.output_proj_ne0 = (p->n_dims > 0) ? p->dims[0] : 0;
        weights_.output_proj_ne1 = (p->n_dims > 1) ? p->dims[1] : 1;
        weights_.output_proj_dtype = ggml_to_data_type(p->type);
    }

    // --- Per-layer weights ---
    weights_.layers.resize(hp.num_layers);

    for (int64_t i = 0; i < hp.num_layers; ++i) {
        auto& lw = weights_.layers[i];

        // Build name prefixes for this layer
        std::string blk_idx = std::to_string(i);
        std::vector<std::string> prefixes = {
            "blk." + blk_idx + ".",
            "layers." + blk_idx + ".",
            "model.layers." + blk_idx + ".",
        };

        auto find_layer_tensor = [&](const std::vector<std::string>& suffixes) -> const GgufTensorInfo* {
            for (const auto& pf : prefixes) {
                for (const auto& suf : suffixes) {
                    if (auto* ti = gguf_->tensor_info(pf + suf)) return ti;
                }
            }
            return nullptr;
        };

        auto set_weight = [&](const void*& ptr, size_t& bytes,
                               int64_t& ne0, int64_t& ne1,
                               const std::vector<std::string>& suffixes,
                               DataType& dtype) {
            if (auto* ti = find_layer_tensor(suffixes)) {
                ptr = gguf_->tensor_data(*ti);
                bytes = ggml_nbytes(*ti);
                ne0 = (ti->n_dims > 0) ? static_cast<int64_t>(ti->dims[0]) : 0;
                ne1 = (ti->n_dims > 1) ? static_cast<int64_t>(ti->dims[1]) : 1;
                dtype = ggml_to_data_type(ti->type);
            }
        };

        // Attention norm
        {
            int64_t dummy_ne0 = 0, dummy_ne1 = 0;
            set_weight(lw.attn_norm, lw.attn_norm_bytes,
                        dummy_ne0, dummy_ne1,
                        {"attn_norm.weight", "input_layernorm.weight"},
                        lw.attn_norm_dtype);
        }

        // Q, K, V projections
        set_weight(lw.wq, lw.wq_bytes, lw.wq_ne0, lw.wq_ne1,
                   {"attn_q.weight", "self_attn.q_proj.weight"}, lw.wq_dtype);
        set_weight(lw.wk, lw.wk_bytes, lw.wk_ne0, lw.wk_ne1,
                   {"attn_k.weight", "self_attn.k_proj.weight"}, lw.wk_dtype);
        set_weight(lw.wv, lw.wv_bytes, lw.wv_ne0, lw.wv_ne1,
                   {"attn_v.weight", "self_attn.v_proj.weight"}, lw.wv_dtype);
        set_weight(lw.wo, lw.wo_bytes, lw.wo_ne0, lw.wo_ne1,
                   {"attn_output.weight", "self_attn.o_proj.weight"}, lw.wo_dtype);

        // FFN norm
        {
            int64_t dummy_ne0 = 0, dummy_ne1 = 0;
            set_weight(lw.ffn_norm, lw.ffn_norm_bytes,
                        dummy_ne0, dummy_ne1,
                        {"ffn_norm.weight", "post_attention_layernorm.weight"},
                        lw.ffn_norm_dtype);
        }

        // FFN gate, up, down
        set_weight(lw.w_gate, lw.w_gate_bytes, lw.w_gate_ne0, lw.w_gate_ne1,
                   {"ffn_gate.weight", "mlp.gate_proj.weight"}, lw.w_gate_dtype);
        set_weight(lw.w_up, lw.w_up_bytes, lw.w_up_ne0, lw.w_up_ne1,
                   {"ffn_up.weight", "mlp.up_proj.weight"}, lw.w_up_dtype);
        set_weight(lw.w_down, lw.w_down_bytes, lw.w_down_ne0, lw.w_down_ne1,
                   {"ffn_down.weight", "mlp.down_proj.weight"}, lw.w_down_dtype);

        // MoE gate (router)
        if (hp.num_experts > 0) {
            {
                int64_t dummy_ne0 = 0, dummy_ne1 = 0;
                set_weight(lw.moe_gate, lw.moe_gate_bytes, dummy_ne0, dummy_ne1,
                           {"ffn_gate_exps.weight", "block_sparse_moe.gate.weight"},
                           lw.w_gate_dtype); // reuse w_gate_dtype for moe_gate
            }

            // Expert weights
            lw.expert_w_gate.resize(hp.num_experts);
            lw.expert_w_up.resize(hp.num_experts);
            lw.expert_w_down.resize(hp.num_experts);
            for (int64_t e = 0; e < hp.num_experts; ++e) {
                std::string eidx = std::to_string(e);
                if (auto* ti = find_layer_tensor({
                    "ffn_gate_exps." + eidx + ".weight",
                    "block_sparse_moe.experts." + eidx + ".w1.weight",
                })) {
                    lw.expert_w_gate[e] = gguf_->tensor_data(*ti);
                }
                if (auto* ti = find_layer_tensor({
                    "ffn_up_exps." + eidx + ".weight",
                    "block_sparse_moe.experts." + eidx + ".w3.weight",
                })) {
                    lw.expert_w_up[e] = gguf_->tensor_data(*ti);
                }
                if (auto* ti = find_layer_tensor({
                    "ffn_down_exps." + eidx + ".weight",
                    "block_sparse_moe.experts." + eidx + ".w2.weight",
                })) {
                    lw.expert_w_down[e] = gguf_->tensor_data(*ti);
                }
            }
        }
    }

    // Validate essential weights
    if (!weights_.tok_emb) {
        std::cerr << "Kaguya: Warning — token embedding not found\n";
    }
    if (weights_.layers.empty()) {
        std::cerr << "Kaguya: No layer weights found\n";
        return false;
    }
    if (!weights_.layers[0].wq) {
        std::cerr << "Kaguya: Warning — attention Q weights not found for layer 0\n";
    }

    return true;
}

void ModelLoader::print_info() const {
    gguf_->print_summary();

    const auto& hp = model_.hparams();
    std::cout << "\n--- Model Structure ---\n";
    std::cout << "Architecture:   " << arch_name() << "\n";
    std::cout << "Vocab size:     " << hp.vocab_size << "\n";
    std::cout << "Context length: " << hp.context_length << "\n";
    std::cout << "Embedding dim:  " << hp.emb_dim << "\n";
    std::cout << "Layers:         " << hp.num_layers << "\n";
    std::cout << "Attention heads: " << hp.num_heads << "\n";
    std::cout << "KV heads:       " << hp.num_kv_heads << "\n";
    if (hp.use_gqa) {
        std::cout << "GQA:            YES (rep=" << hp.n_rep << ")\n";
    }
    std::cout << "Head dim:       " << hp.head_dim << "\n";
    if (hp.ffn_dim > 0) {
        std::cout << "FFN dim:        " << hp.ffn_dim << "\n";
    }
    std::cout << "RoPE freq base: " << hp.rope_freq_base << "\n";
    std::cout << "Norm eps:       " << hp.norm_eps << "\n";
    if (hp.num_experts > 0) {
        std::cout << "MoE experts:    " << hp.num_experts
                  << " (top-" << hp.num_experts_per_tok << ")\n";
    }

    // Output projection
    std::cout << "\nTied embeddings: " << (weights_.output_proj ? "no" : "yes") << "\n";

    // Tokenizer info
    if (tokenizer_.is_valid()) {
        std::cout << "\nTokenizer: BPE (vocab_size=" << tokenizer_.vocab_size() << ")\n";
        if (tokenizer_.bos_token_id() >= 0) {
            std::cout << "  BOS token: " << tokenizer_.bos_token_id()
                      << " (" << tokenizer_.token_text(tokenizer_.bos_token_id()) << ")\n";
        }
        if (tokenizer_.eos_token_id() >= 0) {
            std::cout << "  EOS token: " << tokenizer_.eos_token_id()
                      << " (" << tokenizer_.token_text(tokenizer_.eos_token_id()) << ")\n";
        }
    } else {
        std::cout << "\nTokenizer: byte-level (no BPE metadata found)\n";
    }
}

bool ModelLoader::build_tokenizer() {
    // Read tokenizer.ggml.tokens (required)
    std::vector<std::string> tokens;
    {
        auto* meta = gguf_->metadata("tokenizer.ggml.tokens");
        if (!meta || !meta->is_array()) {
            return false;
        }
        const auto& arr = meta->as_array();
        tokens.reserve(arr.size());
        for (const auto& item : arr) {
            if (item.is_string()) {
                tokens.push_back(item.as_string());
            } else {
                tokens.push_back("");
            }
        }
    }

    if (tokens.empty()) return false;

    // Read tokenizer.ggml.scores (optional)
    std::vector<float> scores;
    {
        auto* meta = gguf_->metadata("tokenizer.ggml.scores");
        if (meta && meta->is_array()) {
            const auto& arr = meta->as_array();
            scores.reserve(arr.size());
            for (const auto& item : arr) {
                scores.push_back(static_cast<float>(item.as_float()));
            }
        }
    }

    // Read tokenizer.ggml.token_type (optional)
    std::vector<int32_t> token_types;
    {
        auto* meta = gguf_->metadata("tokenizer.ggml.token_type");
        if (meta && meta->is_array()) {
            const auto& arr = meta->as_array();
            token_types.reserve(arr.size());
            for (const auto& item : arr) {
                token_types.push_back(static_cast<int32_t>(item.as_int()));
            }
        }
    }

    // Read tokenizer.ggml.merges (optional — only for BPE models)
    std::vector<std::string> merges;
    {
        auto* meta = gguf_->metadata("tokenizer.ggml.merges");
        if (meta && meta->is_array()) {
            const auto& arr = meta->as_array();
            merges.reserve(arr.size());
            for (const auto& item : arr) {
                if (item.is_string()) {
                    merges.push_back(item.as_string());
                }
            }
        }
    }

    // Read BOS/EOS token IDs from metadata
    int32_t bos_id = -1;
    int32_t eos_id = -1;

    if (auto v = gguf_->metadata_int("tokenizer.ggml.bos_token_id")) {
        bos_id = static_cast<int32_t>(*v);
    }
    if (auto v = gguf_->metadata_int("tokenizer.ggml.eos_token_id")) {
        eos_id = static_cast<int32_t>(*v);
    }

    // Determine add_bos from architecture
    // Most models add BOS automatically, but some (like Gemma) don't
    bool add_bos = true;
    if (weights_.arch == ModelArch::GEMMA) {
        // Gemma models typically include BOS in the prompt template
        add_bos = false;
    }

    // Check for add_bos_token metadata
    if (auto v = gguf_->metadata_int("tokenizer.ggml.add_bos_token")) {
        add_bos = (*v != 0);
    }

    // Build the tokenizer
    return tokenizer_.build(tokens, scores, token_types, merges, bos_id, eos_id, add_bos);
}

} // namespace kaguya
