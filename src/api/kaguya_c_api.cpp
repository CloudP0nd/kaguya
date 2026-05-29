#include "kaguya/kaguya.h"
#include "kaguya/model.h"
#include "kaguya/model_loader.h"
#include "kaguya/tokenizer.h"
#include "kaguya/pipeline.h"
#include "kaguya/sampling.h"
#include "kaguya/cpu_features.h"
#include "kaguya/memory_manager.h"

#include <cstring>
#include <cstdlib>
#include <memory>
#include <string>
#include <vector>

// ============================================================================
// Internal structures
// ============================================================================

struct kaguya_model {
    kaguya::ModelLoader* loader;
    std::string arch_name_cache;
};

struct kaguya_context {
    const kaguya_model* model;
    kaguya::Pipeline* pipeline;
};

// ============================================================================
// Internal tokenizer — uses BPE if available, falls back to byte-level
// ============================================================================

static std::vector<int32_t> internal_encode(const kaguya_model* model, const std::string& text) {
    if (model && model->loader) {
        const auto& tok = model->loader->tokenizer();
        if (tok.is_valid()) {
            return tok.encode(text);
        }
    }
    // Fallback: byte-level
    std::vector<int32_t> tokens;
    tokens.reserve(text.size());
    for (unsigned char c : text) {
        tokens.push_back(static_cast<int32_t>(c));
    }
    return tokens;
}

static std::string internal_decode(const kaguya_model* model, const int32_t* tokens, int64_t count) {
    if (model && model->loader) {
        const auto& tok = model->loader->tokenizer();
        if (tok.is_valid()) {
            std::vector<int32_t> vec(tokens, tokens + count);
            return tok.decode(vec, true);
        }
    }
    // Fallback: byte-level
    std::string result;
    result.reserve(static_cast<size_t>(count));
    for (int64_t i = 0; i < count; ++i) {
        if (tokens[i] >= 0 && tokens[i] < 256) {
            result += static_cast<char>(tokens[i]);
        }
    }
    return result;
}

// ============================================================================
// Version info
// ============================================================================

const char* kaguya_version(void) {
    return "0.2.0";
}

// ============================================================================
// Model management
// ============================================================================

kaguya_model* kaguya_model_load(const char* path) {
    if (!path) return nullptr;

    auto* m = new (std::nothrow) kaguya_model();
    if (!m) return nullptr;

    m->loader = new (std::nothrow) kaguya::ModelLoader();
    if (!m->loader) {
        delete m;
        return nullptr;
    }

    if (!m->loader->load(path)) {
        delete m->loader;
        delete m;
        return nullptr;
    }

    m->arch_name_cache = m->loader->arch_name();
    return m;
}

void kaguya_model_free(kaguya_model* model) {
    if (model) {
        delete model->loader;
        delete model;
    }
}

const char* kaguya_model_arch(const kaguya_model* model) {
    if (!model) return nullptr;
    return model->arch_name_cache.c_str();
}

int64_t kaguya_model_vocab_size(const kaguya_model* model) {
    if (!model) return 0;
    return model->loader->model().hparams().vocab_size;
}

int64_t kaguya_model_context_length(const kaguya_model* model) {
    if (!model) return 0;
    return model->loader->model().hparams().context_length;
}

int64_t kaguya_model_emb_dim(const kaguya_model* model) {
    if (!model) return 0;
    return model->loader->model().hparams().emb_dim;
}

int64_t kaguya_model_num_layers(const kaguya_model* model) {
    if (!model) return 0;
    return model->loader->model().hparams().num_layers;
}

int64_t kaguya_model_num_heads(const kaguya_model* model) {
    if (!model) return 0;
    return model->loader->model().hparams().num_heads;
}

// ============================================================================
// Inference context
// ============================================================================

kaguya_context* kaguya_context_create(const kaguya_model* model) {
    if (!model) return nullptr;

    auto* ctx = new (std::nothrow) kaguya_context();
    if (!ctx) return nullptr;

    ctx->model = model;
    ctx->pipeline = new (std::nothrow) kaguya::Pipeline(model->loader->model());
    if (!ctx->pipeline) {
        delete ctx;
        return nullptr;
    }

    return ctx;
}

void kaguya_context_free(kaguya_context* ctx) {
    if (ctx) {
        delete ctx->pipeline;
        delete ctx;
    }
}

void kaguya_context_reset(kaguya_context* ctx) {
    if (ctx && ctx->pipeline) {
        ctx->pipeline->reset();
    }
}

int64_t kaguya_context_position(const kaguya_context* ctx) {
    if (!ctx || !ctx->pipeline) return -1;
    return ctx->pipeline->current_pos();
}

// ============================================================================
// Tokenization
// ============================================================================

int kaguya_tokenize(const kaguya_context* ctx,
                    const char* text,
                    int32_t** out_tokens,
                    int64_t* out_count)
{
    if (!text || !out_tokens || !out_count) return -1;

    auto tokens = internal_encode(ctx ? ctx->model : nullptr, text);
    *out_count = static_cast<int64_t>(tokens.size());

    if (tokens.empty()) {
        *out_tokens = nullptr;
        return 0;
    }

    *out_tokens = static_cast<int32_t*>(std::malloc(tokens.size() * sizeof(int32_t)));
    if (!*out_tokens) return -1;

    std::memcpy(*out_tokens, tokens.data(), tokens.size() * sizeof(int32_t));
    return 0;
}

char* kaguya_detokenize(const kaguya_context* ctx,
                         const int32_t* tokens,
                         int64_t count)
{
    if (!tokens || count <= 0) return nullptr;

    std::string text = internal_decode(ctx ? ctx->model : nullptr, tokens, count);
    char* result = static_cast<char*>(std::malloc(text.size() + 1));
    if (!result) return nullptr;

    std::memcpy(result, text.c_str(), text.size() + 1);
    return result;
}

void kaguya_tokens_free(int32_t* tokens) {
    std::free(tokens);
}

void kaguya_text_free(char* text) {
    std::free(text);
}

// ============================================================================
// Inference
// ============================================================================

int kaguya_context_prompt_tokens(kaguya_context* ctx,
                                  const int32_t* tokens,
                                  int64_t count)
{
    if (!ctx || !ctx->pipeline) return -1;
    if (!tokens && count > 0) return -1;

    std::vector<int32_t> token_vec(tokens, tokens + count);
    ctx->pipeline->prefill(token_vec);
    return 0;
}

int32_t kaguya_context_decode(kaguya_context* ctx,
                               float temperature,
                               int top_k,
                               float top_p,
                               float repetition_penalty)
{
    if (!ctx || !ctx->pipeline) return -1;

    kaguya::SamplingParams sp;
    sp.temperature = temperature;
    sp.top_k = top_k;
    sp.top_p = top_p;
    sp.repetition_penalty = repetition_penalty;

    kaguya::Sampler sampler(sp);
    return ctx->pipeline->decode(sampler);
}

int32_t* kaguya_context_generate(kaguya_context* ctx,
                                  int64_t n_predict,
                                  float temperature,
                                  int top_k,
                                  float top_p,
                                  float repetition_penalty,
                                  int64_t* out_count)
{
    if (!ctx || !ctx->pipeline || !out_count) return nullptr;

    kaguya::SamplingParams sp;
    sp.temperature = temperature;
    sp.top_k = top_k;
    sp.top_p = top_p;
    sp.repetition_penalty = repetition_penalty;

    kaguya::Sampler sampler(sp);
    auto tokens = ctx->pipeline->generate({}, static_cast<int32_t>(n_predict), sampler);

    *out_count = static_cast<int64_t>(tokens.size());
    if (tokens.empty()) return nullptr;

    int32_t* result = static_cast<int32_t*>(std::malloc(tokens.size() * sizeof(int32_t)));
    if (!result) {
        *out_count = 0;
        return nullptr;
    }

    std::memcpy(result, tokens.data(), tokens.size() * sizeof(int32_t));
    return result;
}

const float* kaguya_context_logits(const kaguya_context* ctx,
                                    int64_t* out_count)
{
    if (!ctx || !ctx->pipeline || !out_count) return nullptr;

    const auto& logits = ctx->pipeline->logits();
    *out_count = static_cast<int64_t>(logits.size());
    return logits.data();
}

// ============================================================================
// Utility
// ============================================================================

void kaguya_init(void) {
    kaguya::CpuFeatureDetector::detect();
    kaguya::MemoryManager::init();
}

static std::string s_cpu_info_cache;

const char* kaguya_cpu_info(void) {
    s_cpu_info_cache = kaguya::CpuFeatureDetector::summary();
    return s_cpu_info_cache.c_str();
}

// ============================================================================
// Streaming generation
// ============================================================================

int32_t* kaguya_context_generate_streaming(kaguya_context* ctx,
                                            int64_t n_predict,
                                            float temperature,
                                            int top_k,
                                            float top_p,
                                            float repetition_penalty,
                                            kaguya_stream_callback callback,
                                            void* user_data,
                                            int64_t* out_count)
{
    if (!ctx || !ctx->pipeline || !out_count) return nullptr;

    kaguya::SamplingParams sp;
    sp.temperature = temperature;
    sp.top_k = top_k;
    sp.top_p = top_p;
    sp.repetition_penalty = repetition_penalty;
    kaguya::Sampler sampler(sp);

    std::vector<int32_t> tokens;
    tokens.reserve(static_cast<size_t>(n_predict));

    for (int64_t i = 0; i < n_predict; ++i) {
        if (ctx->pipeline->is_context_full()) break;

        int32_t token = ctx->pipeline->decode(sampler);
        tokens.push_back(token);

        // Check for EOS
        if (ctx->pipeline->eos_token_id() >= 0 && token == ctx->pipeline->eos_token_id()) {
            break;
        }

        // Call callback if provided
        if (callback) {
            // Decode token text
            std::string token_text;
            if (ctx->model && ctx->model->loader) {
                const auto& tok = ctx->model->loader->tokenizer();
                if (tok.is_valid()) {
                    token_text = tok.decode_token(token, true);
                } else {
                    if (token >= 0 && token < 256) {
                        token_text = std::string(1, static_cast<char>(token));
                    }
                }
            }

            int result = callback(token, token_text.c_str(), user_data);
            if (result != 0) break; // User requested stop
        }
    }

    *out_count = static_cast<int64_t>(tokens.size());
    if (tokens.empty()) return nullptr;

    int32_t* result = static_cast<int32_t*>(std::malloc(tokens.size() * sizeof(int32_t)));
    if (!result) {
        *out_count = 0;
        return nullptr;
    }

    std::memcpy(result, tokens.data(), tokens.size() * sizeof(int32_t));
    return result;
}

int32_t kaguya_model_eos_token_id(const kaguya_model* model) {
    if (!model || !model->loader) return -1;
    const auto& tok = model->loader->tokenizer();
    if (tok.is_valid()) {
        return tok.eos_token_id();
    }
    return -1;
}

void kaguya_context_set_eos_token(kaguya_context* ctx, int32_t eos_id) {
    if (ctx && ctx->pipeline) {
        ctx->pipeline->set_eos_token_id(eos_id);
    }
}
