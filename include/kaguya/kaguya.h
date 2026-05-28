#pragma once
/// @file kaguya.h
/// @brief Phase 5 C API for Kaguya CPU inference engine.
///
/// Simple C-compatible API for loading models, creating inference contexts,
/// generating tokens, and freeing resources. This API enables bindings to
/// other languages (Python, Rust, etc.) and integration into applications
/// that require a stable ABI.
///
/// Usage:
///   kaguya_model* model = kaguya_model_load("model.gguf");
///   kaguya_context* ctx = kaguya_context_create(model);
///   kaguya_context_prompt(ctx, "Hello", 0.8f, 40, 0.95f);
///   int32_t token = kaguya_context_decode(ctx);
///   kaguya_context_free(ctx);
///   kaguya_model_free(model);

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================================
// Opaque handles
// ============================================================================

/// Opaque model handle
typedef struct kaguya_model kaguya_model;

/// Opaque inference context handle
typedef struct kaguya_context kaguya_context;

// ============================================================================
// Version info
// ============================================================================

/// Get library version string (e.g., "0.2.0")
const char* kaguya_version(void);

// ============================================================================
// Model management
// ============================================================================

/// Load a model from a GGUF file
/// @param path Path to the .gguf model file
/// @return Model handle, or NULL on failure
kaguya_model* kaguya_model_load(const char* path);

/// Free a model handle
/// @param model Model handle (may be NULL)
void kaguya_model_free(kaguya_model* model);

/// Get model architecture name (e.g., "llama", "qwen2")
/// @param model Model handle
/// @return Architecture name string (valid until model is freed), or NULL
const char* kaguya_model_arch(const kaguya_model* model);

/// Get model vocabulary size
/// @param model Model handle
/// @return Vocabulary size, or 0 on error
int64_t kaguya_model_vocab_size(const kaguya_model* model);

/// Get model context length (maximum sequence length)
/// @param model Model handle
/// @return Context length, or 0 on error
int64_t kaguya_model_context_length(const kaguya_model* model);

/// Get model embedding dimension
/// @param model Model handle
/// @return Embedding dimension, or 0 on error
int64_t kaguya_model_emb_dim(const kaguya_model* model);

/// Get model number of layers
/// @param model Model handle
/// @return Number of layers, or 0 on error
int64_t kaguya_model_num_layers(const kaguya_model* model);

/// Get model number of attention heads
/// @param model Model handle
/// @return Number of heads, or 0 on error
int64_t kaguya_model_num_heads(const kaguya_model* model);

// ============================================================================
// Inference context
// ============================================================================

/// Create an inference context for a model
/// @param model Model handle
/// @return Context handle, or NULL on failure
kaguya_context* kaguya_context_create(const kaguya_model* model);

/// Free an inference context
/// @param ctx Context handle (may be NULL)
void kaguya_context_free(kaguya_context* ctx);

/// Reset the inference context (clear KV cache, start new sequence)
/// @param ctx Context handle
void kaguya_context_reset(kaguya_context* ctx);

/// Get current position in the context
/// @param ctx Context handle
/// @return Current position (0-based), or -1 on error
int64_t kaguya_context_position(const kaguya_context* ctx);

// ============================================================================
// Tokenization (simple byte-level)
// ============================================================================

/// Encode text to token IDs (byte-level, no BPE)
/// Caller must free the returned array with kaguya_tokens_free()
/// @param ctx Context handle
/// @param text Input text (UTF-8)
/// @param out_tokens Output: array of token IDs (caller-allocated)
/// @param out_count Output: number of tokens
/// @return 0 on success, -1 on error
int kaguya_tokenize(const kaguya_context* ctx,
                    const char* text,
                    int32_t** out_tokens,
                    int64_t* out_count);

/// Decode token IDs to text (byte-level)
/// Caller must free the returned string with kaguya_text_free()
/// @param ctx Context handle
/// @param tokens Array of token IDs
/// @param count Number of tokens
/// @return Decoded text string (caller must free), or NULL on error
char* kaguya_detokenize(const kaguya_context* ctx,
                        const int32_t* tokens,
                        int64_t count);

/// Free token array allocated by kaguya_tokenize
/// @param tokens Token array (may be NULL)
void kaguya_tokens_free(int32_t* tokens);

/// Free text string allocated by kaguya_detokenize
/// @param text Text string (may be NULL)
void kaguya_text_free(char* text);

// ============================================================================
// Inference
// ============================================================================

/// Feed prompt tokens to the context (prefill)
/// @param ctx Context handle
/// @param tokens Array of token IDs
/// @param count Number of tokens
/// @return 0 on success, -1 on error
int kaguya_context_prompt_tokens(kaguya_context* ctx,
                                  const int32_t* tokens,
                                  int64_t count);

/// Generate one token from the current context
/// @param ctx Context handle
/// @param temperature Sampling temperature
/// @param top_k Top-K sampling parameter
/// @param top_p Top-P (nucleus) sampling parameter
/// @param repetition_penalty Repetition penalty (1.0 = disabled)
/// @return Generated token ID, or -1 on error
int32_t kaguya_context_decode(kaguya_context* ctx,
                               float temperature,
                               int top_k,
                               float top_p,
                               float repetition_penalty);

/// Generate multiple tokens from the current context
/// Caller must free the returned array with kaguya_tokens_free()
/// @param ctx Context handle
/// @param n_predict Number of tokens to generate
/// @param temperature Sampling temperature
/// @param top_k Top-K sampling parameter
/// @param top_p Top-P sampling parameter
/// @param repetition_penalty Repetition penalty
/// @param out_count Output: actual number of tokens generated
/// @return Array of generated token IDs (caller must free), or NULL on error
int32_t* kaguya_context_generate(kaguya_context* ctx,
                                  int64_t n_predict,
                                  float temperature,
                                  int top_k,
                                  float top_p,
                                  float repetition_penalty,
                                  int64_t* out_count);

/// Get logits from the last forward pass
/// @param ctx Context handle
/// @param out_count Output: number of logits (vocab_size)
/// @return Pointer to logits array (valid until next decode/generate), or NULL
const float* kaguya_context_logits(const kaguya_context* ctx,
                                    int64_t* out_count);

// ============================================================================
// Utility
// ============================================================================

/// Initialize Kaguya library (detect CPU features, init memory manager)
/// Must be called before any other function. Can be called multiple times.
void kaguya_init(void);

/// Get CPU feature summary string
/// @return Summary string (static storage, do not free)
const char* kaguya_cpu_info(void);

#ifdef __cplusplus
}
#endif
