#pragma once
/// @file special_ops.h
/// @brief Phase 3 special operation kernels: Softmax, RMSNorm, LayerNorm, RoPE, Activation.
///
/// Kernel hierarchy (fallback chain):
///   AVX-512 FP32 → Scalar
///
/// Primary target: AVX-512 BF16/VNNI platform (AVX-512F always available)

#include <cstdint>
#include <cstddef>

namespace kaguya::kernels {

// ---- Softmax ----
/// In-place softmax over x[0..n), with optional temperature
void softmax_scalar(float* x, int64_t n, float temperature = 1.0f);
void softmax_avx512(float* x, int64_t n, float temperature = 1.0f);
void softmax_dispatch(float* x, int64_t n, float temperature = 1.0f);

// ---- RMSNorm ----
/// RMSNorm: out[i] = (x[i] / sqrt(mean(x^2) + eps)) * weight[i]
/// weight may be nullptr (then just normalize)
void rmsnorm_scalar(float* out, const float* x, const float* weight, int64_t n, float eps = 1e-5f);
void rmsnorm_avx512(float* out, const float* x, const float* weight, int64_t n, float eps = 1e-5f);
void rmsnorm_dispatch(float* out, const float* x, const float* weight, int64_t n, float eps = 1e-5f);

// ---- LayerNorm ----
void layernorm_scalar(float* out, const float* x, const float* weight, const float* bias, int64_t n, float eps = 1e-5f);
void layernorm_avx512(float* out, const float* x, const float* weight, const float* bias, int64_t n, float eps = 1e-5f);
void layernorm_dispatch(float* out, const float* x, const float* weight, const float* bias, int64_t n, float eps = 1e-5f);

// ---- RoPE (Rotary Position Embedding) ----
/// Apply RoPE to query/key vectors in-place
/// x: [n_heads, head_dim] — interleaved pairs (x0,x1,x2,x3...) where (x0,x1) is a pair
/// pos: current position
/// freq_base, freq_scale: rope parameters
void rope_scalar(float* x, int64_t n_heads, int64_t head_dim, int64_t pos, float freq_base = 10000.0f, float freq_scale = 1.0f);
void rope_avx512(float* x, int64_t n_heads, int64_t head_dim, int64_t pos, float freq_base = 10000.0f, float freq_scale = 1.0f);
void rope_dispatch(float* x, int64_t n_heads, int64_t head_dim, int64_t pos, float freq_base = 10000.0f, float freq_scale = 1.0f);

// ---- Activation functions ----
/// SiLU: x * sigmoid(x) = x / (1 + exp(-x))
void silu_scalar(float* x, int64_t n);
void silu_avx512(float* x, int64_t n);
void silu_dispatch(float* x, int64_t n);

/// GELU (approximate): 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
void gelu_scalar(float* x, int64_t n);
void gelu_avx512(float* x, int64_t n);
void gelu_dispatch(float* x, int64_t n);

} // namespace kaguya::kernels
