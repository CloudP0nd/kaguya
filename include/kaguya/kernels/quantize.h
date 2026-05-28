#pragma once
/// @file quantize.h
/// @brief Phase 3 quantization/dequantization kernels for Kaguya CPU inference engine.
///
/// Supports Q4_0, Q5_0, Q5_1, Q8_0 block quantization formats (ggml-compatible).
/// Provides dequantization, quantization (for testing), and fused dequantize+GEMM kernels.
///
/// Block layouts:
///   Q4_0: 18 bytes = {fp16 d, uint8_t qs[16]}          → 32 FP32 values
///   Q5_0: 22 bytes = {fp16 d, uint8_t qh[4], uint8_t qs[16]} → 32 FP32 values
///   Q5_1: 24 bytes = {fp16 d, fp16 m, uint8_t qh[4], uint8_t qs[16]} → 32 FP32 values
///   Q8_0: 34 bytes = {fp16 d, int8_t qs[32]}            → 32 FP32 values

#include <cstdint>
#include <cstddef>
#include "kaguya/tensor.h"

namespace kaguya::kernels {

// ---- Dequantization: quantized → FP32 ----

/// Dequantize a single block of Q4_0 to FP32
/// block: 18 bytes = {float16 d, uint8_t qs[16]} → 32 FP32 values
void dequantize_q4_0_block(const void* block, float* out);

/// Dequantize Q4_0 tensor data to FP32
/// n_blocks: number of Q4_0 blocks
void dequantize_q4_0(const void* data, float* out, int64_t n_blocks);

/// Dequantize Q8_0 tensor data to FP32
/// block: 34 bytes = {float16 d, int8_t qs[32]} → 32 FP32 values
void dequantize_q8_0(const void* data, float* out, int64_t n_blocks);

/// Dequantize Q5_0 tensor data to FP32
/// block: 22 bytes = {float16 d, uint8_t qh[4], uint8_t qs[16]} → 32 FP32 values
void dequantize_q5_0(const void* data, float* out, int64_t n_blocks);

/// Dequantize Q5_1 tensor data to FP32
/// block: 24 bytes = {float16 d, float16 m, uint8_t qh[4], uint8_t qs[16]} → 32 FP32 values
void dequantize_q5_1(const void* data, float* out, int64_t n_blocks);

/// Generic dequantize dispatch based on DataType
void dequantize_dispatch(DataType dtype, const void* data, float* out, int64_t n_elements);

// ---- Quantization: FP32 → quantized ----

/// Quantize FP32 to Q4_0 (for testing)
void quantize_q4_0(const float* in, void* out, int64_t n_blocks);

/// Quantize FP32 to Q8_0 (for testing)
void quantize_q8_0(const float* in, void* out, int64_t n_blocks);

// ---- Fused dequantize+GEMM ----

/// Fused Q4_0 dequantize + GEMM: C[M,N] = A_deq[M,K] * B[N,K] (B transposed)
/// A is Q4_0 data of total size n_a_blocks blocks
/// B is FP32 [N,K] row-major
/// This is the key optimization: avoid materializing the full FP32 matrix
void gemm_dequantize_q4_0_fused(
    const void* a_quantized, int64_t n_a_blocks,
    const float* B, int64_t N, int64_t K,
    float* C, int64_t ldc);

/// Fused Q8_0 dequantize + GEMM
void gemm_dequantize_q8_0_fused(
    const void* a_quantized, int64_t n_a_blocks,
    const float* B, int64_t N, int64_t K,
    float* C, int64_t ldc);

} // namespace kaguya::kernels
