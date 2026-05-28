/// @file bench_quantize.cpp
/// @brief Phase 5 Quantization micro-benchmarks.
///
/// Measures throughput of dequantization and fused dequantize+GEMM kernels
/// for Q4_0, Q5_0, Q5_1, Q8_0 block formats.

#include "kaguya/kernels/quantize.h"
#include "kaguya/kernels/dispatcher.h"
#include "kaguya/cpu_features.h"
#include "kaguya/memory_manager.h"
#include "kaguya/tensor.h"

#include "benchmark/benchmark.h"

#include <vector>
#include <random>
#include <cstring>

// ============================================================================
// Helper data generation
// ============================================================================

static constexpr int64_t BLOCK_SIZE = 32; // ggml block size

static std::vector<float> random_fp32(int64_t n) {
    std::vector<float> data(static_cast<size_t>(n));
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (auto& v : data) v = dist(rng);
    return data;
}

// ============================================================================
// Dequantization benchmarks
// ============================================================================

// --- Q4_0 ---

static void bench_dequantize_q4_0(benchmark::State& state) {
    const int64_t n_blocks = state.range(0);
    const int64_t n_elements = n_blocks * BLOCK_SIZE;

    auto fp32_data = random_fp32(n_elements);
    std::vector<uint8_t> q4_data(static_cast<size_t>(n_blocks * 18)); // 18 bytes/block
    kaguya::kernels::quantize_q4_0(fp32_data.data(), q4_data.data(), n_blocks);

    std::vector<float> output(static_cast<size_t>(n_elements));

    for (auto _ : state) {
        kaguya::kernels::dequantize_q4_0(q4_data.data(), output.data(), n_blocks);
        benchmark::DoNotOptimize(output.data());
    }

    state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) * n_elements);
    state.SetBytesProcessed(static_cast<int64_t>(state.iterations()) * n_blocks * 18);
}

BENCHMARK(bench_dequantize_q4_0)->Arg(1024)->Unit(benchmark::kMicrosecond);     // 32K elements
BENCHMARK(bench_dequantize_q4_0)->Arg(4096)->Unit(benchmark::kMicrosecond);     // 128K elements
BENCHMARK(bench_dequantize_q4_0)->Arg(16384)->Unit(benchmark::kMicrosecond);    // 512K elements

// --- Q8_0 ---

static void bench_dequantize_q8_0(benchmark::State& state) {
    const int64_t n_blocks = state.range(0);
    const int64_t n_elements = n_blocks * BLOCK_SIZE;

    auto fp32_data = random_fp32(n_elements);
    std::vector<uint8_t> q8_data(static_cast<size_t>(n_blocks * 34)); // 34 bytes/block
    kaguya::kernels::quantize_q8_0(fp32_data.data(), q8_data.data(), n_blocks);

    std::vector<float> output(static_cast<size_t>(n_elements));

    for (auto _ : state) {
        kaguya::kernels::dequantize_q8_0(q8_data.data(), output.data(), n_blocks);
        benchmark::DoNotOptimize(output.data());
    }

    state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) * n_elements);
    state.SetBytesProcessed(static_cast<int64_t>(state.iterations()) * n_blocks * 34);
}

BENCHMARK(bench_dequantize_q8_0)->Arg(1024)->Unit(benchmark::kMicrosecond);
BENCHMARK(bench_dequantize_q8_0)->Arg(4096)->Unit(benchmark::kMicrosecond);
BENCHMARK(bench_dequantize_q8_0)->Arg(16384)->Unit(benchmark::kMicrosecond);

// --- Q5_0 ---

static void bench_dequantize_q5_0(benchmark::State& state) {
    const int64_t n_blocks = state.range(0);
    const int64_t n_elements = n_blocks * BLOCK_SIZE;

    // Create Q5_0 data by first making FP32 data, then constructing blocks manually
    // For benchmarking, we just need valid-sized data — the output won't be meaningful
    std::vector<uint8_t> q5_data(static_cast<size_t>(n_blocks * 22), 0); // 22 bytes/block
    // Set some non-zero scale values to avoid all-zero
    for (int64_t i = 0; i < n_blocks; ++i) {
        uint16_t scale = 0x3C00; // 1.0 in FP16
        std::memcpy(q5_data.data() + i * 22, &scale, 2);
    }

    std::vector<float> output(static_cast<size_t>(n_elements));

    for (auto _ : state) {
        kaguya::kernels::dequantize_q5_0(q5_data.data(), output.data(), n_blocks);
        benchmark::DoNotOptimize(output.data());
    }

    state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) * n_elements);
    state.SetBytesProcessed(static_cast<int64_t>(state.iterations()) * n_blocks * 22);
}

BENCHMARK(bench_dequantize_q5_0)->Arg(1024)->Unit(benchmark::kMicrosecond);
BENCHMARK(bench_dequantize_q5_0)->Arg(4096)->Unit(benchmark::kMicrosecond);

// --- Q5_1 ---

static void bench_dequantize_q5_1(benchmark::State& state) {
    const int64_t n_blocks = state.range(0);
    const int64_t n_elements = n_blocks * BLOCK_SIZE;

    std::vector<uint8_t> q5_data(static_cast<size_t>(n_blocks * 24), 0); // 24 bytes/block
    for (int64_t i = 0; i < n_blocks; ++i) {
        uint16_t scale = 0x3C00;
        std::memcpy(q5_data.data() + i * 24, &scale, 2);
        uint16_t min_val = 0x3400; // small offset
        std::memcpy(q5_data.data() + i * 24 + 2, &min_val, 2);
    }

    std::vector<float> output(static_cast<size_t>(n_elements));

    for (auto _ : state) {
        kaguya::kernels::dequantize_q5_1(q5_data.data(), output.data(), n_blocks);
        benchmark::DoNotOptimize(output.data());
    }

    state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) * n_elements);
    state.SetBytesProcessed(static_cast<int64_t>(state.iterations()) * n_blocks * 24);
}

BENCHMARK(bench_dequantize_q5_1)->Arg(1024)->Unit(benchmark::kMicrosecond);
BENCHMARK(bench_dequantize_q5_1)->Arg(4096)->Unit(benchmark::kMicrosecond);

// ============================================================================
// BF16 / F16 dequantization
// ============================================================================

static void bench_dequantize_bf16(benchmark::State& state) {
    const int64_t n = state.range(0);
    std::vector<uint16_t> bf16_data(static_cast<size_t>(n), 0x3F80); // 1.0 in BF16
    std::vector<float> output(static_cast<size_t>(n));

    for (auto _ : state) {
        kaguya::kernels::dequantize_bf16(bf16_data.data(), output.data(), n);
        benchmark::DoNotOptimize(output.data());
    }

    state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) * n);
    state.SetBytesProcessed(static_cast<int64_t>(state.iterations()) * n * 2);  // 2 bytes/element in
}

BENCHMARK(bench_dequantize_bf16)->Arg(4096)->Unit(benchmark::kMicrosecond);
BENCHMARK(bench_dequantize_bf16)->Arg(16384)->Unit(benchmark::kMicrosecond);
BENCHMARK(bench_dequantize_bf16)->Arg(65536)->Unit(benchmark::kMicrosecond);

static void bench_dequantize_f16(benchmark::State& state) {
    const int64_t n = state.range(0);
    std::vector<uint16_t> f16_data(static_cast<size_t>(n), 0x3C00); // 1.0 in FP16
    std::vector<float> output(static_cast<size_t>(n));

    for (auto _ : state) {
        kaguya::kernels::dequantize_f16(f16_data.data(), output.data(), n);
        benchmark::DoNotOptimize(output.data());
    }

    state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) * n);
    state.SetBytesProcessed(static_cast<int64_t>(state.iterations()) * n * 2);
}

BENCHMARK(bench_dequantize_f16)->Arg(4096)->Unit(benchmark::kMicrosecond);
BENCHMARK(bench_dequantize_f16)->Arg(16384)->Unit(benchmark::kMicrosecond);
BENCHMARK(bench_dequantize_f16)->Arg(65536)->Unit(benchmark::kMicrosecond);

// ============================================================================
// Quantization benchmarks (FP32 → quantized)
// ============================================================================

static void bench_quantize_q4_0(benchmark::State& state) {
    const int64_t n_blocks = state.range(0);
    const int64_t n_elements = n_blocks * BLOCK_SIZE;

    auto fp32_data = random_fp32(n_elements);
    std::vector<uint8_t> q4_data(static_cast<size_t>(n_blocks * 18));

    for (auto _ : state) {
        kaguya::kernels::quantize_q4_0(fp32_data.data(), q4_data.data(), n_blocks);
        benchmark::DoNotOptimize(q4_data.data());
    }

    state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) * n_elements);
}

BENCHMARK(bench_quantize_q4_0)->Arg(1024)->Unit(benchmark::kMicrosecond);
BENCHMARK(bench_quantize_q4_0)->Arg(4096)->Unit(benchmark::kMicrosecond);

static void bench_quantize_q8_0(benchmark::State& state) {
    const int64_t n_blocks = state.range(0);
    const int64_t n_elements = n_blocks * BLOCK_SIZE;

    auto fp32_data = random_fp32(n_elements);
    std::vector<uint8_t> q8_data(static_cast<size_t>(n_blocks * 34));

    for (auto _ : state) {
        kaguya::kernels::quantize_q8_0(fp32_data.data(), q8_data.data(), n_blocks);
        benchmark::DoNotOptimize(q8_data.data());
    }

    state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) * n_elements);
}

BENCHMARK(bench_quantize_q8_0)->Arg(1024)->Unit(benchmark::kMicrosecond);
BENCHMARK(bench_quantize_q8_0)->Arg(4096)->Unit(benchmark::kMicrosecond);

// ============================================================================
// Fused dequantize + GEMM
// ============================================================================

static void bench_fused_q4_0_gemm(benchmark::State& state) {
    const int64_t M = state.range(0);
    const int64_t K = state.range(1);
    const int64_t N = state.range(2);

    const int64_t n_a_blocks = (M * K) / BLOCK_SIZE;

    // Create Q4_0 data for A matrix
    auto fp32_a = random_fp32(M * K);
    std::vector<uint8_t> q4_a(static_cast<size_t>(n_a_blocks * 18));
    kaguya::kernels::quantize_q4_0(fp32_a.data(), q4_a.data(), n_a_blocks);

    auto B_data = random_fp32(N * K);
    auto C = random_fp32(M * N);

    for (auto _ : state) {
        kaguya::kernels::gemm_dequantize_q4_0_fused(
            q4_a.data(), n_a_blocks,
            B_data.data(), N, K,
            C.data(), N);
        benchmark::DoNotOptimize(C.data());
    }

    const double flops = 2.0 * M * N * K;
    state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) * static_cast<int64_t>(flops));
}

BENCHMARK(bench_fused_q4_0_gemm)->Args({1, 4096, 4096})->Unit(benchmark::kMicrosecond);
BENCHMARK(bench_fused_q4_0_gemm)->Args({32, 4096, 4096})->Unit(benchmark::kMicrosecond);

static void bench_fused_q8_0_gemm(benchmark::State& state) {
    const int64_t M = state.range(0);
    const int64_t K = state.range(1);
    const int64_t N = state.range(2);

    const int64_t n_a_blocks = (M * K) / BLOCK_SIZE;

    auto fp32_a = random_fp32(M * K);
    std::vector<uint8_t> q8_a(static_cast<size_t>(n_a_blocks * 34));
    kaguya::kernels::quantize_q8_0(fp32_a.data(), q8_a.data(), n_a_blocks);

    auto B_data = random_fp32(N * K);
    auto C = random_fp32(M * N);

    for (auto _ : state) {
        kaguya::kernels::gemm_dequantize_q8_0_fused(
            q8_a.data(), n_a_blocks,
            B_data.data(), N, K,
            C.data(), N);
        benchmark::DoNotOptimize(C.data());
    }

    const double flops = 2.0 * M * N * K;
    state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) * static_cast<int64_t>(flops));
}

BENCHMARK(bench_fused_q8_0_gemm)->Args({1, 4096, 4096})->Unit(benchmark::kMicrosecond);
BENCHMARK(bench_fused_q8_0_gemm)->Args({32, 4096, 4096})->Unit(benchmark::kMicrosecond);
