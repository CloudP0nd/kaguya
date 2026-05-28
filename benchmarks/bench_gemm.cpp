/// @file bench_gemm.cpp
/// @brief Phase 5 GEMM micro-benchmarks.
///
/// Measures throughput of FP32/FP16/BF16/INT8 GEMM kernels across
/// the fallback chain (AVX-512 → AVX2 → Scalar).

#include "kaguya/kernels/gemm.h"
#include "kaguya/kernels/dispatcher.h"
#include "kaguya/cpu_features.h"
#include "kaguya/memory_manager.h"

#include "benchmark/benchmark.h"

#include <vector>
#include <random>
#include <cmath>

// ============================================================================
// Helper: aligned buffer with random data
// ============================================================================

static std::vector<float> random_matrix(int64_t rows, int64_t cols) {
    std::vector<float> mat(static_cast<size_t>(rows * cols));
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (auto& v : mat) v = dist(rng);
    return mat;
}

// ============================================================================
// FP32 GEMM benchmarks — dispatch (auto-selects best kernel)
// ============================================================================

static void bench_gemm_dispatch(benchmark::State& state) {
    const int64_t M = state.range(0);
    const int64_t K = state.range(1);
    const int64_t N = state.range(2);

    auto A = random_matrix(M, K);
    auto B = random_matrix(K, N);
    auto C = random_matrix(M, N);

    for (auto _ : state) {
        kaguya::kernels::GemmParams params;
        params.M = M; params.N = N; params.K = K;
        params.A = A.data(); params.lda = K;
        params.B = B.data(); params.ldb = N;
        params.C = C.data(); params.ldc = N;
        params.alpha = 1.0f; params.beta = 0.0f;
        kaguya::kernels::gemm_dispatch(params);
        benchmark::DoNotOptimize(C.data());
    }

    // Report GFLOPS
    const double flops = 2.0 * M * N * K;
    state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) * static_cast<int64_t>(flops));
    state.SetLabel("GFLOPS");
}

// Common GEMM sizes for transformer inference
BENCHMARK(bench_gemm_dispatch)->Args({1, 4096, 4096})->Unit(benchmark::kMicrosecond);     // GEMV decode
BENCHMARK(bench_gemm_dispatch)->Args({32, 4096, 4096})->Unit(benchmark::kMicrosecond);    // Small batch
BENCHMARK(bench_gemm_dispatch)->Args({128, 4096, 4096})->Unit(benchmark::kMicrosecond);   // Prefill chunk
BENCHMARK(bench_gemm_dispatch)->Args({4096, 4096, 4096})->Unit(benchmark::kMillisecond);  // Full GEMM

// FFN dimension sizes
BENCHMARK(bench_gemm_dispatch)->Args({1, 4096, 11008})->Unit(benchmark::kMicrosecond);    // GEMV FFN gate
BENCHMARK(bench_gemm_dispatch)->Args({1, 11008, 4096})->Unit(benchmark::kMicrosecond);    // GEMV FFN down

// ============================================================================
// FP32 Scalar GEMM benchmark (baseline)
// ============================================================================

static void bench_gemm_scalar(benchmark::State& state) {
    const int64_t M = state.range(0);
    const int64_t K = state.range(1);
    const int64_t N = state.range(2);

    auto A = random_matrix(M, K);
    auto B = random_matrix(K, N);
    auto C = random_matrix(M, N);

    for (auto _ : state) {
        kaguya::kernels::GemmParams params;
        params.M = M; params.N = N; params.K = K;
        params.A = A.data(); params.lda = K;
        params.B = B.data(); params.ldb = N;
        params.C = C.data(); params.ldc = N;
        params.alpha = 1.0f; params.beta = 0.0f;
        kaguya::kernels::gemm_scalar(params);
        benchmark::DoNotOptimize(C.data());
    }

    const double flops = 2.0 * M * N * K;
    state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) * static_cast<int64_t>(flops));
}

BENCHMARK(bench_gemm_scalar)->Args({1, 4096, 4096})->Unit(benchmark::kMicrosecond);
BENCHMARK(bench_gemm_scalar)->Args({4096, 4096, 4096})->Unit(benchmark::kMillisecond);

// ============================================================================
// FP32 AVX-512 GEMM benchmark
// ============================================================================

static void bench_gemm_avx512(benchmark::State& state) {
    const int64_t M = state.range(0);
    const int64_t K = state.range(1);
    const int64_t N = state.range(2);

    auto A = random_matrix(M, K);
    auto B = random_matrix(K, N);
    auto C = random_matrix(M, N);

    for (auto _ : state) {
        kaguya::kernels::GemmParams params;
        params.M = M; params.N = N; params.K = K;
        params.A = A.data(); params.lda = K;
        params.B = B.data(); params.ldb = N;
        params.C = C.data(); params.ldc = N;
        params.alpha = 1.0f; params.beta = 0.0f;
        kaguya::kernels::gemm_avx512(params);
        benchmark::DoNotOptimize(C.data());
    }

    const double flops = 2.0 * M * N * K;
    state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) * static_cast<int64_t>(flops));
}

BENCHMARK(bench_gemm_avx512)->Args({1, 4096, 4096})->Unit(benchmark::kMicrosecond);
BENCHMARK(bench_gemm_avx512)->Args({4096, 4096, 4096})->Unit(benchmark::kMillisecond);

// ============================================================================
// FP32 AVX2 GEMM benchmark
// ============================================================================

static void bench_gemm_avx2(benchmark::State& state) {
    const int64_t M = state.range(0);
    const int64_t K = state.range(1);
    const int64_t N = state.range(2);

    auto A = random_matrix(M, K);
    auto B = random_matrix(K, N);
    auto C = random_matrix(M, N);

    for (auto _ : state) {
        kaguya::kernels::GemmParams params;
        params.M = M; params.N = N; params.K = K;
        params.A = A.data(); params.lda = K;
        params.B = B.data(); params.ldb = N;
        params.C = C.data(); params.ldc = N;
        params.alpha = 1.0f; params.beta = 0.0f;
        kaguya::kernels::gemm_avx2(params);
        benchmark::DoNotOptimize(C.data());
    }

    const double flops = 2.0 * M * N * K;
    state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) * static_cast<int64_t>(flops));
}

BENCHMARK(bench_gemm_avx2)->Args({1, 4096, 4096})->Unit(benchmark::kMicrosecond);
BENCHMARK(bench_gemm_avx2)->Args({4096, 4096, 4096})->Unit(benchmark::kMillisecond);

// ============================================================================
// GEMV-specific benchmarks (M=1, the decode case)
// ============================================================================

static void bench_gemv_dispatch(benchmark::State& state) {
    const int64_t K = state.range(0);
    const int64_t N_out = state.range(1);  // output dim

    auto W = random_matrix(N_out, K);
    auto x = random_matrix(1, K);
    auto y = random_matrix(1, N_out);

    for (auto _ : state) {
        kaguya::kernels::GemmParams params;
        params.M = N_out; params.N = 1; params.K = K;
        params.A = W.data(); params.lda = K;
        params.B = x.data(); params.ldb = 1;
        params.C = y.data(); params.ldc = 1;
        params.alpha = 1.0f; params.beta = 0.0f;
        kaguya::kernels::gemm_dispatch(params);
        benchmark::DoNotOptimize(y.data());
    }

    const double flops = 2.0 * N_out * K;
    state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) * static_cast<int64_t>(flops));
}

// Typical transformer decode sizes
BENCHMARK(bench_gemv_dispatch)->Args({4096, 4096})->Unit(benchmark::kMicrosecond);       // QKV
BENCHMARK(bench_gemv_dispatch)->Args({4096, 11008})->Unit(benchmark::kMicrosecond);      // FFN gate
BENCHMARK(bench_gemv_dispatch)->Args({11008, 4096})->Unit(benchmark::kMicrosecond);      // FFN down
BENCHMARK(bench_gemv_dispatch)->Args({4096, 32000})->Unit(benchmark::kMicrosecond);      // Output proj

BENCHMARK_MAIN();
