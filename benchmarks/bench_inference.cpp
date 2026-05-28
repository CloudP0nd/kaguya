/// @file bench_inference.cpp
/// @brief Phase 5 End-to-end inference benchmarks.
///
/// Measures end-to-end token generation throughput (tokens/s)
/// using the Pipeline with synthetic models of various sizes.
/// Also benchmarks individual special operations (Softmax, RMSNorm, RoPE, etc.)

#include "kaguya/kernels/special_ops.h"
#include "kaguya/kernels/gemm.h"
#include "kaguya/kernels/dispatcher.h"
#include "kaguya/cpu_features.h"
#include "kaguya/memory_manager.h"
#include "kaguya/model.h"
#include "kaguya/pipeline.h"
#include "kaguya/sampling.h"
#include "kaguya/kv_cache.h"

#include "benchmark/benchmark.h"

#include <vector>
#include <random>
#include <cmath>

// ============================================================================
// Helper: random data
// ============================================================================

static std::vector<float> random_data(int64_t n) {
    std::vector<float> data(static_cast<size_t>(n));
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (auto& v : data) v = dist(rng);
    return data;
}

// ============================================================================
// Special operations benchmarks
// ============================================================================

// --- Softmax ---

static void bench_softmax(benchmark::State& state) {
    const int64_t n = state.range(0);
    auto data = random_data(n);

    for (auto _ : state) {
        kaguya::kernels::softmax_dispatch(data.data(), n);
        benchmark::DoNotOptimize(data.data());
    }
    state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) * n);
}

BENCHMARK(bench_softmax)->Arg(64)->Unit(benchmark::kNanosecond);
BENCHMARK(bench_softmax)->Arg(256)->Unit(benchmark::kNanosecond);
BENCHMARK(bench_softmax)->Arg(1024)->Unit(benchmark::kNanosecond);
BENCHMARK(bench_softmax)->Arg(4096)->Unit(benchmark::kMicrosecond);

// --- RMSNorm ---

static void bench_rmsnorm(benchmark::State& state) {
    const int64_t n = state.range(0);
    auto x = random_data(n);
    auto weight = random_data(n);
    auto out = random_data(n);

    for (auto _ : state) {
        kaguya::kernels::rmsnorm_dispatch(out.data(), x.data(), weight.data(), n);
        benchmark::DoNotOptimize(out.data());
    }
    state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) * n);
}

BENCHMARK(bench_rmsnorm)->Arg(512)->Unit(benchmark::kNanosecond);
BENCHMARK(bench_rmsnorm)->Arg(2048)->Unit(benchmark::kNanosecond);
BENCHMARK(bench_rmsnorm)->Arg(4096)->Unit(benchmark::kNanosecond);

// --- LayerNorm ---

static void bench_layernorm(benchmark::State& state) {
    const int64_t n = state.range(0);
    auto x = random_data(n);
    auto weight = random_data(n);
    auto bias = random_data(n);
    auto out = random_data(n);

    for (auto _ : state) {
        kaguya::kernels::layernorm_dispatch(out.data(), x.data(), weight.data(), bias.data(), n);
        benchmark::DoNotOptimize(out.data());
    }
    state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) * n);
}

BENCHMARK(bench_layernorm)->Arg(512)->Unit(benchmark::kNanosecond);
BENCHMARK(bench_layernorm)->Arg(2048)->Unit(benchmark::kNanosecond);
BENCHMARK(bench_layernorm)->Arg(4096)->Unit(benchmark::kNanosecond);

// --- RoPE ---

static void bench_rope(benchmark::State& state) {
    const int64_t n_heads = state.range(0);
    const int64_t head_dim = state.range(1);
    const int64_t total = n_heads * head_dim;
    auto x = random_data(total);

    int64_t pos = 0;
    for (auto _ : state) {
        kaguya::kernels::rope_dispatch(x.data(), n_heads, head_dim, pos, 10000.0f, 1.0f);
        benchmark::DoNotOptimize(x.data());
        ++pos;
    }
    state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) * total);
}

BENCHMARK(bench_rope)->Args({32, 128})->Unit(benchmark::kNanosecond);    // LLaMA-7B style
BENCHMARK(bench_rope)->Args({32, 64})->Unit(benchmark::kNanosecond);     // Smaller model
BENCHMARK(bench_rope)->Args({8, 64})->Unit(benchmark::kNanosecond);      // Tiny model

// --- SiLU ---

static void bench_silu(benchmark::State& state) {
    const int64_t n = state.range(0);
    auto data = random_data(n);

    for (auto _ : state) {
        kaguya::kernels::silu_dispatch(data.data(), n);
        benchmark::DoNotOptimize(data.data());
    }
    state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) * n);
}

BENCHMARK(bench_silu)->Arg(4096)->Unit(benchmark::kNanosecond);
BENCHMARK(bench_silu)->Arg(11008)->Unit(benchmark::kNanosecond);   // FFN dim
BENCHMARK(bench_silu)->Arg(32768)->Unit(benchmark::kNanosecond);

// --- GELU ---

static void bench_gelu(benchmark::State& state) {
    const int64_t n = state.range(0);
    auto data = random_data(n);

    for (auto _ : state) {
        kaguya::kernels::gelu_dispatch(data.data(), n);
        benchmark::DoNotOptimize(data.data());
    }
    state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) * n);
}

BENCHMARK(bench_gelu)->Arg(4096)->Unit(benchmark::kNanosecond);
BENCHMARK(bench_gelu)->Arg(11008)->Unit(benchmark::kNanosecond);

// ============================================================================
// End-to-end pipeline benchmarks (synthetic model)
// ============================================================================

namespace {

/// Create a synthetic model for benchmarking
kaguya::Model create_bench_model(int64_t emb_dim, int64_t num_layers,
                                  int64_t num_heads, int64_t ffn_dim,
                                  int64_t vocab_size = 32000) {
    kaguya::Model model;
    kaguya::ModelWeights weights;
    kaguya::HyperParams hp;
    hp.vocab_size = vocab_size;
    hp.context_length = 512;
    hp.emb_dim = emb_dim;
    hp.num_layers = num_layers;
    hp.num_heads = num_heads;
    hp.num_kv_heads = num_heads;
    hp.head_dim = emb_dim / num_heads;
    hp.ffn_dim = ffn_dim;
    hp.rope_freq_base = 10000.0f;
    hp.rope_freq_scale = 1.0f;
    hp.norm_eps = 1e-5f;
    hp.n_rot = hp.head_dim;
    hp.n_embd_head_k = hp.head_dim;
    hp.n_embd_head_v = hp.head_dim;
    hp.use_gqa = false;
    hp.n_rep = 1;

    weights.hparams = hp;
    weights.arch = kaguya::ModelArch::LLAMA;

    // Allocate static weight data
    static std::vector<float> tok_emb_data;
    tok_emb_data.assign(static_cast<size_t>(hp.vocab_size * hp.emb_dim), 0.01f);
    weights.tok_emb = tok_emb_data.data();
    weights.tok_emb_bytes = tok_emb_data.size() * sizeof(float);
    weights.tok_emb_ne0 = hp.emb_dim;
    weights.tok_emb_ne1 = hp.vocab_size;
    weights.tok_emb_dtype = kaguya::DataType::F32;

    static std::vector<float> output_norm_data;
    output_norm_data.assign(static_cast<size_t>(hp.emb_dim), 1.0f);
    weights.output_norm = output_norm_data.data();
    weights.output_norm_bytes = output_norm_data.size() * sizeof(float);
    weights.output_norm_dtype = kaguya::DataType::F32;

    static std::vector<float> output_proj_data;
    output_proj_data.assign(static_cast<size_t>(hp.vocab_size * hp.emb_dim), 0.01f);
    weights.output_proj = output_proj_data.data();
    weights.output_proj_bytes = output_proj_data.size() * sizeof(float);
    weights.output_proj_ne0 = hp.emb_dim;
    weights.output_proj_ne1 = hp.vocab_size;
    weights.output_proj_dtype = kaguya::DataType::F32;

    weights.layers.resize(static_cast<size_t>(hp.num_layers));
    static std::vector<std::vector<float>> layer_weights_storage;
    layer_weights_storage.clear();

    for (int64_t l = 0; l < hp.num_layers; ++l) {
        auto& lw = weights.layers[static_cast<size_t>(l)];

        auto alloc = [&](int64_t n, float val) -> const float* {
            layer_weights_storage.emplace_back(static_cast<size_t>(n), val);
            return layer_weights_storage.back().data();
        };

        lw.attn_norm = alloc(hp.emb_dim, 1.0f);
        lw.attn_norm_bytes = hp.emb_dim * sizeof(float);
        lw.attn_norm_dtype = kaguya::DataType::F32;

        lw.ffn_norm = alloc(hp.emb_dim, 1.0f);
        lw.ffn_norm_bytes = hp.emb_dim * sizeof(float);
        lw.ffn_norm_dtype = kaguya::DataType::F32;

        int64_t kv_dim = hp.num_kv_heads * hp.head_dim;

        lw.wq = alloc(hp.num_heads * hp.head_dim * hp.emb_dim, 0.01f);
        lw.wq_bytes = hp.num_heads * hp.head_dim * hp.emb_dim * sizeof(float);
        lw.wq_ne0 = hp.emb_dim; lw.wq_ne1 = hp.num_heads * hp.head_dim;
        lw.wq_dtype = kaguya::DataType::F32;

        lw.wk = alloc(kv_dim * hp.emb_dim, 0.01f);
        lw.wk_bytes = kv_dim * hp.emb_dim * sizeof(float);
        lw.wk_ne0 = hp.emb_dim; lw.wk_ne1 = kv_dim;
        lw.wk_dtype = kaguya::DataType::F32;

        lw.wv = alloc(kv_dim * hp.emb_dim, 0.01f);
        lw.wv_bytes = kv_dim * hp.emb_dim * sizeof(float);
        lw.wv_ne0 = hp.emb_dim; lw.wv_ne1 = kv_dim;
        lw.wv_dtype = kaguya::DataType::F32;

        lw.wo = alloc(hp.emb_dim * hp.num_heads * hp.head_dim, 0.01f);
        lw.wo_bytes = hp.emb_dim * hp.num_heads * hp.head_dim * sizeof(float);
        lw.wo_ne0 = hp.num_heads * hp.head_dim; lw.wo_ne1 = hp.emb_dim;
        lw.wo_dtype = kaguya::DataType::F32;

        lw.w_gate = alloc(hp.ffn_dim * hp.emb_dim, 0.01f);
        lw.w_gate_bytes = hp.ffn_dim * hp.emb_dim * sizeof(float);
        lw.w_gate_ne0 = hp.emb_dim; lw.w_gate_ne1 = hp.ffn_dim;
        lw.w_gate_dtype = kaguya::DataType::F32;

        lw.w_up = alloc(hp.ffn_dim * hp.emb_dim, 0.01f);
        lw.w_up_bytes = hp.ffn_dim * hp.emb_dim * sizeof(float);
        lw.w_up_ne0 = hp.emb_dim; lw.w_up_ne1 = hp.ffn_dim;
        lw.w_up_dtype = kaguya::DataType::F32;

        lw.w_down = alloc(hp.emb_dim * hp.ffn_dim, 0.01f);
        lw.w_down_bytes = hp.emb_dim * hp.ffn_dim * sizeof(float);
        lw.w_down_ne0 = hp.ffn_dim; lw.w_down_ne1 = hp.emb_dim;
        lw.w_down_dtype = kaguya::DataType::F32;
    }

    model.set_weights(std::move(weights));
    return model;
}

} // anonymous namespace

// ============================================================================
// Full pipeline decode benchmark
// ============================================================================

static void bench_pipeline_decode(benchmark::State& state) {
    const int64_t emb_dim = state.range(0);
    const int64_t num_layers = state.range(1);
    const int64_t num_heads = state.range(2);
    const int64_t ffn_dim = state.range(3);

    auto model = create_bench_model(emb_dim, num_layers, num_heads, ffn_dim);
    kaguya::Pipeline pipeline(model);

    kaguya::SamplingParams sp;
    sp.temperature = 1.0f;
    sp.top_k = 0;
    sp.top_p = 1.0f;
    sp.seed = 42;
    kaguya::Sampler sampler(sp);

    // Prefill with a single token
    pipeline.prefill({0});

    int64_t total_tokens = 0;

    for (auto _ : state) {
        int32_t tok = pipeline.decode(sampler);
        benchmark::DoNotOptimize(tok);
        ++total_tokens;
    }

    state.SetItemsProcessed(total_tokens);
}

// Tiny model (fast iteration)
BENCHMARK(bench_pipeline_decode)
    ->Args({64, 2, 2, 128})
    ->Unit(benchmark::kMicrosecond)
    ->Iterations(10);

// Small model
BENCHMARK(bench_pipeline_decode)
    ->Args({256, 4, 4, 512})
    ->Unit(benchmark::kMicrosecond)
    ->Iterations(5);

// Medium model (representative of 7B-class at small scale)
BENCHMARK(bench_pipeline_decode)
    ->Args({512, 4, 8, 1024})
    ->Unit(benchmark::kMillisecond)
    ->Iterations(3);

// ============================================================================
// Prefill throughput benchmark
// ============================================================================

static void bench_pipeline_prefill(benchmark::State& state) {
    const int64_t emb_dim = state.range(0);
    const int64_t num_layers = state.range(1);
    const int64_t num_heads = state.range(2);
    const int64_t ffn_dim = state.range(3);
    const int64_t prompt_len = state.range(4);

    auto model = create_bench_model(emb_dim, num_layers, num_heads, ffn_dim);

    kaguya::SamplingParams sp;
    sp.temperature = 1.0f;
    sp.top_k = 0;
    sp.top_p = 1.0f;
    sp.seed = 42;

    std::vector<int32_t> prompt(static_cast<size_t>(prompt_len), 0);

    int64_t total_tokens = 0;

    for (auto _ : state) {
        kaguya::Pipeline pipeline(model);
        kaguya::Sampler sampler(sp);
        pipeline.prefill(prompt);
        total_tokens += prompt_len;
    }

    state.SetItemsProcessed(total_tokens);
}

BENCHMARK(bench_pipeline_prefill)
    ->Args({64, 2, 2, 128, 8})
    ->Unit(benchmark::kMicrosecond)
    ->Iterations(10);

BENCHMARK(bench_pipeline_prefill)
    ->Args({64, 2, 2, 128, 32})
    ->Unit(benchmark::kMicrosecond)
    ->Iterations(5);

BENCHMARK(bench_pipeline_prefill)
    ->Args({256, 4, 4, 512, 8})
    ->Unit(benchmark::kMillisecond)
    ->Iterations(3);
