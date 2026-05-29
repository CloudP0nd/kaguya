#include "kaguya/cpu_features.h"
#include "kaguya/memory_manager.h"
#include "kaguya/thread_pool.h"
#include "kaguya/model.h"
#include "kaguya/model_loader.h"
#include "kaguya/tokenizer.h"
#include "kaguya/pipeline.h"
#include "kaguya/sampling.h"
#include "kaguya/kernels/dispatcher.h"

#include <iostream>
#include <string>
#include <sstream>
#include <cstdlib>
#include <chrono>
#include <vector>
#include <algorithm>
#include <csignal>

// ============================================================================
// CLI version info
// ============================================================================

static constexpr const char* KAGUYA_VERSION = "0.2.0";

// ============================================================================
// Global state for signal handling
// ============================================================================

static volatile sig_atomic_t g_interrupted = 0;

static void signal_handler(int /*sig*/) {
    g_interrupted = 1;
}

// ============================================================================
// Tokenizer wrapper — uses BPE if available, falls back to byte-level
// ============================================================================

/// Tokenizer interface that wraps either BpeTokenizer or byte-level fallback
class TokenizerWrapper {
public:
    TokenizerWrapper() = default;

    /// Set the BPE tokenizer (from model loader)
    void set_bpe(const kaguya::BpeTokenizer* bpe) {
        bpe_ = bpe;
    }

    /// Encode text to token IDs
    std::vector<int32_t> encode(const std::string& text, bool add_bos = true) const {
        if (bpe_ && bpe_->is_valid()) {
            return bpe_->encode(text, add_bos, false);
        }
        // Fallback: byte-level tokenizer
        std::vector<int32_t> tokens;
        tokens.reserve(text.size() + 1);
        if (add_bos) {
            tokens.push_back(0); // BOS at id=0 (convention for byte-level)
        }
        for (unsigned char c : text) {
            tokens.push_back(static_cast<int32_t>(c));
        }
        return tokens;
    }

    /// Decode token IDs back to text
    std::string decode(const std::vector<int32_t>& tokens, bool skip_special = true) const {
        if (bpe_ && bpe_->is_valid()) {
            return bpe_->decode(tokens, skip_special);
        }
        // Fallback: byte-level
        std::string result;
        result.reserve(tokens.size());
        for (int32_t t : tokens) {
            if (t >= 0 && t < 256) {
                result += static_cast<char>(t);
            }
        }
        return result;
    }

    /// Decode a single token
    std::string decode_token(int32_t token, bool skip_special = true) const {
        if (bpe_ && bpe_->is_valid()) {
            return bpe_->decode_token(token, skip_special);
        }
        if (token >= 0 && token < 256) {
            return std::string(1, static_cast<char>(token));
        }
        return "";
    }

    /// Get EOS token ID (-1 if not available)
    int32_t eos_token_id() const {
        if (bpe_ && bpe_->is_valid()) {
            return bpe_->eos_token_id();
        }
        return -1;
    }

    /// Check if BPE tokenizer is available
    bool has_bpe() const { return bpe_ && bpe_->is_valid(); }

private:
    const kaguya::BpeTokenizer* bpe_ = nullptr;
};

// ============================================================================
// Command-line argument parser
// ============================================================================

struct CliOptions {
    std::string model_path;
    int num_threads = 0;
    int n_predict = 128;
    float temperature = 0.8f;
    int top_k = 40;
    float top_p = 0.95f;
    float min_p = 0.0f;
    float repetition_penalty = 1.0f;
    int seed = -1;
    bool cpu_info_only = false;
    bool model_info_only = false;
    std::string prompt_text;
    bool interactive = false;
    bool stream = true;       // Stream tokens one by one
    bool bench_mode = false;  // Quick benchmark mode
    int bench_warmup = 3;
    int bench_iterations = 10;
};

static void print_usage(const char* prog) {
    std::cout << "Kaguya v" << KAGUYA_VERSION << " — CPU Inference Engine\n"
              << "\n"
              << "Usage: " << prog << " [options] <model.gguf>\n"
              << "\n"
              << "Options:\n"
              << "  -h, --help              Show this help\n"
              << "  -t, --threads N         Number of threads (default: auto)\n"
              << "  -n, --n-predict N       Number of tokens to predict (default: 128)\n"
              << "  -i, --interactive       Interactive chat mode\n"
              << "  -p, --prompt TEXT       Input prompt text (non-interactive)\n"
              << "  --stream / --no-stream  Stream output token-by-token (default: on)\n"
              << "\n"
              << "Sampling:\n"
              << "  --temp N                Temperature (default: 0.8)\n"
              << "  --top-k N               Top-K sampling (default: 40)\n"
              << "  --top-p N               Top-P sampling (default: 0.95)\n"
              << "  --min-p N               Min-P sampling (default: 0.0)\n"
              << "  --repeat-penalty N      Repetition penalty (default: 1.0)\n"
              << "  --seed N                Random seed (default: random)\n"
              << "\n"
              << "Info:\n"
              << "  --cpu-info              Print CPU feature detection and exit\n"
              << "  --model-info            Print model info and exit\n"
              << "\n"
              << "Benchmark:\n"
              << "  --bench                 Run inference benchmark\n"
              << "  --bench-warmup N        Warmup iterations (default: 3)\n"
              << "  --bench-iterations N    Benchmark iterations (default: 10)\n"
              << "\n";
}

static CliOptions parse_args(int argc, char** argv) {
    CliOptions opts;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "-h" || arg == "--help") {
            print_usage(argv[0]);
            std::exit(0);
        } else if (arg == "--cpu-info") {
            opts.cpu_info_only = true;
        } else if (arg == "--model-info") {
            opts.model_info_only = true;
        } else if ((arg == "-t" || arg == "--threads") && i + 1 < argc) {
            opts.num_threads = std::atoi(argv[++i]);
        } else if ((arg == "-n" || arg == "--n-predict") && i + 1 < argc) {
            opts.n_predict = std::atoi(argv[++i]);
        } else if (arg == "--temp" && i + 1 < argc) {
            opts.temperature = std::atof(argv[++i]);
        } else if (arg == "--top-k" && i + 1 < argc) {
            opts.top_k = std::atoi(argv[++i]);
        } else if (arg == "--top-p" && i + 1 < argc) {
            opts.top_p = std::atof(argv[++i]);
        } else if (arg == "--min-p" && i + 1 < argc) {
            opts.min_p = std::atof(argv[++i]);
        } else if (arg == "--repeat-penalty" && i + 1 < argc) {
            opts.repetition_penalty = std::atof(argv[++i]);
        } else if (arg == "--seed" && i + 1 < argc) {
            opts.seed = std::atoi(argv[++i]);
        } else if ((arg == "-p" || arg == "--prompt") && i + 1 < argc) {
            opts.prompt_text = argv[++i];
        } else if (arg == "-i" || arg == "--interactive") {
            opts.interactive = true;
        } else if (arg == "--stream") {
            opts.stream = true;
        } else if (arg == "--no-stream") {
            opts.stream = false;
        } else if (arg == "--bench") {
            opts.bench_mode = true;
        } else if (arg == "--bench-warmup" && i + 1 < argc) {
            opts.bench_warmup = std::atoi(argv[++i]);
        } else if (arg == "--bench-iterations" && i + 1 < argc) {
            opts.bench_iterations = std::atoi(argv[++i]);
        } else if (arg[0] != '-') {
            opts.model_path = arg;
        }
    }
    return opts;
}

// ============================================================================
// Model loading
// ============================================================================

static bool load_model(const CliOptions& opts, kaguya::ModelLoader& loader) {
    if (opts.model_path.empty()) {
        std::cerr << "Error: No model path specified.\n";
        return false;
    }

    std::cout << "Loading model: " << opts.model_path << " ...\n";
    auto load_start = std::chrono::steady_clock::now();

    if (!loader.load(opts.model_path)) {
        std::cerr << "Failed to load model: " << opts.model_path << "\n";
        return false;
    }

    auto load_end = std::chrono::steady_clock::now();
    auto load_ms = std::chrono::duration_cast<std::chrono::milliseconds>(load_end - load_start).count();

    const auto& model = loader.model();
    const auto& hp = model.hparams();

    std::cout << "Model loaded in " << load_ms << " ms\n";
    std::cout << "  Architecture: " << loader.arch_name() << "\n";
    std::cout << "  Dimensions:   " << hp.emb_dim << "d, " << hp.num_layers << " layers, "
              << hp.num_heads << " heads";
    if (hp.use_gqa) {
        std::cout << " (GQA: " << hp.num_kv_heads << " KV heads)";
    }
    std::cout << "\n";
    std::cout << "  Context:      " << hp.context_length << " tokens\n";
    std::cout << "  FFN dim:      " << hp.ffn_dim << "\n";
    std::cout << "  Vocab size:   " << hp.vocab_size << "\n\n";

    return true;
}

// ============================================================================
// Generation with streaming
// ============================================================================

struct GenerationResult {
    std::vector<int32_t> tokens;
    double tokens_per_sec;
    int64_t elapsed_ms;
    int64_t prefill_ms;
};

static GenerationResult generate_streaming(
    kaguya::Pipeline& pipeline,
    kaguya::Sampler& sampler,
    const std::vector<int32_t>& prompt_tokens,
    int n_predict,
    bool stream,
    const TokenizerWrapper& tokenizer)
{
    GenerationResult result;
    result.tokens_per_sec = 0.0;
    result.elapsed_ms = 0;
    result.prefill_ms = 0;

    // Prefill phase
    auto prefill_start = std::chrono::steady_clock::now();
    pipeline.prefill(prompt_tokens);
    auto prefill_end = std::chrono::steady_clock::now();
    result.prefill_ms = std::chrono::duration_cast<std::chrono::milliseconds>(prefill_end - prefill_start).count();

    // Decode phase — token by token
    auto decode_start = std::chrono::steady_clock::now();

    int32_t eos_id = tokenizer.eos_token_id();

    for (int i = 0; i < n_predict; ++i) {
        if (g_interrupted) break;
        if (pipeline.current_pos() >= pipeline.model().hparams().context_length) break;

        int32_t token = pipeline.decode(sampler);
        result.tokens.push_back(token);

        // Check for EOS
        if (eos_id >= 0 && token == eos_id) break;

        if (stream) {
            // Stream individual token
            std::string text = tokenizer.decode_token(token);
            std::cout << text << std::flush;
        }
    }

    auto decode_end = std::chrono::steady_clock::now();
    result.elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(decode_end - decode_start).count();

    if (!result.tokens.empty() && result.elapsed_ms > 0) {
        result.tokens_per_sec = result.tokens.size() * 1000.0 / result.elapsed_ms;
    }

    if (stream && !result.tokens.empty()) {
        std::cout << "\n";
    }

    return result;
}

// ============================================================================
// Interactive chat mode
// ============================================================================

static void interactive_chat(
    kaguya::Pipeline& pipeline,
    kaguya::Sampler& sampler,
    CliOptions& opts,
    const TokenizerWrapper& tokenizer)
{
    std::cout << "=== Interactive Chat Mode ===\n";
    std::cout << "Type your prompt and press Enter to generate.\n";
    std::cout << "Commands: /quit, /reset, /help, /set <param> <value>\n\n";

    std::string line;
    kaguya::SamplingParams sp;
    sp.temperature = opts.temperature;
    sp.top_k = opts.top_k;
    sp.top_p = opts.top_p;
    sp.min_p = opts.min_p;
    sp.repetition_penalty = opts.repetition_penalty;
    sp.seed = opts.seed;

    while (true) {
        if (g_interrupted) break;

        std::cout << "> " << std::flush;
        if (!std::getline(std::cin, line)) break;
        if (line.empty()) continue;

        // Command parsing
        if (line == "/quit" || line == "/exit") {
            break;
        } else if (line == "/reset") {
            pipeline.reset();
            std::cout << "Context reset.\n\n";
            continue;
        } else if (line == "/help") {
            std::cout << "Commands:\n"
                      << "  /quit, /exit    Exit interactive mode\n"
                      << "  /reset          Reset conversation context\n"
                      << "  /help           Show this help\n"
                      << "  /set temp N     Set temperature\n"
                      << "  /set top-k N    Set top-K\n"
                      << "  /set top-p N    Set top-P\n"
                      << "  /set min-p N    Set min-P\n"
                      << "  /set n-predict N  Set max tokens to generate\n"
                      << "  /stats          Show current parameters\n\n";
            continue;
        } else if (line == "/stats") {
            std::cout << "  temperature:     " << sp.temperature << "\n"
                      << "  top_k:           " << sp.top_k << "\n"
                      << "  top_p:           " << sp.top_p << "\n"
                      << "  min_p:           " << sp.min_p << "\n"
                      << "  repetition_penalty: " << sp.repetition_penalty << "\n"
                      << "  n_predict:       " << opts.n_predict << "\n"
                      << "  position:        " << pipeline.current_pos() << " / "
                      << pipeline.model().hparams().context_length << "\n\n";
            continue;
        } else if (line.substr(0, 5) == "/set ") {
            std::istringstream iss(line.substr(5));
            std::string param;
            iss >> param;
            if (param == "temp" || param == "temperature") {
                iss >> sp.temperature;
                sampler = kaguya::Sampler(sp);
                std::cout << "  temperature = " << sp.temperature << "\n\n";
            } else if (param == "top-k") {
                iss >> sp.top_k;
                sampler = kaguya::Sampler(sp);
                std::cout << "  top_k = " << sp.top_k << "\n\n";
            } else if (param == "top-p") {
                iss >> sp.top_p;
                sampler = kaguya::Sampler(sp);
                std::cout << "  top_p = " << sp.top_p << "\n\n";
            } else if (param == "min-p") {
                iss >> sp.min_p;
                sampler = kaguya::Sampler(sp);
                std::cout << "  min_p = " << sp.min_p << "\n\n";
            } else if (param == "n-predict" || param == "n_predict") {
                int np;
                iss >> np;
                opts.n_predict = np;
                std::cout << "  n_predict = " << np << "\n\n";
            } else {
                std::cout << "  Unknown parameter: " << param << "\n\n";
            }
            continue;
        }

        // Tokenize and generate
        auto prompt_tokens = tokenizer.encode(line, pipeline.current_pos() == 0);

        auto gen_result = generate_streaming(pipeline, sampler, prompt_tokens,
                                              opts.n_predict, opts.stream, tokenizer);

        // Print performance stats
        std::cout << "\n[" << gen_result.tokens.size() << " tokens, "
                  << gen_result.elapsed_ms << " ms, "
                  << std::fixed;
        if (gen_result.tokens_per_sec > 0) {
            std::cout.precision(1);
            std::cout << gen_result.tokens_per_sec << " tok/s";
        } else {
            std::cout << "--- tok/s";
        }
        std::cout << " | pos " << pipeline.current_pos() << "/"
                  << pipeline.model().hparams().context_length << "]\n\n";
    }
}

// ============================================================================
// Benchmark mode
// ============================================================================

static void run_benchmark(
    kaguya::Pipeline& pipeline,
    kaguya::Sampler& sampler,
    const CliOptions& opts,
    const TokenizerWrapper& /*tokenizer*/)
{
    std::cout << "=== Inference Benchmark ===\n\n";

    const auto& hp = pipeline.model().hparams();
    const int bench_tokens = 32;

    std::cout << "Warmup: " << opts.bench_warmup << " iterations (" << bench_tokens << " tokens each)...\n";

    // Warmup
    for (int i = 0; i < opts.bench_warmup; ++i) {
        pipeline.reset();
        pipeline.generate({0}, bench_tokens, sampler);
    }

    std::cout << "Benchmark: " << opts.bench_iterations << " iterations...\n\n";

    // Benchmark runs
    std::vector<double> tok_per_sec_list;
    tok_per_sec_list.reserve(static_cast<size_t>(opts.bench_iterations));

    for (int i = 0; i < opts.bench_iterations; ++i) {
        pipeline.reset();

        auto start = std::chrono::steady_clock::now();
        auto tokens = pipeline.generate({0}, bench_tokens, sampler);
        auto end = std::chrono::steady_clock::now();

        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        double tps = (ms > 0) ? (tokens.size() * 1000.0 / ms) : 0.0;
        tok_per_sec_list.push_back(tps);

        std::cout << "  Run " << (i + 1) << ": " << tokens.size() << " tokens in "
                  << ms << " ms (" << std::fixed;
        std::cout.precision(1);
        std::cout << tps << " tok/s)\n";
    }

    // Statistics
    std::sort(tok_per_sec_list.begin(), tok_per_sec_list.end());

    double avg = 0;
    for (double v : tok_per_sec_list) avg += v;
    avg /= static_cast<double>(tok_per_sec_list.size());

    double median = tok_per_sec_list[tok_per_sec_list.size() / 2];
    double min_val = tok_per_sec_list.front();
    double max_val = tok_per_sec_list.back();

    std::cout << "\n--- Results ---\n";
    std::cout << "  Average: " << std::fixed;
    std::cout.precision(1);
    std::cout << avg << " tok/s\n";
    std::cout << "  Median:  " << median << " tok/s\n";
    std::cout << "  Min:     " << min_val << " tok/s\n";
    std::cout << "  Max:     " << max_val << " tok/s\n";
    std::cout << "\n  Model: " << hp.emb_dim << "d, " << hp.num_layers << " layers, "
              << hp.num_heads << " heads\n";
    std::cout << "  Tokens per run: " << bench_tokens << "\n";
    std::cout << "  Kernel target: ";
    auto target = kaguya::select_kernel_target();
    switch (target) {
        case kaguya::KernelTarget::AVX512: std::cout << "AVX-512"; break;
        case kaguya::KernelTarget::AVX2:   std::cout << "AVX2"; break;
        case kaguya::KernelTarget::Scalar:  std::cout << "Scalar"; break;
        case kaguya::KernelTarget::AMX:     std::cout << "AMX"; break;
    }
    std::cout << "\n\n";
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char** argv) {
    // Install signal handlers
    std::signal(SIGINT, signal_handler);
    std::signal(SIGTERM, signal_handler);

    auto opts = parse_args(argc, argv);

    // Initialize core systems
    kaguya::CpuFeatureDetector::detect();
    kaguya::MemoryManager::init();

    // CPU info mode
    if (opts.cpu_info_only) {
        std::cout << "Kaguya v" << KAGUYA_VERSION << " — CPU Info\n";
        std::cout << "================================\n\n";
        std::cout << kaguya::CpuFeatureDetector::summary();
        return 0;
    }

    if (opts.model_path.empty()) {
        print_usage(argv[0]);
        return 1;
    }

    // Print banner
    std::cout << "Kaguya v" << KAGUYA_VERSION << " — CPU Inference Engine\n";
    std::cout << "====================================\n\n";

    // Load model
    kaguya::ModelLoader loader;
    if (!load_model(opts, loader)) {
        return 1;
    }

    // Model info mode
    if (opts.model_info_only) {
        loader.print_info();
        return 0;
    }

    const auto& model = loader.model();

    // Set up inference pipeline
    kaguya::Pipeline pipeline(model);

    // Configure sampler
    kaguya::SamplingParams sampling_params;
    sampling_params.temperature = opts.temperature;
    sampling_params.top_k = opts.top_k;
    sampling_params.top_p = opts.top_p;
    sampling_params.min_p = opts.min_p;
    sampling_params.repetition_penalty = opts.repetition_penalty;
    sampling_params.seed = opts.seed;
    kaguya::Sampler sampler(sampling_params);

    // Set up tokenizer (BPE from model, with byte-level fallback)
    TokenizerWrapper tokenizer;
    tokenizer.set_bpe(&loader.tokenizer());

    if (tokenizer.has_bpe()) {
        std::cout << "Tokenizer: BPE (vocab_size=" << loader.tokenizer().vocab_size();
        if (loader.tokenizer().bos_token_id() >= 0) {
            std::cout << ", BOS=" << loader.tokenizer().bos_token_id();
        }
        if (loader.tokenizer().eos_token_id() >= 0) {
            std::cout << ", EOS=" << loader.tokenizer().eos_token_id();
        }
        std::cout << ")\n\n";
    } else {
        std::cout << "Tokenizer: byte-level (no BPE metadata found)\n\n";
    }

    // Benchmark mode
    if (opts.bench_mode) {
        run_benchmark(pipeline, sampler, opts, tokenizer);
        return 0;
    }

    // Interactive mode
    if (opts.interactive) {
        interactive_chat(pipeline, sampler, opts, tokenizer);
        return 0;
    }

    // Single-shot generation mode (with -p / --prompt)
    if (!opts.prompt_text.empty()) {
        std::cout << "Prompt: " << opts.prompt_text << "\n\n";

        auto prompt_tokens = tokenizer.encode(opts.prompt_text);

        auto gen_result = generate_streaming(pipeline, sampler, prompt_tokens,
                                              opts.n_predict, opts.stream, tokenizer);

        // Performance stats
        std::cout << "\n--- Statistics ---\n";
        std::cout << "  Prefill:  " << gen_result.prefill_ms << " ms\n";
        std::cout << "  Decode:   " << gen_result.tokens.size() << " tokens in "
                  << gen_result.elapsed_ms << " ms\n";
        if (gen_result.tokens_per_sec > 0) {
            std::cout << std::fixed;
            std::cout.precision(1);
            std::cout << "  Speed:    " << gen_result.tokens_per_sec << " tok/s\n";
        }
        std::cout << "\n";
        return 0;
    }

    // Default: generate with dummy prompt (basic demo)
    {
        std::cout << "Generating " << opts.n_predict << " tokens...\n\n";

        std::vector<int32_t> prompt_tokens;
        if (tokenizer.has_bpe()) {
            prompt_tokens = tokenizer.encode("", true); // Just BOS token
        } else {
            prompt_tokens = {0}; // BOS for byte-level
        }
        auto gen_result = generate_streaming(pipeline, sampler, prompt_tokens,
                                              opts.n_predict, opts.stream, tokenizer);

        // Performance stats
        std::cout << "\n--- Statistics ---\n";
        std::cout << "  Prefill:  " << gen_result.prefill_ms << " ms\n";
        std::cout << "  Decode:   " << gen_result.tokens.size() << " tokens in "
                  << gen_result.elapsed_ms << " ms\n";
        if (gen_result.tokens_per_sec > 0) {
            std::cout << std::fixed;
            std::cout.precision(1);
            std::cout << "  Speed:    " << gen_result.tokens_per_sec << " tok/s\n";
        }
        std::cout << "\n";
    }

    return 0;
}
