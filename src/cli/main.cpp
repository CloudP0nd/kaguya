#include "kaguya/cpu_features.h"
#include "kaguya/memory_manager.h"
#include "kaguya/thread_pool.h"
#include "kaguya/model.h"
#include "kaguya/model_loader.h"
#include "kaguya/pipeline.h"
#include "kaguya/sampling.h"

#include <iostream>
#include <string>
#include <cstdlib>

static void print_usage(const char* prog) {
    std::cout << "Usage: " << prog << " [options] <model.gguf>\n"
              << "\nOptions:\n"
              << "  -h, --help          Show this help\n"
              << "  -t, --threads N     Number of threads (default: auto)\n"
              << "  -n, --n-predict N   Number of tokens to predict (default: 128)\n"
              << "  --temp N            Temperature (default: 0.8)\n"
              << "  --top-k N           Top-K sampling (default: 40)\n"
              << "  --top-p N           Top-P sampling (default: 0.95)\n"
              << "  --cpu-info          Print CPU feature detection and exit\n"
              << "  --model-info        Print model info and exit\n"
              << "  -p, --prompt TEXT   Input prompt text\n"
              << "\n";
}

int main(int argc, char** argv) {
    if (argc < 2) {
        print_usage(argv[0]);
        return 1;
    }

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

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "-h" || arg == "--help") {
            print_usage(argv[0]);
            return 0;
        } else if (arg == "--cpu-info") {
            cpu_info_only = true;
        } else if (arg == "--model-info") {
            model_info_only = true;
        } else if ((arg == "-t" || arg == "--threads") && i + 1 < argc) {
            num_threads = std::atoi(argv[++i]);
        } else if ((arg == "-n" || arg == "--n-predict") && i + 1 < argc) {
            n_predict = std::atoi(argv[++i]);
        } else if (arg == "--temp" && i + 1 < argc) {
            temperature = std::atof(argv[++i]);
        } else if (arg == "--top-k" && i + 1 < argc) {
            top_k = std::atoi(argv[++i]);
        } else if (arg == "--top-p" && i + 1 < argc) {
            top_p = std::atof(argv[++i]);
        } else if (arg == "--min-p" && i + 1 < argc) {
            min_p = std::atof(argv[++i]);
        } else if (arg == "--repeat-penalty" && i + 1 < argc) {
            repetition_penalty = std::atof(argv[++i]);
        } else if (arg == "--seed" && i + 1 < argc) {
            seed = std::atoi(argv[++i]);
        } else if ((arg == "-p" || arg == "--prompt") && i + 1 < argc) {
            prompt_text = argv[++i];
        } else if (arg[0] != '-') {
            model_path = arg;
        }
    }

    // Initialize core systems
    kaguya::CpuFeatureDetector::detect();
    kaguya::MemoryManager::init();

    // CPU info mode
    if (cpu_info_only) {
        std::cout << kaguya::CpuFeatureDetector::summary();
        return 0;
    }

    if (model_path.empty()) {
        std::cerr << "Error: No model path specified.\n";
        return 1;
    }

    std::cout << "Kaguya v0.1.0 — CPU Inference Engine\n";
    std::cout << "====================================\n\n";
    std::cout << kaguya::CpuFeatureDetector::summary() << "\n";

    // Load model using ModelLoader
    kaguya::ModelLoader loader;
    if (!loader.load(model_path)) {
        std::cerr << "Failed to load model: " << model_path << "\n";
        return 1;
    }

    if (model_info_only) {
        loader.print_info();
        return 0;
    }

    const auto& model = loader.model();
    const auto& hp = model.hparams();

    std::cout << "Model: " << loader.arch_name() << " "
              << hp.emb_dim << "d " << hp.num_layers << "L "
              << hp.num_heads << "H";
    if (hp.use_gqa) {
        std::cout << " (GQA: " << hp.num_kv_heads << " KV heads)";
    }
    std::cout << "\n\n";

    // Set up inference pipeline
    kaguya::Pipeline pipeline(model);

    // Configure sampler
    kaguya::SamplingParams sampling_params;
    sampling_params.temperature = temperature;
    sampling_params.top_k = top_k;
    sampling_params.top_p = top_p;
    sampling_params.min_p = min_p;
    sampling_params.repetition_penalty = repetition_penalty;
    sampling_params.seed = seed;
    kaguya::Sampler sampler(sampling_params);

    // For now, use dummy prompt tokens (no tokenizer yet — Phase 5)
    // In a full implementation, the prompt text would be tokenized here.
    // We'll generate from a single start token as a demonstration.
    std::vector<int32_t> prompt_tokens = {0}; // BOS token

    std::cout << "Generating " << n_predict << " tokens...\n\n";

    auto start_time = std::chrono::steady_clock::now();

    auto tokens = pipeline.generate(prompt_tokens, n_predict, sampler);

    auto end_time = std::chrono::steady_clock::now();
    auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();

    // Output generated token IDs
    std::cout << "Generated tokens: ";
    for (size_t i = 0; i < tokens.size(); ++i) {
        if (i > 0) std::cout << " ";
        std::cout << tokens[i];
    }
    std::cout << "\n\n";

    // Performance stats
    double tokens_per_sec = (elapsed_ms > 0) ? (tokens.size() * 1000.0 / elapsed_ms) : 0.0;
    std::cout << "Performance: " << tokens.size() << " tokens in " << elapsed_ms << " ms";
    if (tokens_per_sec > 0) {
        std::cout << " (" << tokens_per_sec << " tokens/s)";
    }
    std::cout << "\n";

    return 0;
}
