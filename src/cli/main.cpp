#include "kaguya/cpu_features.h"
#include "kaguya/memory_manager.h"
#include "kaguya/thread_pool.h"
#include "kaguya/model.h"
#include "kaguya/model_loader.h"
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
    bool cpu_info_only = false;
    bool model_info_only = false;

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

    std::cout << "Model loaded successfully.\n";
    // Inference loop will be added in Phase 4

    return 0;
}
