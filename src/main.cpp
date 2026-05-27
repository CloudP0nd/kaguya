// Kaguya — CLI entry point
#include "kaguya/cpu_features.h"
#include "kaguya/memory/memory_manager.h"
#include "kaguya/model/loader.h"

#include <iostream>
#include <string>
#include <cstdlib>

static void print_usage(const char* prog) {
    std::cout << "Usage: " << prog << " [options] <model.gguf>\n"
              << "\nOptions:\n"
              << "  -h, --help          Show this help\n"
              << "  --cpu-info          Print CPU feature detection and exit\n"
              << "  --model-info        Print model info and exit\n\n";
}

int main(int argc, char** argv) {
    if (argc < 2) { print_usage(argv[0]); return 1; }

    std::string model_path;
    bool cpu_info_only = false;
    bool model_info_only = false;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "-h" || arg == "--help") { print_usage(argv[0]); return 0; }
        else if (arg == "--cpu-info") { cpu_info_only = true; }
        else if (arg == "--model-info") { model_info_only = true; }
        else if (arg[0] != '-') { model_path = arg; }
    }

    kaguya::CpuFeatureDetector::detect();
    kaguya::MemoryManager::init();

    if (cpu_info_only) {
        std::cout << kaguya::CpuFeatureDetector::summary();
        return 0;
    }

    std::cout << "Kaguya v0.1.0 — CPU Inference Engine\n====================================\n\n";
    std::cout << kaguya::CpuFeatureDetector::summary() << "\n";

    if (model_path.empty()) {
        std::cerr << "Error: No model path specified.\n";
        return 1;
    }

    kaguya::ModelLoader loader;
    if (!loader.load(model_path)) {
        std::cerr << "Failed to load model.\n";
        return 1;
    }

    if (model_info_only) {
        loader.print_info();
        return 0;
    }

    std::cout << "Model loaded successfully.\n";
    return 0;
}
