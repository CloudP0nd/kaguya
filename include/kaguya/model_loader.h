#pragma once
// Kaguya — model_loader
// High-level model loader: GGUF file -> ModelWeights

#include <string>
#include <memory>
#include "kaguya/model.h"
#include "kaguya/gguf_loader.h"
#include "kaguya/tokenizer.h"

namespace kaguya {

/// Load a model from a GGUF file
class ModelLoader {
public:
    ModelLoader();
    ~ModelLoader();

    /// Load model from GGUF file path
    /// @param path Path to .gguf file
    /// @param mmap Use memory-mapped I/O
    /// @return true on success
    bool load(const std::string& path, bool mmap = true);

    /// Get the loaded model
    const Model& model() const { return model_; }

    /// Get the GGUF loader (for raw metadata access)
    const GgufLoader& gguf() const { return *gguf_; }

    /// Get the BPE tokenizer (built from GGUF metadata)
    const BpeTokenizer& tokenizer() const { return tokenizer_; }

    /// Get model architecture name
    std::string arch_name() const;

    /// Print model info
    void print_info() const;

private:
    std::unique_ptr<GgufLoader> gguf_;
    Model model_;
    ModelWeights weights_;  // Temporary storage during loading
    BpeTokenizer tokenizer_;

    /// Extract hyperparameters from GGUF metadata
    bool extract_hparams();

    /// Build model weight references from GGUF tensor data
    bool build_weight_refs();

    /// Build BPE tokenizer from GGUF metadata
    bool build_tokenizer();

    /// Helper: try multiple name variants for a tensor
    const GgufTensorInfo* find_tensor(const std::vector<std::string>& names);

    /// Get tensor data by name with shape info
    const void* get_tensor(const std::string& name, size_t& out_bytes,
                            int64_t& ne0, int64_t& ne1);
};

} // namespace kaguya
