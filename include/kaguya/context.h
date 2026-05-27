#pragma once

#include <memory>
#include <vector>
#include <cstdint>
#include "kaguya/model.h"

namespace kaguya {

/// Inference context — holds generation state
class Context {
public:
    explicit Context(const Model& model);
    ~Context() = default;

    /// Maximum context length
    int64_t context_length() const { return hparams_.context_length; }

    /// Current position in the sequence
    int64_t current_pos() const { return pos_; }

    /// Reset the context (start new sequence)
    void reset();

    /// Advance position by n tokens
    void advance(int64_t n) { pos_ += n; }

    /// Model reference
    const Model& model() const { return model_; }

private:
    const Model& model_;
    const HyperParams& hparams_;
    int64_t pos_ = 0;
};

} // namespace kaguya
