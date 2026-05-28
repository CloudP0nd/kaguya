#pragma once
/// @file batch.h
/// @brief Phase 4 Batch inference for multiple sequences.
///
/// Manages multiple independent sequences, each with its own
/// pipeline and KV cache. Processes sequences round-robin.

#include <vector>
#include <memory>
#include <cstdint>
#include "kaguya/pipeline.h"
#include "kaguya/sampling.h"

namespace kaguya {

/// A single sequence in a batch
struct BatchSequence {
    std::vector<int32_t> prompt;       ///< Input prompt tokens
    std::vector<int32_t> output;       ///< Generated output tokens
    int32_t n_predict = 128;           ///< Max tokens to generate
    bool finished = false;             ///< Whether generation is complete
};

/// Batch inference manager
class BatchInference {
public:
    /// Create batch inference for the given model
    /// @param model The model to use for inference
    /// @param max_batch_size Maximum number of concurrent sequences
    BatchInference(const Model& model, int64_t max_batch_size = 4);

    ~BatchInference() = default;

    /// Add a sequence to the batch
    /// @param prompt Input prompt tokens
    /// @param n_predict Number of tokens to generate
    /// @return Sequence ID
    int add_sequence(const std::vector<int32_t>& prompt, int32_t n_predict = 128);

    /// Run one step for all active sequences
    /// @param sampler Sampler to use for token selection
    void step(Sampler& sampler);

    /// Run until all sequences are finished
    /// @param sampler Sampler to use for token selection
    void run_all(Sampler& sampler);

    /// Get a sequence by ID
    const BatchSequence& sequence(int id) const;

    /// Number of sequences
    int num_sequences() const { return static_cast<int>(sequences_.size()); }

    /// Number of active (unfinished) sequences
    int num_active() const;

    /// Check if all sequences are finished
    bool all_finished() const { return num_active() == 0; }

private:
    const Model& model_;
    int64_t max_batch_size_;
    std::vector<BatchSequence> sequences_;
    std::vector<std::unique_ptr<Pipeline>> pipelines_;
};

} // namespace kaguya
