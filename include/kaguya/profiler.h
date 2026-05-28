#pragma once
/// @file profiler.h
/// @brief Phase 6 Memory bandwidth profiler and inference profiler.
///
/// Provides:
/// - MemoryBandwidthProfiler: measures sequential read/write/copy bandwidth
/// - InferenceProfiler: records per-layer timing for attention and FFN
/// - compute_perplexity: computes perplexity of a token sequence

#include <cstdint>
#include <string>
#include <chrono>
#include <unordered_map>
#include <vector>

namespace kaguya {

// Forward declaration
class Pipeline;

/// Result of a memory bandwidth measurement
struct BandwidthResult {
    double read_gb_s;   ///< Sequential read bandwidth (GB/s)
    double write_gb_s;  ///< Sequential write bandwidth (GB/s)
    double copy_gb_s;   ///< Sequential copy bandwidth (GB/s)
};

/// Measures memory bandwidth using sequential access patterns
class MemoryBandwidthProfiler {
public:
    /// Measure sequential read, write, and copy bandwidth
    /// @param buffer_size_mb Size of the buffer to use (default 64MB)
    /// @return Bandwidth results in GB/s
    static BandwidthResult measure_bandwidth(double buffer_size_mb = 64.0);
};

/// Records timing for individual transformer layer components
class InferenceProfiler {
public:
    InferenceProfiler() = default;

    /// Record timing for a full transformer layer
    /// @param layer Layer index
    /// @param duration Time taken for the layer
    void profile_layer(int64_t layer, std::chrono::nanoseconds duration);

    /// Record timing for attention computation
    /// @param layer Layer index
    /// @param duration Time taken for attention
    void profile_attention(int64_t layer, std::chrono::nanoseconds duration);

    /// Record timing for FFN computation
    /// @param layer Layer index
    /// @param duration Time taken for FFN
    void profile_ffn(int64_t layer, std::chrono::nanoseconds duration);

    /// Get a summary string with timing breakdown per layer
    /// @return Formatted string with per-layer and total timing
    std::string summary() const;

    /// Reset all recorded timings
    void reset();

private:
    struct LayerTiming {
        std::chrono::nanoseconds total{0};
        std::chrono::nanoseconds attention{0};
        std::chrono::nanoseconds ffn{0};
        int layer_count = 0;
        int attention_count = 0;
        int ffn_count = 0;
    };

    std::unordered_map<int64_t, LayerTiming> timings_;
};

/// Compute perplexity of a token sequence under the model
/// PPL = exp(-1/N * sum(log(p(token_i | context))))
/// For each position i, runs forward pass, applies softmax to logits,
/// and extracts probability of token_i.
/// @param pipeline The inference pipeline (will be reset internally)
/// @param tokens Token sequence to evaluate
/// @return Perplexity value (lower is better)
double compute_perplexity(Pipeline& pipeline, const std::vector<int32_t>& tokens);

} // namespace kaguya
