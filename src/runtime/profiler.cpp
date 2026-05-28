#include "kaguya/profiler.h"
#include "kaguya/pipeline.h"
#include "kaguya/memory_manager.h"
#include "kaguya/kernels/special_ops.h"

#include <cstring>
#include <cmath>
#include <algorithm>
#include <sstream>
#include <iomanip>

namespace kaguya {

// ============================================================================
// MemoryBandwidthProfiler
// ============================================================================

/// Noinline read accumulator to prevent compiler optimization
__attribute__((noinline))
static float accumulate_read(const float* data, size_t n) {
    float sum = 0.0f;
    for (size_t i = 0; i < n; ++i) {
        sum += data[i];
    }
    // Use volatile store to prevent the entire computation from being optimized away
    volatile float sink = sum;
    (void)sink;
    return sum;
}

/// Noinline write filler to prevent compiler optimization
__attribute__((noinline))
static void accumulate_write(float* data, size_t n, float val) {
    for (size_t i = 0; i < n; ++i) {
        data[i] = val;
    }
}

/// Noinline copy to prevent compiler optimization
__attribute__((noinline))
static void accumulate_copy(float* dst, const float* src, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        dst[i] = src[i];
    }
}

BandwidthResult MemoryBandwidthProfiler::measure_bandwidth(double buffer_size_mb) {
    BandwidthResult result{0.0, 0.0, 0.0};

    const size_t buffer_size_bytes = static_cast<size_t>(buffer_size_mb * 1024.0 * 1024.0);
    const size_t n_elements = buffer_size_bytes / sizeof(float);

    // Allocate buffer using MemoryManager with Aligned64
    float* buf1 = static_cast<float*>(MemoryManager::allocate(
        buffer_size_bytes, MemFlags::Aligned64));
    float* buf2 = static_cast<float*>(MemoryManager::allocate(
        buffer_size_bytes, MemFlags::Aligned64));

    if (!buf1 || !buf2) {
        // Fallback to aligned_alloc
        if (buf1) MemoryManager::deallocate(buf1, buffer_size_bytes);
        if (buf2) MemoryManager::deallocate(buf2, buffer_size_bytes);

        buf1 = static_cast<float*>(std::aligned_alloc(64, buffer_size_bytes));
        buf2 = static_cast<float*>(std::aligned_alloc(64, buffer_size_bytes));

        if (!buf1 || !buf2) {
            if (buf1) std::free(buf1);
            if (buf2) std::free(buf2);
            return result;
        }
    }

    // Initialize buffer with non-zero data
    for (size_t i = 0; i < n_elements; ++i) {
        buf1[i] = static_cast<float>(i % 256);
        buf2[i] = 0.0f;
    }

    const int n_iterations = 5;
    const double total_bytes = static_cast<double>(buffer_size_bytes);

    // Measure read bandwidth
    {
        auto start = std::chrono::steady_clock::now();
        volatile float sink = 0.0f;
        for (int iter = 0; iter < n_iterations; ++iter) {
            sink = accumulate_read(buf1, n_elements);
        }
        (void)sink;
        auto end = std::chrono::steady_clock::now();
        double elapsed_s = std::chrono::duration<double>(end - start).count();
        result.read_gb_s = (total_bytes * n_iterations / (1024.0 * 1024.0 * 1024.0)) / elapsed_s;
    }

    // Measure write bandwidth
    {
        auto start = std::chrono::steady_clock::now();
        for (int iter = 0; iter < n_iterations; ++iter) {
            accumulate_write(buf2, n_elements, static_cast<float>(iter));
        }
        auto end = std::chrono::steady_clock::now();
        double elapsed_s = std::chrono::duration<double>(end - start).count();
        result.write_gb_s = (total_bytes * n_iterations / (1024.0 * 1024.0 * 1024.0)) / elapsed_s;
    }

    // Measure copy bandwidth (read + write)
    {
        auto start = std::chrono::steady_clock::now();
        for (int iter = 0; iter < n_iterations; ++iter) {
            accumulate_copy(buf2, buf1, n_elements);
        }
        auto end = std::chrono::steady_clock::now();
        double elapsed_s = std::chrono::duration<double>(end - start).count();
        // Copy involves both read and write, report as 2x bytes transferred
        result.copy_gb_s = (2.0 * total_bytes * n_iterations / (1024.0 * 1024.0 * 1024.0)) / elapsed_s;
    }

    // Cleanup
    MemoryManager::deallocate(buf1, buffer_size_bytes);
    MemoryManager::deallocate(buf2, buffer_size_bytes);

    return result;
}

// ============================================================================
// InferenceProfiler
// ============================================================================

void InferenceProfiler::profile_layer(int64_t layer, std::chrono::nanoseconds duration) {
    auto& t = timings_[layer];
    t.total += duration;
    t.layer_count++;
}

void InferenceProfiler::profile_attention(int64_t layer, std::chrono::nanoseconds duration) {
    auto& t = timings_[layer];
    t.attention += duration;
    t.attention_count++;
}

void InferenceProfiler::profile_ffn(int64_t layer, std::chrono::nanoseconds duration) {
    auto& t = timings_[layer];
    t.ffn += duration;
    t.ffn_count++;
}

std::string InferenceProfiler::summary() const {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(3);

    oss << "=== Inference Profile Summary ===\n";
    oss << std::setw(8) << "Layer"
        << std::setw(14) << "Total(ms)"
        << std::setw(14) << "Attn(ms)"
        << std::setw(14) << "FFN(ms)"
        << std::setw(14) << "Other(ms)"
        << "\n";

    std::chrono::nanoseconds grand_total{0};
    std::chrono::nanoseconds grand_attention{0};
    std::chrono::nanoseconds grand_ffn{0};

    // Collect all layer indices and sort
    std::vector<int64_t> layers;
    layers.reserve(timings_.size());
    for (const auto& [layer_id, _] : timings_) {
        layers.push_back(layer_id);
    }
    std::sort(layers.begin(), layers.end());

    for (const auto& layer_id : layers) {
        const auto& t = timings_.at(layer_id);
        double total_ms = std::chrono::duration<double, std::milli>(t.total).count();
        double attn_ms = std::chrono::duration<double, std::milli>(t.attention).count();
        double ffn_ms = std::chrono::duration<double, std::milli>(t.ffn).count();
        double other_ms = total_ms - attn_ms - ffn_ms;

        oss << std::setw(8) << layer_id
            << std::setw(14) << total_ms
            << std::setw(14) << attn_ms
            << std::setw(14) << ffn_ms
            << std::setw(14) << other_ms
            << "\n";

        grand_total += t.total;
        grand_attention += t.attention;
        grand_ffn += t.ffn;
    }

    double gt_ms = std::chrono::duration<double, std::milli>(grand_total).count();
    double ga_ms = std::chrono::duration<double, std::milli>(grand_attention).count();
    double gf_ms = std::chrono::duration<double, std::milli>(grand_ffn).count();
    double go_ms = gt_ms - ga_ms - gf_ms;

    oss << std::string(64, '-') << "\n";
    oss << std::setw(8) << "TOTAL"
        << std::setw(14) << gt_ms
        << std::setw(14) << ga_ms
        << std::setw(14) << gf_ms
        << std::setw(14) << go_ms
        << "\n";

    if (gt_ms > 0.0) {
        oss << std::setw(8) << "%"
            << std::setw(14) << "100.0"
            << std::setw(14) << (ga_ms / gt_ms * 100.0)
            << std::setw(14) << (gf_ms / gt_ms * 100.0)
            << std::setw(14) << (go_ms / gt_ms * 100.0)
            << "\n";
    }

    return oss.str();
}

void InferenceProfiler::reset() {
    timings_.clear();
}

// ============================================================================
// Perplexity computation
// ============================================================================

/// Helper: apply softmax to logits and return probabilities
static void softmax_for_ppl(float* logits, int64_t n) {
    // Find max for numerical stability
    float max_val = logits[0];
    for (int64_t i = 1; i < n; ++i) {
        if (logits[i] > max_val) max_val = logits[i];
    }

    double sum = 0.0;
    for (int64_t i = 0; i < n; ++i) {
        logits[i] = std::exp(logits[i] - max_val);
        sum += static_cast<double>(logits[i]);
    }

    float inv_sum = 1.0f / static_cast<float>(sum);
    for (int64_t i = 0; i < n; ++i) {
        logits[i] *= inv_sum;
    }
}

double compute_perplexity(Pipeline& pipeline, const std::vector<int32_t>& tokens) {
    if (tokens.size() < 2) {
        return std::numeric_limits<double>::infinity();
    }

    pipeline.reset();

    double log_prob_sum = 0.0;
    const int64_t N = static_cast<int64_t>(tokens.size());

    // Process tokens one at a time
    // After processing token[i], the logits represent the model's prediction for token[i+1]
    for (int64_t i = 0; i < N - 1; ++i) {
        // Process one token
        pipeline.prefill({tokens[i]});

        // Get logits (these predict the next token)
        const auto& logits = pipeline.logits();
        const int64_t vocab_size = static_cast<int64_t>(logits.size());

        if (vocab_size == 0) continue;

        // Copy logits to apply softmax (we don't want to modify the pipeline's internal state)
        std::vector<float> probs(logits.begin(), logits.end());
        softmax_for_ppl(probs.data(), vocab_size);

        // Extract probability of the actual next token
        int32_t next_token = tokens[i + 1];
        if (next_token >= 0 && next_token < vocab_size) {
            float p = probs[next_token];
            if (p > 0.0f) {
                log_prob_sum += std::log(static_cast<double>(p));
            } else {
                log_prob_sum += -30.0; // Log of very small probability
            }
        } else {
            log_prob_sum += -30.0;
        }
    }

    // PPL = exp(-1/N * sum(log(p)))
    const int64_t num_predictions = N - 1;
    double ppl = std::exp(-log_prob_sum / static_cast<double>(num_predictions));
    return ppl;
}

} // namespace kaguya
