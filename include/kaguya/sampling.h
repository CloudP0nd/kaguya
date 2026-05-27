#pragma once

#include <vector>
#include <cstdint>
#include <random>

namespace kaguya {

/// Sampling parameters
struct SamplingParams {
    float temperature      = 1.0f;
    int   top_k           = 40;
    float top_p           = 0.95f;
    float min_p           = 0.0f;
    float repetition_penalty = 1.0f;
    int   repetition_last_n  = 64;
    int   seed            = -1;  ///< -1 = random
};

/// Token sampler
class Sampler {
public:
    explicit Sampler(const SamplingParams& params = {});

    /// Sample a single token from logits
    int32_t sample(const float* logits, int64_t vocab_size,
                   const std::vector<int32_t>& recent_tokens);

    /// Apply temperature scaling
    static void apply_temperature(float* logits, int64_t vocab_size, float temperature);

    /// Apply repetition penalty
    static void apply_repetition_penalty(float* logits, int64_t vocab_size,
                                          const std::vector<int32_t>& recent_tokens,
                                          float penalty, int last_n);

    /// Top-K filtering (sets logits below top-k to -inf)
    static void apply_top_k(float* logits, int64_t vocab_size, int top_k);

    /// Top-P (nucleus) filtering
    static void apply_top_p(float* logits, int64_t vocab_size, float top_p);

    /// Min-P filtering
    static void apply_min_p(float* logits, int64_t vocab_size, float min_p);

    /// Argmax (greedy) selection
    static int32_t argmax(const float* logits, int64_t vocab_size);

private:
    SamplingParams params_;
    std::mt19937 rng_;
};

} // namespace kaguya
