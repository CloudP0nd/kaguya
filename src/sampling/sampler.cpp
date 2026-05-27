#include "kaguya/sampling.h"
#include <algorithm>
#include <cmath>
#include <numeric>

namespace kaguya {

Sampler::Sampler(const SamplingParams& params) : params_(params) {
    if (params_.seed >= 0) {
        rng_.seed(static_cast<uint32_t>(params_.seed));
    } else {
        std::random_device rd;
        rng_.seed(rd());
    }
}

int32_t Sampler::sample(const float* logits, int64_t vocab_size,
                          const std::vector<int32_t>& recent_tokens) {
    // Copy logits for manipulation
    std::vector<float> probs(logits, logits + vocab_size);

    // Apply repetition penalty
    if (params_.repetition_penalty != 1.0f && !recent_tokens.empty()) {
        apply_repetition_penalty(probs.data(), vocab_size, recent_tokens,
                                  params_.repetition_penalty, params_.repetition_last_n);
    }

    // Apply temperature
    if (params_.temperature != 1.0f) {
        apply_temperature(probs.data(), vocab_size, params_.temperature);
    }

    // Apply top-k
    if (params_.top_k > 0) {
        apply_top_k(probs.data(), vocab_size, params_.top_k);
    }

    // Apply top-p
    if (params_.top_p < 1.0f) {
        apply_top_p(probs.data(), vocab_size, params_.top_p);
    }

    // Apply min-p
    if (params_.min_p > 0.0f) {
        apply_min_p(probs.data(), vocab_size, params_.min_p);
    }

    // Greedy at temperature 0
    if (params_.temperature <= 0.0f) {
        return argmax(probs.data(), vocab_size);
    }

    // Softmax
    float max_val = *std::max_element(probs.begin(), probs.end());
    float sum = 0.0f;
    for (auto& p : probs) {
        p = std::exp(p - max_val);
        sum += p;
    }
    for (auto& p : probs) {
        p /= sum;
    }

    // Sample from distribution
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    float r = dist(rng_);
    float cumsum = 0.0f;
    for (int64_t i = 0; i < vocab_size; ++i) {
        cumsum += probs[i];
        if (cumsum >= r) return static_cast<int32_t>(i);
    }
    return static_cast<int32_t>(vocab_size - 1);
}

void Sampler::apply_temperature(float* logits, int64_t vocab_size, float temperature) {
    float inv_temp = 1.0f / temperature;
    for (int64_t i = 0; i < vocab_size; ++i) {
        logits[i] *= inv_temp;
    }
}

void Sampler::apply_repetition_penalty(float* logits, int64_t vocab_size,
                                          const std::vector<int32_t>& recent_tokens,
                                          float penalty, int last_n) {
    int start = std::max(0, static_cast<int>(recent_tokens.size()) - last_n);
    for (int i = start; i < static_cast<int>(recent_tokens.size()); ++i) {
        int32_t tok = recent_tokens[i];
        if (tok >= 0 && tok < vocab_size) {
            if (logits[tok] > 0) {
                logits[tok] /= penalty;
            } else {
                logits[tok] *= penalty;
            }
        }
    }
}

void Sampler::apply_top_k(float* logits, int64_t vocab_size, int top_k) {
    if (top_k >= vocab_size) return;

    // Find the top-k threshold
    std::vector<float> sorted(logits, logits + vocab_size);
    std::partial_sort(sorted.begin(), sorted.begin() + top_k, sorted.end(),
                       std::greater<float>());
    float threshold = sorted[top_k - 1];

    for (int64_t i = 0; i < vocab_size; ++i) {
        if (logits[i] < threshold) logits[i] = -INFINITY;
    }
}

void Sampler::apply_top_p(float* logits, int64_t vocab_size, float top_p) {
    // Softmax first
    float max_val = *std::max_element(logits, logits + vocab_size);
    std::vector<std::pair<float, int64_t>> indexed;
    indexed.reserve(vocab_size);
    float sum = 0.0f;
    for (int64_t i = 0; i < vocab_size; ++i) {
        if (logits[i] == -INFINITY) continue;
        float p = std::exp(logits[i] - max_val);
        sum += p;
        indexed.emplace_back(p, i);
    }
    for (auto& [p, _] : indexed) p /= sum;

    // Sort descending
    std::sort(indexed.begin(), indexed.end(), std::greater{});

    // Find cutoff
    float cumsum = 0.0f;
    float cutoff_prob = 0.0f;
    for (const auto& [p, idx] : indexed) {
        cumsum += p;
        if (cumsum > top_p) {
            cutoff_prob = p;
            break;
        }
    }

    // Filter
    for (int64_t i = 0; i < vocab_size; ++i) {
        float p = std::exp(logits[i] - max_val) / sum;
        if (p < cutoff_prob) logits[i] = -INFINITY;
    }
}

void Sampler::apply_min_p(float* logits, int64_t vocab_size, float min_p) {
    float max_val = *std::max_element(logits, logits + vocab_size);
    float threshold = max_val + std::log(min_p);
    for (int64_t i = 0; i < vocab_size; ++i) {
        if (logits[i] < threshold) logits[i] = -INFINITY;
    }
}

int32_t Sampler::argmax(const float* logits, int64_t vocab_size) {
    int32_t best = 0;
    float best_val = logits[0];
    for (int64_t i = 1; i < vocab_size; ++i) {
        if (logits[i] > best_val) {
            best_val = logits[i];
            best = static_cast<int32_t>(i);
        }
    }
    return best;
}

} // namespace kaguya
