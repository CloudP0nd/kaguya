#include "kaguya/batch.h"

#include <algorithm>

namespace kaguya {

BatchInference::BatchInference(const Model& model, int64_t max_batch_size)
    : model_(model), max_batch_size_(max_batch_size) {}

int BatchInference::add_sequence(const std::vector<int32_t>& prompt, int32_t n_predict) {
    if (static_cast<int64_t>(sequences_.size()) >= max_batch_size_) {
        return -1; // Batch full
    }

    int id = static_cast<int>(sequences_.size());

    BatchSequence seq;
    seq.prompt = prompt;
    seq.n_predict = n_predict;
    seq.finished = false;
    sequences_.push_back(std::move(seq));

    // Create pipeline for this sequence
    auto pipeline = std::make_unique<Pipeline>(model_);
    pipeline->prefill(prompt);
    pipelines_.push_back(std::move(pipeline));

    return id;
}

void BatchInference::step(Sampler& sampler) {
    for (int i = 0; i < static_cast<int>(sequences_.size()); ++i) {
        auto& seq = sequences_[static_cast<size_t>(i)];
        if (seq.finished) continue;

        // Generate one token
        int32_t token = pipelines_[static_cast<size_t>(i)]->decode(sampler);
        seq.output.push_back(token);

        // Check if finished
        if (static_cast<int32_t>(seq.output.size()) >= seq.n_predict) {
            seq.finished = true;
        }
    }
}

void BatchInference::run_all(Sampler& sampler) {
    while (!all_finished()) {
        step(sampler);
    }
}

const BatchSequence& BatchInference::sequence(int id) const {
    return sequences_[static_cast<size_t>(id)];
}

int BatchInference::num_active() const {
    int count = 0;
    for (const auto& seq : sequences_) {
        if (!seq.finished) ++count;
    }
    return count;
}

} // namespace kaguya
