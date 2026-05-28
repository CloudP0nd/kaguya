#include "kaguya/kv_cache.h"

#include <cstring>

namespace kaguya {

KVCache::KVCache(int64_t n_layers, int64_t n_kv_heads, int64_t head_dim, int64_t max_seq_len)
    : n_layers_(n_layers),
      n_kv_heads_(n_kv_heads),
      head_dim_(head_dim),
      max_seq_len_(max_seq_len),
      n_positions_(0),
      layer_stride_(n_kv_heads * max_seq_len * head_dim),
      head_stride_(max_seq_len * head_dim),
      pos_stride_(head_dim)
{
    const int64_t total_elements = n_layers * layer_stride_;
    key_buf_.resize(static_cast<size_t>(total_elements), 0.0f);
    value_buf_.resize(static_cast<size_t>(total_elements), 0.0f);
}

float* KVCache::key(int64_t layer, int64_t kv_head, int64_t pos) {
    const int64_t offset = layer * layer_stride_ + kv_head * head_stride_ + pos * pos_stride_;
    return key_buf_.data() + offset;
}

const float* KVCache::key(int64_t layer, int64_t kv_head, int64_t pos) const {
    const int64_t offset = layer * layer_stride_ + kv_head * head_stride_ + pos * pos_stride_;
    return key_buf_.data() + offset;
}

float* KVCache::value(int64_t layer, int64_t kv_head, int64_t pos) {
    const int64_t offset = layer * layer_stride_ + kv_head * head_stride_ + pos * pos_stride_;
    return value_buf_.data() + offset;
}

const float* KVCache::value(int64_t layer, int64_t kv_head, int64_t pos) const {
    const int64_t offset = layer * layer_stride_ + kv_head * head_stride_ + pos * pos_stride_;
    return value_buf_.data() + offset;
}

float* KVCache::key_head(int64_t layer, int64_t kv_head) {
    const int64_t offset = layer * layer_stride_ + kv_head * head_stride_;
    return key_buf_.data() + offset;
}

const float* KVCache::key_head(int64_t layer, int64_t kv_head) const {
    const int64_t offset = layer * layer_stride_ + kv_head * head_stride_;
    return key_buf_.data() + offset;
}

float* KVCache::value_head(int64_t layer, int64_t kv_head) {
    const int64_t offset = layer * layer_stride_ + kv_head * head_stride_;
    return value_buf_.data() + offset;
}

const float* KVCache::value_head(int64_t layer, int64_t kv_head) const {
    const int64_t offset = layer * layer_stride_ + kv_head * head_stride_;
    return value_buf_.data() + offset;
}

void KVCache::advance(int64_t n) {
    n_positions_ += n;
}

void KVCache::reset() {
    n_positions_ = 0;
    std::memset(key_buf_.data(), 0, key_buf_.size() * sizeof(float));
    std::memset(value_buf_.data(), 0, value_buf_.size() * sizeof(float));
}

} // namespace kaguya
