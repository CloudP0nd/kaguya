#include "kaguya/kv_cache.h"

#include <cstring>
#include <cstdlib>

namespace kaguya {

/// HugePages are only beneficial for allocations >= 2MB
static constexpr size_t HUGEPAGES_THRESHOLD = 2UL * 1024UL * 1024UL;

KVCache::KVCache(int64_t n_layers, int64_t n_kv_heads, int64_t head_dim, int64_t max_seq_len)
    : n_layers_(n_layers),
      n_kv_heads_(n_kv_heads),
      head_dim_(head_dim),
      max_seq_len_(max_seq_len),
      n_positions_(0),
      layer_stride_(n_kv_heads * max_seq_len * head_dim),
      head_stride_(max_seq_len * head_dim),
      pos_stride_(head_dim),
      key_buf_(nullptr),
      key_buf_size_(0),
      value_buf_(nullptr),
      value_buf_size_(0),
      key_uses_memory_manager_(false),
      value_uses_memory_manager_(false)
{
    // Validate dimensions
    if (n_layers <= 0 || n_kv_heads <= 0 || head_dim <= 0 || max_seq_len <= 0) {
        throw std::invalid_argument(
            "KVCache: all dimensions must be positive (n_layers=" +
            std::to_string(n_layers) + ", n_kv_heads=" + std::to_string(n_kv_heads) +
            ", head_dim=" + std::to_string(head_dim) + ", max_seq_len=" + std::to_string(max_seq_len) + ")");
    }

    // Check for overflow: total_elements = n_layers * layer_stride
    const int64_t total_elements = n_layers * layer_stride_;
    if (total_elements / n_layers != layer_stride_ || total_elements < 0) {
        throw std::length_error(
            "KVCache: total size would overflow (n_layers=" + std::to_string(n_layers) +
            ", layer_stride=" + std::to_string(layer_stride_) + ")");
    }
    const size_t total_bytes = static_cast<size_t>(total_elements) * sizeof(float);

    // Initialize MemoryManager (safe to call multiple times)
    MemoryManager::init();

    // Allocate key buffer
    // Only request HugePages for large allocations (>= 2MB) because
    // MemoryManager::allocate() with HugePages rounds up to 2MB for mmap,
    // and the deallocate() heuristic relies on size >= 2MB to use munmap.
    if (total_bytes >= HUGEPAGES_THRESHOLD) {
        key_buf_ = static_cast<float*>(MemoryManager::allocate(
            total_bytes, MemFlags::HugePages | MemFlags::Aligned64));
    }
    if (key_buf_) {
        key_uses_memory_manager_ = true;
    } else {
        // Fallback: try Aligned64 only
        key_buf_ = static_cast<float*>(MemoryManager::allocate(
            total_bytes, MemFlags::Aligned64));
        if (key_buf_) {
            key_uses_memory_manager_ = true;
        } else {
            // Final fallback: aligned_alloc directly
            key_buf_ = static_cast<float*>(std::aligned_alloc(64, total_bytes));
            key_uses_memory_manager_ = false;
        }
    }
    key_buf_size_ = total_bytes;

    // Allocate value buffer: same strategy
    if (total_bytes >= HUGEPAGES_THRESHOLD) {
        value_buf_ = static_cast<float*>(MemoryManager::allocate(
            total_bytes, MemFlags::HugePages | MemFlags::Aligned64));
    }
    if (value_buf_) {
        value_uses_memory_manager_ = true;
    } else {
        value_buf_ = static_cast<float*>(MemoryManager::allocate(
            total_bytes, MemFlags::Aligned64));
        if (value_buf_) {
            value_uses_memory_manager_ = true;
        } else {
            value_buf_ = static_cast<float*>(std::aligned_alloc(64, total_bytes));
            value_uses_memory_manager_ = false;
        }
    }
    value_buf_size_ = total_bytes;

    // Zero-initialize buffers
    if (key_buf_) {
        std::memset(key_buf_, 0, total_bytes);
    }
    if (value_buf_) {
        std::memset(value_buf_, 0, total_bytes);
    }
}

KVCache::~KVCache() {
    if (key_buf_) {
        if (key_uses_memory_manager_) {
            MemoryManager::deallocate(key_buf_, key_buf_size_);
        } else {
            std::free(key_buf_);
        }
        key_buf_ = nullptr;
    }
    if (value_buf_) {
        if (value_uses_memory_manager_) {
            MemoryManager::deallocate(value_buf_, value_buf_size_);
        } else {
            std::free(value_buf_);
        }
        value_buf_ = nullptr;
    }
    key_buf_size_ = 0;
    value_buf_size_ = 0;
}

KVCache::KVCache(KVCache&& other) noexcept
    : n_layers_(other.n_layers_),
      n_kv_heads_(other.n_kv_heads_),
      head_dim_(other.head_dim_),
      max_seq_len_(other.max_seq_len_),
      n_positions_(other.n_positions_),
      layer_stride_(other.layer_stride_),
      head_stride_(other.head_stride_),
      pos_stride_(other.pos_stride_),
      key_buf_(other.key_buf_),
      key_buf_size_(other.key_buf_size_),
      value_buf_(other.value_buf_),
      value_buf_size_(other.value_buf_size_),
      key_uses_memory_manager_(other.key_uses_memory_manager_),
      value_uses_memory_manager_(other.value_uses_memory_manager_)
{
    other.key_buf_ = nullptr;
    other.key_buf_size_ = 0;
    other.value_buf_ = nullptr;
    other.value_buf_size_ = 0;
    other.key_uses_memory_manager_ = false;
    other.value_uses_memory_manager_ = false;
    other.n_positions_ = 0;
}

KVCache& KVCache::operator=(KVCache&& other) noexcept {
    if (this != &other) {
        // Deallocate current buffers
        if (key_buf_) {
            if (key_uses_memory_manager_) {
                MemoryManager::deallocate(key_buf_, key_buf_size_);
            } else {
                std::free(key_buf_);
            }
        }
        if (value_buf_) {
            if (value_uses_memory_manager_) {
                MemoryManager::deallocate(value_buf_, value_buf_size_);
            } else {
                std::free(value_buf_);
            }
        }

        // Move from other
        n_layers_ = other.n_layers_;
        n_kv_heads_ = other.n_kv_heads_;
        head_dim_ = other.head_dim_;
        max_seq_len_ = other.max_seq_len_;
        n_positions_ = other.n_positions_;
        layer_stride_ = other.layer_stride_;
        head_stride_ = other.head_stride_;
        pos_stride_ = other.pos_stride_;
        key_buf_ = other.key_buf_;
        key_buf_size_ = other.key_buf_size_;
        value_buf_ = other.value_buf_;
        value_buf_size_ = other.value_buf_size_;
        key_uses_memory_manager_ = other.key_uses_memory_manager_;
        value_uses_memory_manager_ = other.value_uses_memory_manager_;

        // Reset other
        other.key_buf_ = nullptr;
        other.key_buf_size_ = 0;
        other.value_buf_ = nullptr;
        other.value_buf_size_ = 0;
        other.key_uses_memory_manager_ = false;
        other.value_uses_memory_manager_ = false;
        other.n_positions_ = 0;
    }
    return *this;
}

float* KVCache::key(int64_t layer, int64_t kv_head, int64_t pos) {
    const int64_t offset = layer * layer_stride_ + kv_head * head_stride_ + pos * pos_stride_;
    return key_buf_ + offset;
}

const float* KVCache::key(int64_t layer, int64_t kv_head, int64_t pos) const {
    const int64_t offset = layer * layer_stride_ + kv_head * head_stride_ + pos * pos_stride_;
    return key_buf_ + offset;
}

float* KVCache::value(int64_t layer, int64_t kv_head, int64_t pos) {
    const int64_t offset = layer * layer_stride_ + kv_head * head_stride_ + pos * pos_stride_;
    return value_buf_ + offset;
}

const float* KVCache::value(int64_t layer, int64_t kv_head, int64_t pos) const {
    const int64_t offset = layer * layer_stride_ + kv_head * head_stride_ + pos * pos_stride_;
    return value_buf_ + offset;
}

float* KVCache::key_head(int64_t layer, int64_t kv_head) {
    const int64_t offset = layer * layer_stride_ + kv_head * head_stride_;
    return key_buf_ + offset;
}

const float* KVCache::key_head(int64_t layer, int64_t kv_head) const {
    const int64_t offset = layer * layer_stride_ + kv_head * head_stride_;
    return key_buf_ + offset;
}

float* KVCache::value_head(int64_t layer, int64_t kv_head) {
    const int64_t offset = layer * layer_stride_ + kv_head * head_stride_;
    return value_buf_ + offset;
}

const float* KVCache::value_head(int64_t layer, int64_t kv_head) const {
    const int64_t offset = layer * layer_stride_ + kv_head * head_stride_;
    return value_buf_ + offset;
}

void KVCache::advance(int64_t n) {
    n_positions_ += n;
}

void KVCache::reset() {
    n_positions_ = 0;
    if (key_buf_) {
        std::memset(key_buf_, 0, key_buf_size_);
    }
    if (value_buf_) {
        std::memset(value_buf_, 0, value_buf_size_);
    }
}

} // namespace kaguya
