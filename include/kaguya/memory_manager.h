#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>
#include <memory>

namespace kaguya {

/// Memory allocation flags
enum class MemFlags : uint32_t {
    None        = 0,
    HugePages   = 1 << 0,  ///< Use 2MB huge pages
    HugePages1G = 1 << 1,  ///< Use 1GB huge pages
    NUMAAware   = 1 << 2,  ///< Bind to specific NUMA node
    Aligned64   = 1 << 3,  ///< 64-byte alignment (default for SIMD)
};

inline MemFlags operator|(MemFlags a, MemFlags b) {
    return static_cast<MemFlags>(static_cast<uint32_t>(a) | static_cast<uint32_t>(b));
}

inline bool operator&(MemFlags a, MemFlags b) {
    return (static_cast<uint32_t>(a) & static_cast<uint32_t>(b)) != 0;
}

/// RAII memory allocation with Huge Pages and NUMA support
class MemoryManager {
public:
    /// Initialize memory manager (detect Huge Pages availability, NUMA topology)
    static void init();

    /// Allocate aligned memory with optional Huge Pages and NUMA binding
    /// @param size Number of bytes to allocate
    /// @param flags Allocation flags
    /// @param numa_node NUMA node for binding (-1 = any)
    /// @return Pointer to allocated memory, or nullptr on failure
    static void* allocate(size_t size, MemFlags flags = MemFlags::Aligned64, int numa_node = -1);

    /// Free memory allocated by this manager
    static void deallocate(void* ptr, size_t size);

    /// Allocate and zero-fill
    static void* allocate_zero(size_t size, MemFlags flags = MemFlags::Aligned64, int numa_node = -1);

    /// Check if Huge Pages are available
    static bool huge_pages_available();

    /// Get total Huge Pages allocated
    static size_t huge_pages_allocated() { return huge_pages_allocated_; }

    /// Get total memory allocated
    static size_t total_allocated() { return total_allocated_; }

private:
    static bool huge_pages_available_;
    static size_t huge_pages_allocated_;
    static size_t total_allocated_;
};

/// RAII wrapper for MemoryManager allocations
class AlignedBuffer {
public:
    AlignedBuffer() : ptr_(nullptr), size_(0) {}

    explicit AlignedBuffer(size_t size, MemFlags flags = MemFlags::Aligned64, int numa_node = -1)
        : ptr_(MemoryManager::allocate(size, flags, numa_node)), size_(size) {}

    ~AlignedBuffer() {
        if (ptr_) MemoryManager::deallocate(ptr_, size_);
    }

    // Move-only
    AlignedBuffer(const AlignedBuffer&) = delete;
    AlignedBuffer& operator=(const AlignedBuffer&) = delete;
    AlignedBuffer(AlignedBuffer&& other) noexcept
        : ptr_(other.ptr_), size_(other.size_) {
        other.ptr_ = nullptr;
        other.size_ = 0;
    }
    AlignedBuffer& operator=(AlignedBuffer&& other) noexcept {
        if (this != &other) {
            if (ptr_) MemoryManager::deallocate(ptr_, size_);
            ptr_ = other.ptr_;
            size_ = other.size_;
            other.ptr_ = nullptr;
            other.size_ = 0;
        }
        return *this;
    }

    void* data() { return ptr_; }
    const void* data() const { return ptr_; }
    size_t size() const { return size_; }
    bool empty() const { return ptr_ == nullptr; }

    template<typename T>
    T* as() { return reinterpret_cast<T*>(ptr_); }

    template<typename T>
    const T* as() const { return reinterpret_cast<const T*>(ptr_); }

private:
    void* ptr_;
    size_t size_;
};

} // namespace kaguya
