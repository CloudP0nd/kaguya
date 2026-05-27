#include "kaguya/memory/memory_manager.h"

#include <cstdlib>
#include <cstring>
#include <cerrno>
#include <iostream>

#ifdef __linux__
#include <sys/mman.h>
#ifdef KAGUYA_HAS_NUMA
#include <numa.h>
#include <numaif.h>
#endif
#endif

namespace kaguya {

bool MemoryManager::huge_pages_available_ = false;
size_t MemoryManager::huge_pages_allocated_ = 0;
size_t MemoryManager::total_allocated_ = 0;

void MemoryManager::init() {
#ifdef __linux__
    // Check for 2MB huge pages
    void* test = mmap(nullptr, 2 * 1024 * 1024, PROT_READ | PROT_WRITE,
                       MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB, -1, 0);
    if (test != MAP_FAILED) {
        huge_pages_available_ = true;
        munmap(test, 2 * 1024 * 1024);
    } else {
        // Fallback: will use MADV_HUGEPAGE
        huge_pages_available_ = true;
    }
#else
    huge_pages_available_ = false;
#endif
}

bool MemoryManager::huge_pages_available() {
    return huge_pages_available_;
}

void* MemoryManager::allocate(size_t size, MemFlags flags, int numa_node) {
    if (size == 0) return nullptr;

    void* ptr = nullptr;

#ifdef __linux__
    // Try Huge Pages first if requested
    if (flags & MemFlags::HugePages) {
        int page_size = (flags & MemFlags::HugePages1G) ? (1 << 30) : (2 << 20);
        size_t aligned_size = ((size + page_size - 1) / page_size) * page_size;

        int mmap_flags = MAP_PRIVATE | MAP_ANONYMOUS;
        if (page_size == (2 << 20)) mmap_flags |= MAP_HUGETLB;
        else if (page_size == (1 << 30)) mmap_flags |= MAP_HUGETLB | (30 << 26);

        ptr = mmap(nullptr, aligned_size, PROT_READ | PROT_WRITE, mmap_flags, -1, 0);

        if (ptr == MAP_FAILED) {
            // Fallback: regular mmap + MADV_HUGEPAGE
            ptr = mmap(nullptr, aligned_size, PROT_READ | PROT_WRITE,
                        MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
            if (ptr != MAP_FAILED) {
                madvise(ptr, aligned_size, MADV_HUGEPAGE);
            }
        }

        if (ptr != MAP_FAILED) {
            huge_pages_allocated_ += aligned_size;
            total_allocated_ += aligned_size;
        } else {
            ptr = nullptr;
        }
    }

    // NUMA-aware allocation
    if (!ptr && (flags & MemFlags::NUMAAware) && numa_node >= 0) {
        size_t aligned_size = ((size + 4095) / 4096) * 4096;
        ptr = mmap(nullptr, aligned_size, PROT_READ | PROT_WRITE,
                    MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (ptr != MAP_FAILED) {
#ifdef KAGUYA_HAS_NUMA
            // Bind to NUMA node via mbind
            unsigned long nodemask = 1UL << numa_node;
            long ret = mbind(ptr, aligned_size, MPOL_BIND, &nodemask,
                             sizeof(nodemask) * 8, 0);
            if (ret != 0) {
                // mbind failed, but memory is still usable
            }
#else
            (void)numa_node;
#endif
            total_allocated_ += aligned_size;
        } else {
            ptr = nullptr;
        }
    }
#endif

    // Fallback: aligned_alloc
    if (!ptr) {
        size_t alignment = (flags & MemFlags::Aligned64) ? 64 : sizeof(void*);
        size_t aligned_size = ((size + alignment - 1) / alignment) * alignment;

        ptr = std::aligned_alloc(alignment, aligned_size);

        if (ptr) {
            total_allocated_ += aligned_size;
        }
    }

    return ptr;
}

void MemoryManager::deallocate(void* ptr, size_t size) {
    if (!ptr) return;

#ifdef __linux__
    // Heuristic: sizes >= 2MB likely came from mmap
    if (size >= 2 * 1024 * 1024) {
        size_t page_size = 4096;
        size_t aligned_size = ((size + page_size - 1) / page_size) * page_size;
        munmap(ptr, aligned_size);
    } else {
        free(ptr);
    }
#else
    free(ptr);
#endif
}

void* MemoryManager::allocate_zero(size_t size, MemFlags flags, int numa_node) {
    void* ptr = allocate(size, flags, numa_node);
    if (ptr) {
        memset(ptr, 0, size);
    }
    return ptr;
}

} // namespace kaguya
