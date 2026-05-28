#pragma once
/// @file cache_tuner.h
/// @brief Phase 6 Cache-aware GEMM tile tuning.
///
/// Reads L1/L2 cache sizes from CpuFeatureDetector and computes
/// optimal GEMM tile sizes based on cache capacity.
/// Formula: tile_size = sqrt(L2_size / (3 * sizeof(float)))
/// Ensures 3 tile panels (A, B, C) fit in L2 cache.

#include <cstdint>

namespace kaguya::kernels {

/// Cache-aware GEMM tile size tuner
class CacheTuner {
public:
    /// Get the singleton instance (lazy initialization on first call)
    static const CacheTuner& get();

    /// Get the optimal GEMM tile size for the current CPU's cache hierarchy.
    /// Computed as: tile_size = sqrt(L2_size / (3 * sizeof(float)))
    /// Clamped to [16, 512] for practical tile sizes.
    int64_t get_tile_size() const { return tile_size_; }

    /// Get the detected L2 cache size in bytes (0 if unknown)
    int64_t l2_size_bytes() const { return l2_size_bytes_; }

private:
    CacheTuner();

    int64_t l2_size_bytes_ = 0;
    int64_t tile_size_ = 64; // default fallback
};

} // namespace kaguya::kernels
