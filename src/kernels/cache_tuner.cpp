#include "kaguya/kernels/cache_tuner.h"
#include "kaguya/cpu_features.h"

#include <cmath>
#include <algorithm>

namespace kaguya::kernels {

CacheTuner::CacheTuner() {
    const auto& info = kaguya::CpuFeatureDetector::get();

    // Find L2 cache size from detected cache info
    for (const auto& cache : info.caches) {
        if (cache.level == 2) {
            l2_size_bytes_ = static_cast<int64_t>(cache.size_kb) * 1024;
            break;
        }
    }

    // Compute optimal tile size: sqrt(L2_size / (3 * sizeof(float)))
    // This ensures 3 tile panels (A, B, C) fit in L2 cache.
    if (l2_size_bytes_ > 0) {
        constexpr int64_t num_panels = 3;
        constexpr int64_t element_size = static_cast<int64_t>(sizeof(float));
        const double tile_f = std::sqrt(
            static_cast<double>(l2_size_bytes_) / static_cast<double>(num_panels * element_size)
        );
        tile_size_ = static_cast<int64_t>(tile_f);
    }

    // Clamp to practical range [16, 512]
    tile_size_ = std::clamp(tile_size_, int64_t{16}, int64_t{512});
}

const CacheTuner& CacheTuner::get() {
    static const CacheTuner instance;
    return instance;
}

} // namespace kaguya::kernels
