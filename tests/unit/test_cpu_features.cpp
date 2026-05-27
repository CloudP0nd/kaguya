#include <gtest/gtest.h>
#include "kaguya/cpu_features.h"

TEST(CpuFeatures, DetectDoesNotCrash) {
    auto info = kaguya::CpuFeatureDetector::detect();
    EXPECT_GT(info.num_logical_cpus, 0);
}

TEST(CpuFeatures, VendorIsSet) {
    const auto& info = kaguya::CpuFeatureDetector::get();
    EXPECT_FALSE(info.vendor.empty());
}

TEST(CpuFeatures, CachesDetected) {
    const auto& info = kaguya::CpuFeatureDetector::get();
    EXPECT_FALSE(info.caches.empty());
}

TEST(CpuFeatures, BestKernelTargetIsValid) {
    auto target = kaguya::CpuFeatureDetector::best_kernel_target();
    EXPECT_TRUE(target == "amx" || target == "avx512" || target == "avx2" || target == "scalar");
}

TEST(CpuFeatures, SummaryNotEmpty) {
    auto s = kaguya::CpuFeatureDetector::summary();
    EXPECT_FALSE(s.empty());
}
