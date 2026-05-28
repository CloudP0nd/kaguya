#include <gtest/gtest.h>
#include <vector>
#include <cmath>
#include <cstring>
#include <random>
#include <cstdint>
#include <numeric>

#include "kaguya/kernels/special_ops.h"
#include "kaguya/cpu_features.h"

using namespace kaguya;
using namespace kaguya::kernels;

// ============================================================================
// Helpers
// ============================================================================

static void fill_random(std::vector<float>& v, unsigned seed = 42) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-2.0f, 2.0f);
    for (auto& x : v) x = dist(rng);
}

static float rel_err(float a, float b) {
    if (std::abs(a) < 1e-6f && std::abs(b) < 1e-6f) return 0.0f;
    return std::abs(a - b) / std::max(std::abs(a), std::abs(b));
}

// ============================================================================
// Softmax tests
// ============================================================================

TEST(SoftmaxScalar, SimpleKnown) {
    // softmax([1, 2, 3]) ≈ [0.0900, 0.2447, 0.6652]
    std::vector<float> x = {1.0f, 2.0f, 3.0f};
    softmax_scalar(x.data(), 3);
    EXPECT_NEAR(x[0], 0.0900f, 1e-3f);
    EXPECT_NEAR(x[1], 0.2447f, 1e-3f);
    EXPECT_NEAR(x[2], 0.6652f, 1e-3f);
    // Sum should be 1
    float sum = x[0] + x[1] + x[2];
    EXPECT_NEAR(sum, 1.0f, 1e-6f);
}

TEST(SoftmaxScalar, Uniform) {
    // softmax of all-equal values should be uniform
    const int64_t n = 10;
    std::vector<float> x(n, 5.0f);
    softmax_scalar(x.data(), n);
    for (int64_t i = 0; i < n; ++i) {
        EXPECT_NEAR(x[i], 0.1f, 1e-6f);
    }
}

TEST(SoftmaxScalar, LargeValues) {
    // Large values should not overflow
    std::vector<float> x = {1000.0f, 1001.0f, 1002.0f};
    softmax_scalar(x.data(), 3);
    float sum = x[0] + x[1] + x[2];
    EXPECT_NEAR(sum, 1.0f, 1e-6f);
    // The largest should dominate
    EXPECT_GT(x[2], x[1]);
    EXPECT_GT(x[1], x[0]);
}

TEST(SoftmaxScalar, NegativeValues) {
    std::vector<float> x = {-1.0f, -2.0f, -3.0f};
    softmax_scalar(x.data(), 3);
    float sum = x[0] + x[1] + x[2];
    EXPECT_NEAR(sum, 1.0f, 1e-6f);
    // Most negative should be smallest probability
    EXPECT_GT(x[0], x[1]);
    EXPECT_GT(x[1], x[2]);
}

TEST(SoftmaxScalar, Temperature) {
    // Higher temperature flattens, lower temperature sharpens
    std::vector<float> x1 = {1.0f, 2.0f, 3.0f};
    std::vector<float> x2 = {1.0f, 2.0f, 3.0f};
    softmax_scalar(x1.data(), 3, 0.5f);  // sharper
    softmax_scalar(x2.data(), 3, 2.0f);  // flatter
    // Low temp: x1[2] should be larger (sharper)
    EXPECT_GT(x1[2] - x1[0], x2[2] - x2[0]);
}

TEST(SoftmaxScalar, SingleElement) {
    std::vector<float> x = {42.0f};
    softmax_scalar(x.data(), 1);
    EXPECT_FLOAT_EQ(x[0], 1.0f);
}

TEST(SoftmaxScalar, Empty) {
    std::vector<float> x;
    EXPECT_NO_THROW(softmax_scalar(x.data(), 0));
}

TEST(SoftmaxAVX512, MatchesScalar) {
    const auto& info = CpuFeatureDetector::get();
    if (!(info.flags.avx512f && info.flags.os_avx512)) {
        GTEST_SKIP() << "AVX-512 not available";
    }

    const int64_t n = 64;
    std::vector<float> x_scalar(n), x_avx512(n);
    fill_random(x_scalar);
    std::copy(x_scalar.begin(), x_scalar.end(), x_avx512.begin());

    softmax_scalar(x_scalar.data(), n);
    softmax_avx512(x_avx512.data(), n);

    for (int64_t i = 0; i < n; ++i) {
        EXPECT_NEAR(x_scalar[i], x_avx512[i], 1e-5f) << "at index " << i;
    }
}

TEST(SoftmaxAVX512, OddLength) {
    const auto& info = CpuFeatureDetector::get();
    if (!(info.flags.avx512f && info.flags.os_avx512)) {
        GTEST_SKIP() << "AVX-512 not available";
    }

    const int64_t n = 37;
    std::vector<float> x_scalar(n), x_avx512(n);
    fill_random(x_scalar, 77);
    std::copy(x_scalar.begin(), x_scalar.end(), x_avx512.begin());

    softmax_scalar(x_scalar.data(), n);
    softmax_avx512(x_avx512.data(), n);

    for (int64_t i = 0; i < n; ++i) {
        EXPECT_NEAR(x_scalar[i], x_avx512[i], 1e-5f) << "at index " << i;
    }
}

TEST(SoftmaxAVX512, SumsToOne) {
    const auto& info = CpuFeatureDetector::get();
    if (!(info.flags.avx512f && info.flags.os_avx512)) {
        GTEST_SKIP() << "AVX-512 not available";
    }

    const int64_t n = 128;
    std::vector<float> x(n);
    fill_random(x, 99);
    softmax_avx512(x.data(), n);

    float sum = 0.0f;
    for (int64_t i = 0; i < n; ++i) sum += x[i];
    EXPECT_NEAR(sum, 1.0f, 1e-5f);
}

TEST(SoftmaxDispatch, MatchesScalar) {
    const int64_t n = 48;
    std::vector<float> x_scalar(n), x_dispatch(n);
    fill_random(x_scalar, 123);
    std::copy(x_scalar.begin(), x_scalar.end(), x_dispatch.begin());

    softmax_scalar(x_scalar.data(), n);
    softmax_dispatch(x_dispatch.data(), n);

    for (int64_t i = 0; i < n; ++i) {
        EXPECT_NEAR(x_scalar[i], x_dispatch[i], 1e-5f) << "at index " << i;
    }
}

// ============================================================================
// RMSNorm tests
// ============================================================================

TEST(RMSNormScalar, SimpleKnown) {
    const int64_t n = 4;
    std::vector<float> x = {1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<float> out(n);
    std::vector<float> weight = {1.0f, 1.0f, 1.0f, 1.0f};

    rmsnorm_scalar(out.data(), x.data(), weight.data(), n);

    // sum_sq = 1+4+9+16 = 30, mean_sq = 7.5, rms = sqrt(7.5+1e-5) ≈ 2.73861
    // inv_rms ≈ 0.36515
    float inv_rms = 1.0f / std::sqrt(7.5f + 1e-5f);
    for (int64_t i = 0; i < n; ++i) {
        EXPECT_NEAR(out[i], x[i] * inv_rms, 1e-4f) << "at index " << i;
    }
}

TEST(RMSNormScalar, NoWeight) {
    const int64_t n = 4;
    std::vector<float> x = {1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<float> out(n);

    rmsnorm_scalar(out.data(), x.data(), nullptr, n);

    float inv_rms = 1.0f / std::sqrt(7.5f + 1e-5f);
    for (int64_t i = 0; i < n; ++i) {
        EXPECT_NEAR(out[i], x[i] * inv_rms, 1e-4f);
    }
}

TEST(RMSNormScalar, WithWeight) {
    const int64_t n = 3;
    std::vector<float> x = {3.0f, 4.0f, 0.0f};
    std::vector<float> out(n);
    std::vector<float> weight = {2.0f, 0.5f, 1.0f};

    rmsnorm_scalar(out.data(), x.data(), weight.data(), n);

    // sum_sq = 9+16+0 = 25, mean_sq = 25/3, inv_rms = 1/sqrt(25/3+eps)
    float inv_rms = 1.0f / std::sqrt(25.0f / 3.0f + 1e-5f);
    EXPECT_NEAR(out[0], 3.0f * inv_rms * 2.0f, 1e-4f);
    EXPECT_NEAR(out[1], 4.0f * inv_rms * 0.5f, 1e-4f);
    EXPECT_NEAR(out[2], 0.0f, 1e-6f);
}

TEST(RMSNormAVX512, MatchesScalar) {
    const auto& info = CpuFeatureDetector::get();
    if (!(info.flags.avx512f && info.flags.os_avx512)) {
        GTEST_SKIP() << "AVX-512 not available";
    }

    const int64_t n = 64;
    std::vector<float> x(n), weight(n), out_scalar(n), out_avx512(n);
    fill_random(x);
    fill_random(weight, 55);

    rmsnorm_scalar(out_scalar.data(), x.data(), weight.data(), n);
    rmsnorm_avx512(out_avx512.data(), x.data(), weight.data(), n);

    for (int64_t i = 0; i < n; ++i) {
        EXPECT_NEAR(out_scalar[i], out_avx512[i], 1e-4f) << "at index " << i;
    }
}

TEST(RMSNormAVX512, OddLength) {
    const auto& info = CpuFeatureDetector::get();
    if (!(info.flags.avx512f && info.flags.os_avx512)) {
        GTEST_SKIP() << "AVX-512 not available";
    }

    const int64_t n = 37;
    std::vector<float> x(n), weight(n), out_scalar(n), out_avx512(n);
    fill_random(x, 77);
    fill_random(weight, 88);

    rmsnorm_scalar(out_scalar.data(), x.data(), weight.data(), n);
    rmsnorm_avx512(out_avx512.data(), x.data(), weight.data(), n);

    for (int64_t i = 0; i < n; ++i) {
        EXPECT_NEAR(out_scalar[i], out_avx512[i], 1e-4f) << "at index " << i;
    }
}

TEST(RMSNormDispatch, MatchesScalar) {
    const int64_t n = 48;
    std::vector<float> x(n), weight(n), out_scalar(n), out_dispatch(n);
    fill_random(x, 200);
    fill_random(weight, 300);

    rmsnorm_scalar(out_scalar.data(), x.data(), weight.data(), n);
    rmsnorm_dispatch(out_dispatch.data(), x.data(), weight.data(), n);

    for (int64_t i = 0; i < n; ++i) {
        EXPECT_NEAR(out_scalar[i], out_dispatch[i], 1e-4f) << "at index " << i;
    }
}

// ============================================================================
// LayerNorm tests
// ============================================================================

TEST(LayerNormScalar, SimpleKnown) {
    const int64_t n = 4;
    std::vector<float> x = {1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<float> out(n);
    std::vector<float> weight = {1.0f, 1.0f, 1.0f, 1.0f};
    std::vector<float> bias = {0.0f, 0.0f, 0.0f, 0.0f};

    layernorm_scalar(out.data(), x.data(), weight.data(), bias.data(), n);

    // mean = 2.5, var = ((1-2.5)^2 + (2-2.5)^2 + (3-2.5)^2 + (4-2.5)^2) / 4 = 1.25
    // inv_std = 1/sqrt(1.25+eps)
    // Normalized: (-1.5, -0.5, 0.5, 1.5) * inv_std
    float mean = 2.5f;
    float var = 1.25f;
    float inv_std = 1.0f / std::sqrt(var + 1e-5f);
    EXPECT_NEAR(out[0], (-1.5f) * inv_std, 1e-4f);
    EXPECT_NEAR(out[1], (-0.5f) * inv_std, 1e-4f);
    EXPECT_NEAR(out[2], (0.5f) * inv_std, 1e-4f);
    EXPECT_NEAR(out[3], (1.5f) * inv_std, 1e-4f);
}

TEST(LayerNormScalar, WithBias) {
    const int64_t n = 3;
    std::vector<float> x = {1.0f, 1.0f, 1.0f};
    std::vector<float> out(n);
    std::vector<float> weight = {2.0f, 2.0f, 2.0f};
    std::vector<float> bias = {1.0f, 2.0f, 3.0f};

    layernorm_scalar(out.data(), x.data(), weight.data(), bias.data(), n);

    // mean = 1.0, var = 0.0, inv_std = 1/sqrt(eps) (very large)
    // But normalized values are all 0, so out = 0*weight + bias = bias
    EXPECT_NEAR(out[0], 1.0f, 1e-3f);
    EXPECT_NEAR(out[1], 2.0f, 1e-3f);
    EXPECT_NEAR(out[2], 3.0f, 1e-3f);
}

TEST(LayerNormAVX512, MatchesScalar) {
    const auto& info = CpuFeatureDetector::get();
    if (!(info.flags.avx512f && info.flags.os_avx512)) {
        GTEST_SKIP() << "AVX-512 not available";
    }

    const int64_t n = 64;
    std::vector<float> x(n), weight(n), bias(n), out_scalar(n), out_avx512(n);
    fill_random(x);
    fill_random(weight, 55);
    fill_random(bias, 66);

    layernorm_scalar(out_scalar.data(), x.data(), weight.data(), bias.data(), n);
    layernorm_avx512(out_avx512.data(), x.data(), weight.data(), bias.data(), n);

    for (int64_t i = 0; i < n; ++i) {
        EXPECT_NEAR(out_scalar[i], out_avx512[i], 1e-4f) << "at index " << i;
    }
}

TEST(LayerNormAVX512, OddLength) {
    const auto& info = CpuFeatureDetector::get();
    if (!(info.flags.avx512f && info.flags.os_avx512)) {
        GTEST_SKIP() << "AVX-512 not available";
    }

    const int64_t n = 37;
    std::vector<float> x(n), weight(n), bias(n), out_scalar(n), out_avx512(n);
    fill_random(x, 77);
    fill_random(weight, 88);
    fill_random(bias, 99);

    layernorm_scalar(out_scalar.data(), x.data(), weight.data(), bias.data(), n);
    layernorm_avx512(out_avx512.data(), x.data(), weight.data(), bias.data(), n);

    for (int64_t i = 0; i < n; ++i) {
        EXPECT_NEAR(out_scalar[i], out_avx512[i], 1e-4f) << "at index " << i;
    }
}

TEST(LayerNormAVX512, NoBias) {
    const auto& info = CpuFeatureDetector::get();
    if (!(info.flags.avx512f && info.flags.os_avx512)) {
        GTEST_SKIP() << "AVX-512 not available";
    }

    const int64_t n = 32;
    std::vector<float> x(n), weight(n), out_scalar(n), out_avx512(n);
    fill_random(x, 111);
    fill_random(weight, 222);

    layernorm_scalar(out_scalar.data(), x.data(), weight.data(), nullptr, n);
    layernorm_avx512(out_avx512.data(), x.data(), weight.data(), nullptr, n);

    for (int64_t i = 0; i < n; ++i) {
        EXPECT_NEAR(out_scalar[i], out_avx512[i], 1e-4f) << "at index " << i;
    }
}

TEST(LayerNormDispatch, MatchesScalar) {
    const int64_t n = 48;
    std::vector<float> x(n), weight(n), bias(n), out_scalar(n), out_dispatch(n);
    fill_random(x, 400);
    fill_random(weight, 500);
    fill_random(bias, 600);

    layernorm_scalar(out_scalar.data(), x.data(), weight.data(), bias.data(), n);
    layernorm_dispatch(out_dispatch.data(), x.data(), weight.data(), bias.data(), n);

    for (int64_t i = 0; i < n; ++i) {
        EXPECT_NEAR(out_scalar[i], out_dispatch[i], 1e-4f) << "at index " << i;
    }
}

// ============================================================================
// RoPE tests
// ============================================================================

TEST(RopeScalar, BasicRotation) {
    // Single head, head_dim=4 (2 pairs)
    const int64_t n_heads = 1, head_dim = 4, pos = 0;
    std::vector<float> x = {1.0f, 0.0f, 0.0f, 1.0f};

    rope_scalar(x.data(), n_heads, head_dim, pos);

    // At pos=0, theta=0 for all pairs, so cos=1, sin=0 → identity
    EXPECT_NEAR(x[0], 1.0f, 1e-6f);
    EXPECT_NEAR(x[1], 0.0f, 1e-6f);
    EXPECT_NEAR(x[2], 0.0f, 1e-6f);
    EXPECT_NEAR(x[3], 1.0f, 1e-6f);
}

TEST(RopeScalar, PositionOne) {
    // At pos=1, the first pair should be rotated by theta = 1 * freq_base^0 = 1 rad
    // But wait: theta = pos * freq_base^(-2i/head_dim)
    // For i=0: theta = 1 * 10000^0 = 1 (no wait, freq_base^(-2*0/4) = freq_base^0 = 1)
    // Actually i=0: theta = 1 * 10000^(0/4) = 1 * 1 = 1
    // Hmm, -2*0/4 = 0, so theta = pos * 10000^0 * 1.0 = 1.0
    const int64_t n_heads = 1, head_dim = 4, pos = 1;
    std::vector<float> x = {1.0f, 0.0f, 1.0f, 0.0f};

    rope_scalar(x.data(), n_heads, head_dim, pos);

    // For pair i=0: theta = 1.0
    // cos(1) ≈ 0.5403, sin(1) ≈ 0.8415
    // out[0] = 1*cos(1) - 0*sin(1) = cos(1) ≈ 0.5403
    // out[1] = 1*sin(1) + 0*cos(1) = sin(1) ≈ 0.8415
    EXPECT_NEAR(x[0], std::cos(1.0f), 1e-5f);
    EXPECT_NEAR(x[1], std::sin(1.0f), 1e-5f);
}

TEST(RopeScalar, MultipleHeads) {
    const int64_t n_heads = 2, head_dim = 4, pos = 2;
    std::vector<float> x = {1.0f, 0.0f, 0.0f, 1.0f, 1.0f, 0.0f, 0.0f, 1.0f};
    std::vector<float> x_copy = x;

    rope_scalar(x.data(), n_heads, head_dim, pos);

    // Both heads should be rotated the same way
    for (int64_t h = 0; h < n_heads; ++h) {
        const int64_t half_dim = head_dim / 2;
        for (int64_t i = 0; i < half_dim; ++i) {
            const float theta = static_cast<float>(pos) *
                std::pow(10000.0f, -2.0f * static_cast<float>(i) / static_cast<float>(head_dim));
            const float cos_t = std::cos(theta);
            const float sin_t = std::sin(theta);
            const float x0 = x_copy[h * head_dim + 2 * i];
            const float x1 = x_copy[h * head_dim + 2 * i + 1];
            const float expected_0 = x0 * cos_t - x1 * sin_t;
            const float expected_1 = x0 * sin_t + x1 * cos_t;
            EXPECT_NEAR(x[h * head_dim + 2 * i], expected_0, 1e-5f);
            EXPECT_NEAR(x[h * head_dim + 2 * i + 1], expected_1, 1e-5f);
        }
    }
}

TEST(RopeScalar, PreservesMagnitude) {
    const int64_t n_heads = 1, head_dim = 8, pos = 10;
    std::vector<float> x = {3.0f, 4.0f, 1.0f, 2.0f, 0.5f, 0.5f, 1.0f, 0.0f};
    // Pre-compute magnitudes of each pair
    std::vector<float> mags;
    for (int64_t i = 0; i < head_dim / 2; ++i) {
        mags.push_back(std::sqrt(x[2 * i] * x[2 * i] + x[2 * i + 1] * x[2 * i + 1]));
    }

    rope_scalar(x.data(), n_heads, head_dim, pos);

    // Each pair should preserve its magnitude (rotation is magnitude-preserving)
    for (int64_t i = 0; i < head_dim / 2; ++i) {
        float mag = std::sqrt(x[2 * i] * x[2 * i] + x[2 * i + 1] * x[2 * i + 1]);
        EXPECT_NEAR(mag, mags[i], 1e-4f) << "pair " << i;
    }
}

TEST(RopeAVX512, MatchesScalar) {
    const auto& info = CpuFeatureDetector::get();
    if (!(info.flags.avx512f && info.flags.os_avx512)) {
        GTEST_SKIP() << "AVX-512 not available";
    }

    const int64_t n_heads = 4, head_dim = 16, pos = 5;
    std::vector<float> x_scalar(n_heads * head_dim), x_avx512(n_heads * head_dim);
    fill_random(x_scalar);
    std::copy(x_scalar.begin(), x_scalar.end(), x_avx512.begin());

    rope_scalar(x_scalar.data(), n_heads, head_dim, pos);
    rope_avx512(x_avx512.data(), n_heads, head_dim, pos);

    for (int64_t i = 0; i < n_heads * head_dim; ++i) {
        EXPECT_NEAR(x_scalar[i], x_avx512[i], 1e-4f) << "at index " << i;
    }
}

TEST(RopeAVX512, OddHalfDim) {
    const auto& info = CpuFeatureDetector::get();
    if (!(info.flags.avx512f && info.flags.os_avx512)) {
        GTEST_SKIP() << "AVX-512 not available";
    }

    // head_dim = 6 (3 pairs, odd number of pairs — tests scalar tail)
    const int64_t n_heads = 2, head_dim = 6, pos = 3;
    std::vector<float> x_scalar(n_heads * head_dim), x_avx512(n_heads * head_dim);
    fill_random(x_scalar, 88);
    std::copy(x_scalar.begin(), x_scalar.end(), x_avx512.begin());

    rope_scalar(x_scalar.data(), n_heads, head_dim, pos);
    rope_avx512(x_avx512.data(), n_heads, head_dim, pos);

    for (int64_t i = 0; i < n_heads * head_dim; ++i) {
        EXPECT_NEAR(x_scalar[i], x_avx512[i], 1e-4f) << "at index " << i;
    }
}

TEST(RopeDispatch, MatchesScalar) {
    const int64_t n_heads = 2, head_dim = 8, pos = 7;
    std::vector<float> x_scalar(n_heads * head_dim), x_dispatch(n_heads * head_dim);
    fill_random(x_scalar, 123);
    std::copy(x_scalar.begin(), x_scalar.end(), x_dispatch.begin());

    rope_scalar(x_scalar.data(), n_heads, head_dim, pos);
    rope_dispatch(x_dispatch.data(), n_heads, head_dim, pos);

    for (int64_t i = 0; i < n_heads * head_dim; ++i) {
        EXPECT_NEAR(x_scalar[i], x_dispatch[i], 1e-4f) << "at index " << i;
    }
}

// ============================================================================
// SiLU tests
// ============================================================================

TEST(SiLUScalar, KnownValues) {
    // SiLU(0) = 0 / (1 + exp(0)) = 0 / 2 = 0
    // SiLU(1) = 1 / (1 + exp(-1)) ≈ 0.7311
    // SiLU(-1) = -1 / (1 + exp(1)) ≈ -0.2689
    std::vector<float> x = {0.0f, 1.0f, -1.0f};
    silu_scalar(x.data(), 3);

    EXPECT_NEAR(x[0], 0.0f, 1e-6f);
    EXPECT_NEAR(x[1], 1.0f / (1.0f + std::exp(-1.0f)), 1e-5f);
    EXPECT_NEAR(x[2], -1.0f / (1.0f + std::exp(1.0f)), 1e-5f);
}

TEST(SiLUScalar, PositiveInput) {
    // For large positive x, SiLU(x) ≈ x
    std::vector<float> x = {10.0f};
    silu_scalar(x.data(), 1);
    EXPECT_NEAR(x[0], 10.0f, 0.1f);
}

TEST(SiLUScalar, NegativeInput) {
    // For large negative x, SiLU(x) ≈ 0
    std::vector<float> x = {-10.0f};
    silu_scalar(x.data(), 1);
    EXPECT_NEAR(x[0], 0.0f, 0.01f);
}

TEST(SiLUAVX512, MatchesScalar) {
    const auto& info = CpuFeatureDetector::get();
    if (!(info.flags.avx512f && info.flags.os_avx512)) {
        GTEST_SKIP() << "AVX-512 not available";
    }

    const int64_t n = 64;
    std::vector<float> x_scalar(n), x_avx512(n);
    fill_random(x_scalar);
    std::copy(x_scalar.begin(), x_scalar.end(), x_avx512.begin());

    silu_scalar(x_scalar.data(), n);
    silu_avx512(x_avx512.data(), n);

    for (int64_t i = 0; i < n; ++i) {
        EXPECT_NEAR(x_scalar[i], x_avx512[i], 1e-4f) << "at index " << i;
    }
}

TEST(SiLUAVX512, OddLength) {
    const auto& info = CpuFeatureDetector::get();
    if (!(info.flags.avx512f && info.flags.os_avx512)) {
        GTEST_SKIP() << "AVX-512 not available";
    }

    const int64_t n = 37;
    std::vector<float> x_scalar(n), x_avx512(n);
    fill_random(x_scalar, 77);
    std::copy(x_scalar.begin(), x_scalar.end(), x_avx512.begin());

    silu_scalar(x_scalar.data(), n);
    silu_avx512(x_avx512.data(), n);

    for (int64_t i = 0; i < n; ++i) {
        EXPECT_NEAR(x_scalar[i], x_avx512[i], 1e-4f) << "at index " << i;
    }
}

TEST(SiLUDispatch, MatchesScalar) {
    const int64_t n = 48;
    std::vector<float> x_scalar(n), x_dispatch(n);
    fill_random(x_scalar, 200);
    std::copy(x_scalar.begin(), x_scalar.end(), x_dispatch.begin());

    silu_scalar(x_scalar.data(), n);
    silu_dispatch(x_dispatch.data(), n);

    for (int64_t i = 0; i < n; ++i) {
        EXPECT_NEAR(x_scalar[i], x_dispatch[i], 1e-4f) << "at index " << i;
    }
}

// ============================================================================
// GELU tests
// ============================================================================

TEST(GELUScalar, KnownValues) {
    // GELU(0) = 0
    // GELU(1) ≈ 0.8412
    // GELU(-1) ≈ -0.1588
    std::vector<float> x = {0.0f, 1.0f, -1.0f};
    gelu_scalar(x.data(), 3);

    EXPECT_NEAR(x[0], 0.0f, 1e-6f);

    // Reference: 0.5 * 1 * (1 + tanh(sqrt(2/pi) * (1 + 0.044715)))
    float ref1 = 0.5f * 1.0f * (1.0f + std::tanh(0.7978845608028654f * (1.0f + 0.044715f)));
    EXPECT_NEAR(x[1], ref1, 1e-5f);

    float ref_neg1 = 0.5f * (-1.0f) * (1.0f + std::tanh(0.7978845608028654f * (-1.0f + 0.044715f * (-1.0f))));
    EXPECT_NEAR(x[2], ref_neg1, 1e-5f);
}

TEST(GELUScalar, PositiveInput) {
    // For large positive x, GELU(x) ≈ x
    std::vector<float> x = {10.0f};
    gelu_scalar(x.data(), 1);
    EXPECT_NEAR(x[0], 10.0f, 0.01f);
}

TEST(GELUScalar, NegativeInput) {
    // For large negative x, GELU(x) ≈ 0
    std::vector<float> x = {-10.0f};
    gelu_scalar(x.data(), 1);
    EXPECT_NEAR(x[0], 0.0f, 0.01f);
}

TEST(GELUAVX512, MatchesScalar) {
    const auto& info = CpuFeatureDetector::get();
    if (!(info.flags.avx512f && info.flags.os_avx512)) {
        GTEST_SKIP() << "AVX-512 not available";
    }

    const int64_t n = 64;
    std::vector<float> x_scalar(n), x_avx512(n);
    fill_random(x_scalar);
    std::copy(x_scalar.begin(), x_scalar.end(), x_avx512.begin());

    gelu_scalar(x_scalar.data(), n);
    gelu_avx512(x_avx512.data(), n);

    for (int64_t i = 0; i < n; ++i) {
        EXPECT_NEAR(x_scalar[i], x_avx512[i], 1e-4f) << "at index " << i;
    }
}

TEST(GELUAVX512, OddLength) {
    const auto& info = CpuFeatureDetector::get();
    if (!(info.flags.avx512f && info.flags.os_avx512)) {
        GTEST_SKIP() << "AVX-512 not available";
    }

    const int64_t n = 37;
    std::vector<float> x_scalar(n), x_avx512(n);
    fill_random(x_scalar, 88);
    std::copy(x_scalar.begin(), x_scalar.end(), x_avx512.begin());

    gelu_scalar(x_scalar.data(), n);
    gelu_avx512(x_avx512.data(), n);

    for (int64_t i = 0; i < n; ++i) {
        EXPECT_NEAR(x_scalar[i], x_avx512[i], 1e-4f) << "at index " << i;
    }
}

TEST(GELUDispatch, MatchesScalar) {
    const int64_t n = 48;
    std::vector<float> x_scalar(n), x_dispatch(n);
    fill_random(x_scalar, 300);
    std::copy(x_scalar.begin(), x_scalar.end(), x_dispatch.begin());

    gelu_scalar(x_scalar.data(), n);
    gelu_dispatch(x_dispatch.data(), n);

    for (int64_t i = 0; i < n; ++i) {
        EXPECT_NEAR(x_scalar[i], x_dispatch[i], 1e-4f) << "at index " << i;
    }
}

// ============================================================================
// Cross-kernel integration tests
// ============================================================================

TEST(SpecialOpsIntegration, SoftmaxAfterNorm) {
    // Simulate attention: normalize, then softmax
    const int64_t n = 32;
    std::vector<float> x(n), x_copy(n), weight(n, 1.0f), out(n);
    fill_random(x, 42);
    std::copy(x.begin(), x.end(), x_copy.begin());

    // Apply RMSNorm then softmax
    rmsnorm_scalar(out.data(), x.data(), weight.data(), n);
    softmax_scalar(out.data(), n);

    // Result should sum to 1
    float sum = 0.0f;
    for (int64_t i = 0; i < n; ++i) sum += out[i];
    EXPECT_NEAR(sum, 1.0f, 1e-5f);
}

TEST(SpecialOpsIntegration, SiLUGeluConsistency) {
    // At x=0, both SiLU and GELU should output 0
    std::vector<float> x_silu = {0.0f};
    std::vector<float> x_gelu = {0.0f};
    silu_scalar(x_silu.data(), 1);
    gelu_scalar(x_gelu.data(), 1);
    EXPECT_NEAR(x_silu[0], 0.0f, 1e-6f);
    EXPECT_NEAR(x_gelu[0], 0.0f, 1e-6f);
}
