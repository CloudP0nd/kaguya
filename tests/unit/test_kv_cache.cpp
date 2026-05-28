#include <gtest/gtest.h>
#include "kaguya/kv_cache.h"

#include <cstring>
#include <vector>
#include <cmath>

using namespace kaguya;

// ============================================================================
// Construction and dimensions
// ============================================================================

TEST(KvCache, ConstructionSetsDimensions) {
    KVCache cache(4, 8, 64, 128);
    EXPECT_EQ(cache.n_layers(), 4);
    EXPECT_EQ(cache.n_kv_heads(), 8);
    EXPECT_EQ(cache.head_dim(), 64);
    EXPECT_EQ(cache.max_seq_len(), 128);
    EXPECT_EQ(cache.kv_dim(), 8 * 64);
}

TEST(KvCache, InitialPositionIsZero) {
    KVCache cache(2, 4, 32, 64);
    EXPECT_EQ(cache.n_positions(), 0);
}

TEST(KvCache, AdvanceIncrementsPosition) {
    KVCache cache(2, 4, 32, 64);
    cache.advance(1);
    EXPECT_EQ(cache.n_positions(), 1);
    cache.advance(5);
    EXPECT_EQ(cache.n_positions(), 6);
}

TEST(KvCache, ResetClearsPosition) {
    KVCache cache(2, 4, 32, 64);
    cache.advance(10);
    EXPECT_EQ(cache.n_positions(), 10);
    cache.reset();
    EXPECT_EQ(cache.n_positions(), 0);
}

// ============================================================================
// Key/Value storage and retrieval
// ============================================================================

TEST(KvCache, StoreAndRetrieveKey) {
    KVCache cache(1, 2, 4, 8);

    // Store a key vector at layer=0, head=0, pos=0
    float key_data[] = {1.0f, 2.0f, 3.0f, 4.0f};
    float* dst = cache.key(0, 0, 0);
    std::memcpy(dst, key_data, sizeof(key_data));

    // Retrieve and verify
    const float* src = cache.key(0, 0, 0);
    for (int i = 0; i < 4; ++i) {
        EXPECT_FLOAT_EQ(src[i], key_data[i]);
    }
}

TEST(KvCache, StoreAndRetrieveValue) {
    KVCache cache(1, 2, 4, 8);

    float val_data[] = {5.0f, 6.0f, 7.0f, 8.0f};
    float* dst = cache.value(0, 1, 3);
    std::memcpy(dst, val_data, sizeof(val_data));

    const float* src = cache.value(0, 1, 3);
    for (int i = 0; i < 4; ++i) {
        EXPECT_FLOAT_EQ(src[i], val_data[i]);
    }
}

TEST(KvCache, MultiplePositionsDoNotOverlap) {
    KVCache cache(1, 1, 4, 8);

    // Store different data at two positions for the same head
    float data0[] = {1.0f, 2.0f, 3.0f, 4.0f};
    float data1[] = {10.0f, 20.0f, 30.0f, 40.0f};

    std::memcpy(cache.key(0, 0, 0), data0, sizeof(data0));
    std::memcpy(cache.key(0, 0, 1), data1, sizeof(data1));

    // Verify no overlap
    const float* k0 = cache.key(0, 0, 0);
    const float* k1 = cache.key(0, 0, 1);
    for (int i = 0; i < 4; ++i) {
        EXPECT_FLOAT_EQ(k0[i], data0[i]);
        EXPECT_FLOAT_EQ(k1[i], data1[i]);
    }
}

TEST(KvCache, MultipleHeadsDoNotOverlap) {
    KVCache cache(1, 2, 4, 8);

    float data_h0[] = {1.0f, 2.0f, 3.0f, 4.0f};
    float data_h1[] = {10.0f, 20.0f, 30.0f, 40.0f};

    std::memcpy(cache.key(0, 0, 0), data_h0, sizeof(data_h0));
    std::memcpy(cache.key(0, 1, 0), data_h1, sizeof(data_h1));

    const float* k_h0 = cache.key(0, 0, 0);
    const float* k_h1 = cache.key(0, 1, 0);
    for (int i = 0; i < 4; ++i) {
        EXPECT_FLOAT_EQ(k_h0[i], data_h0[i]);
        EXPECT_FLOAT_EQ(k_h1[i], data_h1[i]);
    }
}

TEST(KvCache, MultipleLayersDoNotOverlap) {
    KVCache cache(2, 1, 4, 8);

    float data_l0[] = {1.0f, 2.0f, 3.0f, 4.0f};
    float data_l1[] = {10.0f, 20.0f, 30.0f, 40.0f};

    std::memcpy(cache.key(0, 0, 0), data_l0, sizeof(data_l0));
    std::memcpy(cache.key(1, 0, 0), data_l1, sizeof(data_l1));

    const float* k_l0 = cache.key(0, 0, 0);
    const float* k_l1 = cache.key(1, 0, 0);
    for (int i = 0; i < 4; ++i) {
        EXPECT_FLOAT_EQ(k_l0[i], data_l0[i]);
        EXPECT_FLOAT_EQ(k_l1[i], data_l1[i]);
    }
}

// ============================================================================
// key_head / value_head access
// ============================================================================

TEST(KvCache, KeyHeadReturnsContiguousBlock) {
    KVCache cache(1, 1, 4, 8);

    // Store key vectors at positions 0-2
    for (int pos = 0; pos < 3; ++pos) {
        float* dst = cache.key(0, 0, pos);
        for (int d = 0; d < 4; ++d) {
            dst[d] = static_cast<float>(pos * 4 + d);
        }
    }

    // key_head should return pointer to [max_seq_len, head_dim]
    const float* head_data = cache.key_head(0, 0);
    for (int pos = 0; pos < 3; ++pos) {
        for (int d = 0; d < 4; ++d) {
            EXPECT_FLOAT_EQ(head_data[pos * 4 + d], static_cast<float>(pos * 4 + d));
        }
    }
}

TEST(KvCache, ValueHeadReturnsContiguousBlock) {
    KVCache cache(1, 1, 4, 8);

    for (int pos = 0; pos < 3; ++pos) {
        float* dst = cache.value(0, 0, pos);
        for (int d = 0; d < 4; ++d) {
            dst[d] = static_cast<float>(pos * 4 + d + 100);
        }
    }

    const float* head_data = cache.value_head(0, 0);
    for (int pos = 0; pos < 3; ++pos) {
        for (int d = 0; d < 4; ++d) {
            EXPECT_FLOAT_EQ(head_data[pos * 4 + d], static_cast<float>(pos * 4 + d + 100));
        }
    }
}

// ============================================================================
// Reset clears data
// ============================================================================

TEST(KvCache, ResetZerosData) {
    KVCache cache(1, 1, 4, 8);

    // Store some data
    float data[] = {1.0f, 2.0f, 3.0f, 4.0f};
    std::memcpy(cache.key(0, 0, 0), data, sizeof(data));

    // Reset
    cache.reset();

    // Verify data is zeroed
    const float* k = cache.key(0, 0, 0);
    for (int i = 0; i < 4; ++i) {
        EXPECT_FLOAT_EQ(k[i], 0.0f);
    }
}

// ============================================================================
// Move semantics
// ============================================================================

TEST(KvCache, MoveConstruction) {
    KVCache cache1(2, 4, 32, 64);
    float data[] = {42.0f, 43.0f};
    std::memcpy(cache1.key(0, 0, 0), data, sizeof(data));

    KVCache cache2(std::move(cache1));
    EXPECT_EQ(cache2.n_layers(), 2);
    EXPECT_EQ(cache2.n_kv_heads(), 4);
    EXPECT_EQ(cache2.head_dim(), 32);
    EXPECT_EQ(cache2.max_seq_len(), 64);

    const float* k = cache2.key(0, 0, 0);
    EXPECT_FLOAT_EQ(k[0], 42.0f);
    EXPECT_FLOAT_EQ(k[1], 43.0f);
}
