// Kaguya — BPE tokenizer unit tests

#include <gtest/gtest.h>
#include "kaguya/tokenizer.h"

using namespace kaguya;

// ============================================================================
// Helper: build a minimal BPE tokenizer for testing
// ============================================================================

static BpeTokenizer make_test_tokenizer() {
    // Create a small vocabulary similar to LLaMA-style BPE
    // Token types: 0=NORMAL, 1=UNKNOWN, 2=CONTROL, 3=BYTE
    std::vector<std::string> tokens = {
        "<s>",            // 0 - BOS (CONTROL)
        "</s>",           // 1 - EOS (CONTROL)
        "<unk>",          // 2 - UNK (UNKNOWN)
        "<0x00>",         // 3 - byte token (BYTE)
        "<0x01>",         // 4
        "<0x20>",         // 5 - space byte
        "<0x48>",         // 6 - 'H' byte
        "<0x65>",         // 7 - 'e' byte
        "<0x6C>",         // 8 - 'l' byte
        "<0x6F>",         // 9 - 'o' byte
        "<0x77>",         // 10 - 'w' byte
        "<0x72>",         // 11 - 'r' byte
        "<0x64>",         // 12 - 'd' byte
        "\xE2\x96\x81",  // 13 - ▁ (SentencePiece space)
        "\xE2\x96\x81H", // 14 - ▁H
        "\xE2\x96\x81w", // 15 - ▁w
        "He",             // 16
        "ll",             // 17
        "or",             // 18
        "Hello",          // 19
        "world",          // 20
    };

    std::vector<float> scores(tokens.size(), 0.0f);
    for (int i = 0; i < static_cast<int>(tokens.size()); ++i) {
        scores[i] = -static_cast<float>(i);
    }

    std::vector<int32_t> token_types(tokens.size(), 0); // NORMAL
    token_types[0] = 2;  // CONTROL for BOS
    token_types[1] = 2;  // CONTROL for EOS
    token_types[2] = 1;  // UNKNOWN for UNK
    // Byte tokens
    for (int i = 3; i <= 12; ++i) {
        token_types[i] = 3; // BYTE type
    }

    // BPE merges
    std::vector<std::string> merges = {
        "H e",
        "l l",
        "o r",
    };

    BpeTokenizer tok;
    bool ok = tok.build(tokens, scores, token_types, merges, 0, 1, true);
    EXPECT_TRUE(ok);
    return tok;
}

// ============================================================================
// Build tests
// ============================================================================

TEST(BpeTokenizer, BuildFromMetadata) {
    auto tok = make_test_tokenizer();
    EXPECT_TRUE(tok.is_valid());
    EXPECT_EQ(tok.vocab_size(), 21);
    EXPECT_EQ(tok.bos_token_id(), 0);
    EXPECT_EQ(tok.eos_token_id(), 1);
    EXPECT_EQ(tok.unk_token_id(), 2);
}

TEST(BpeTokenizer, BuildEmptyFails) {
    BpeTokenizer tok;
    EXPECT_FALSE(tok.build({}, {}, {}, {}));
    EXPECT_FALSE(tok.is_valid());
}

TEST(BpeTokenizer, BuildMinimalVocab) {
    std::vector<std::string> tokens = {"<s>", "</s>", "a", "b"};
    BpeTokenizer tok;
    EXPECT_TRUE(tok.build(tokens, {}, {}, {}, 0, 1));
    EXPECT_EQ(tok.vocab_size(), 4);
}

// ============================================================================
// Encode tests
// ============================================================================

TEST(BpeTokenizer, EncodeEmptyText) {
    auto tok = make_test_tokenizer();
    auto tokens = tok.encode("", false, false);
    EXPECT_TRUE(tokens.empty());
}

TEST(BpeTokenizer, EncodeWithBos) {
    auto tok = make_test_tokenizer();
    auto tokens = tok.encode("", true, false);
    ASSERT_EQ(tokens.size(), 1u);
    EXPECT_EQ(tokens[0], tok.bos_token_id());
}

TEST(BpeTokenizer, EncodeWithBosAndEos) {
    auto tok = make_test_tokenizer();
    auto tokens = tok.encode("", true, true);
    ASSERT_EQ(tokens.size(), 2u);
    EXPECT_EQ(tokens[0], tok.bos_token_id());
    EXPECT_EQ(tokens[1], tok.eos_token_id());
}

TEST(BpeTokenizer, EncodeSingleCharacter) {
    // Test encoding a character that has a direct token match via BPE merge
    const std::string sp = "\xE2\x96\x81"; // ▁ (U+2581)
    const std::string sp_a = sp + "a";      // ▁a

    std::vector<std::string> tokens = {
        "<s>", "</s>",    // 0, 1
        sp,               // 2 - ▁
        sp_a,             // 3 - ▁a
        "a",              // 4
    };
    // Merge: ▁ + a -> ▁a
    std::vector<std::string> merges = {
        sp + " " + "a",
    };

    BpeTokenizer tok;
    tok.build(tokens, {}, {}, merges, 0, 1, false);

    auto result = tok.encode("a", false, false);
    // Should produce: ▁a (merged from ▁ + a)
    ASSERT_GE(result.size(), 1u);
    // The result should include token 3 (▁a)
    bool found = false;
    for (auto id : result) {
        if (id == 3) found = true;
    }
    EXPECT_TRUE(found);
}

// ============================================================================
// Decode tests
// ============================================================================

TEST(BpeTokenizer, DecodeKnownTokens) {
    auto tok = make_test_tokenizer();

    // Decode "Hello" + "world"
    std::vector<int32_t> input = {19, 20}; // Hello, world
    std::string result = tok.decode(input, true);
    EXPECT_FALSE(result.empty());
}

TEST(BpeTokenizer, DecodeSkipSpecial) {
    auto tok = make_test_tokenizer();

    std::vector<int32_t> input = {0, 1}; // BOS + EOS (CONTROL type)
    std::string result = tok.decode(input, true);
    EXPECT_TRUE(result.empty()); // Both are CONTROL, skipped
}

TEST(BpeTokenizer, DecodeDontSkipSpecial) {
    auto tok = make_test_tokenizer();

    std::vector<int32_t> input = {0, 1}; // BOS + EOS
    std::string result = tok.decode(input, false);
    // BOS and EOS should be present in output
    EXPECT_FALSE(result.empty());
    EXPECT_NE(result.find("<s>"), std::string::npos);
    EXPECT_NE(result.find("</s>"), std::string::npos);
}

TEST(BpeTokenizer, DecodeByteTokens) {
    auto tok = make_test_tokenizer();

    // Decode byte token for 'H' (id 6 = <0x48>)
    std::string result = tok.decode_token(6);
    EXPECT_EQ(result, "H");
}

// ============================================================================
// Roundtrip tests
// ============================================================================

TEST(BpeTokenizer, RoundtripSimpleWord) {
    // Create a simple tokenizer with ▁ convention
    const std::string sp = "\xE2\x96\x81";
    std::vector<std::string> tokens = {
        "<s>",            // 0
        "</s>",           // 1
        sp,               // 2 - ▁
        sp + "hello",     // 3 - ▁hello
        sp + "world",     // 4 - ▁world
    };
    std::vector<std::string> merges = {};

    BpeTokenizer tok;
    tok.build(tokens, {}, {}, merges, 0, 1, false);

    // Encode "hello world"
    auto encoded = tok.encode("hello world", false, false);
    // Should be: [3, 4] (▁hello, ▁world)
    ASSERT_EQ(encoded.size(), 2u);
    EXPECT_EQ(encoded[0], 3);
    EXPECT_EQ(encoded[1], 4);

    // Decode back
    std::string decoded = tok.decode(encoded, true);
    EXPECT_EQ(decoded, "hello world");
}

TEST(BpeTokenizer, RoundtripWithBosEos) {
    const std::string sp = "\xE2\x96\x81";
    std::vector<std::string> tokens = {
        "<s>",            // 0
        "</s>",           // 1
        sp,               // 2
        sp + "test",      // 3
    };
    std::vector<int32_t> token_types = {2, 2, 0, 0}; // BOS=CONTROL, EOS=CONTROL

    BpeTokenizer tok;
    tok.build(tokens, {}, token_types, {}, 0, 1, true);

    auto encoded = tok.encode("test", true, true);
    // Should be: [0, 3, 1] (BOS, ▁test, EOS)
    ASSERT_EQ(encoded.size(), 3u);
    EXPECT_EQ(encoded[0], 0); // BOS
    EXPECT_EQ(encoded[1], 3); // ▁test
    EXPECT_EQ(encoded[2], 1); // EOS

    // Decode (skip special)
    std::string decoded = tok.decode(encoded, true);
    EXPECT_EQ(decoded, "test");
}

// ============================================================================
// BPE merge tests
// ============================================================================

TEST(BpeTokenizer, BpeMergeApplies) {
    const std::string sp = "\xE2\x96\x81";
    std::vector<std::string> tokens = {
        "<s>",            // 0
        "</s>",           // 1
        "a",              // 2
        "b",              // 3
        "ab",             // 4
        sp,               // 5 - ▁
        sp + "ab",        // 6 - ▁ab
    };
    // Merge "a" + "b" -> "ab"
    std::vector<std::string> merges = {"a b"};

    BpeTokenizer tok;
    tok.build(tokens, {}, {}, merges, 0, 1, false);

    // Encode "ab" — should merge a+b into ab
    auto encoded = tok.encode("ab", false, false);
    ASSERT_GE(encoded.size(), 1u);
    // The result should contain the merged token
    bool has_merged = false;
    for (auto id : encoded) {
        if (id == 4 || id == 6) has_merged = true; // "ab" or "▁ab"
    }
    EXPECT_TRUE(has_merged);
}

// ============================================================================
// Special token detection
// ============================================================================

TEST(BpeTokenizer, SpecialTokenDetection) {
    auto tok = make_test_tokenizer();
    EXPECT_TRUE(tok.is_special_token(0));  // BOS (CONTROL type)
    EXPECT_TRUE(tok.is_special_token(1));  // EOS (CONTROL type)
    EXPECT_TRUE(tok.is_special_token(2));  // UNK (UNKNOWN type)
    EXPECT_FALSE(tok.is_special_token(13)); // ▁ (NORMAL type)
}

// ============================================================================
// Accessor tests
// ============================================================================

TEST(BpeTokenizer, TokenTextAccessor) {
    auto tok = make_test_tokenizer();
    EXPECT_EQ(tok.token_text(0), "<s>");
    EXPECT_EQ(tok.token_text(1), "</s>");
    EXPECT_EQ(tok.token_text(13), "\xE2\x96\x81");
}

TEST(BpeTokenizer, TokenTextOutOfRange) {
    auto tok = make_test_tokenizer();
    EXPECT_EQ(tok.token_text(-1), "");
    EXPECT_EQ(tok.token_text(999), "");
}

TEST(BpeTokenizer, AutoDetectBosEosFromTokenNames) {
    // Test auto-detection when bos_id/eos_id are not provided
    std::vector<std::string> tokens = {"<s>", "</s>", "a"};
    BpeTokenizer tok;
    tok.build(tokens, {}, {}, {}, -1, -1); // No explicit BOS/EOS IDs
    // Should auto-detect from token names
    EXPECT_EQ(tok.bos_token_id(), 0); // <s>
    EXPECT_EQ(tok.eos_token_id(), 1); // </s>
}

TEST(BpeTokenizer, PadTokenAutoDetect) {
    std::vector<std::string> tokens = {"<s>", "</s>", "<pad>", "a"};
    BpeTokenizer tok;
    tok.build(tokens, {}, {}, {}, 0, 1);
    EXPECT_EQ(tok.pad_token_id(), 2); // <pad>
}
