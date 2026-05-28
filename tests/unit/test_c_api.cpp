#include <gtest/gtest.h>
#include "kaguya/kaguya.h"
#include "kaguya/cpu_features.h"
#include "kaguya/memory_manager.h"

#include <cstring>

// ============================================================================
// Version info
// ============================================================================

TEST(CApi, VersionString) {
    const char* ver = kaguya_version();
    ASSERT_NE(ver, nullptr);
    EXPECT_STREQ(ver, "0.2.0");
}

// ============================================================================
// Initialization
// ============================================================================

TEST(CApi, InitDoesNotCrash) {
    // Can be called multiple times
    kaguya_init();
    kaguya_init();
}

TEST(CApi, CpuInfoNotNull) {
    kaguya_init();
    const char* info = kaguya_cpu_info();
    ASSERT_NE(info, nullptr);
    EXPECT_GT(std::strlen(info), 0u);
}

// ============================================================================
// Model loading (requires GGUF file — test error handling)
// ============================================================================

TEST(CApi, ModelLoadInvalidPath) {
    kaguya_model* model = kaguya_model_load("/nonexistent/model.gguf");
    EXPECT_EQ(model, nullptr);
}

TEST(CApi, ModelLoadNullPath) {
    kaguya_model* model = kaguya_model_load(nullptr);
    EXPECT_EQ(model, nullptr);
}

TEST(CApi, ModelFreeNull) {
    // Should not crash
    kaguya_model_free(nullptr);
}

// ============================================================================
// Context creation without model
// ============================================================================

TEST(CApi, ContextCreateNullModel) {
    kaguya_context* ctx = kaguya_context_create(nullptr);
    EXPECT_EQ(ctx, nullptr);
}

TEST(CApi, ContextFreeNull) {
    // Should not crash
    kaguya_context_free(nullptr);
}

// ============================================================================
// Tokenization
// ============================================================================

TEST(CApi, TokenizeBasic) {
    kaguya_init();
    // ctx can be nullptr for simple byte-level tokenization
    int32_t* tokens = nullptr;
    int64_t count = 0;
    int rc = kaguya_tokenize(nullptr, "Hello", &tokens, &count);
    EXPECT_EQ(rc, 0);
    ASSERT_NE(tokens, nullptr);
    EXPECT_EQ(count, 5);

    // Verify byte values
    EXPECT_EQ(tokens[0], 'H');
    EXPECT_EQ(tokens[1], 'e');
    EXPECT_EQ(tokens[2], 'l');
    EXPECT_EQ(tokens[3], 'l');
    EXPECT_EQ(tokens[4], 'o');

    kaguya_tokens_free(tokens);
}

TEST(CApi, TokenizeEmpty) {
    int32_t* tokens = nullptr;
    int64_t count = 0;
    int rc = kaguya_tokenize(nullptr, "", &tokens, &count);
    EXPECT_EQ(rc, 0);
    EXPECT_EQ(count, 0);
    // tokens may be nullptr for empty input
    kaguya_tokens_free(tokens);
}

TEST(CApi, TokenizeNullArgs) {
    int rc = kaguya_tokenize(nullptr, nullptr, nullptr, nullptr);
    EXPECT_EQ(rc, -1);
}

TEST(CApi, DetokenizeBasic) {
    int32_t tokens[] = {'H', 'e', 'l', 'l', 'o'};
    char* text = kaguya_detokenize(nullptr, tokens, 5);
    ASSERT_NE(text, nullptr);
    EXPECT_STREQ(text, "Hello");
    kaguya_text_free(text);
}

TEST(CApi, DetokenizeNullTokens) {
    char* text = kaguya_detokenize(nullptr, nullptr, 0);
    EXPECT_EQ(text, nullptr);
}

TEST(CApi, TextFreeNull) {
    // Should not crash
    kaguya_text_free(nullptr);
}

TEST(CApi, TokensFreeNull) {
    // Should not crash
    kaguya_tokens_free(nullptr);
}

// ============================================================================
// Model info functions (null model)
// ============================================================================

TEST(CApi, ModelArchNull) {
    const char* arch = kaguya_model_arch(nullptr);
    EXPECT_EQ(arch, nullptr);
}

TEST(CApi, ModelVocabSizeNull) {
    int64_t vs = kaguya_model_vocab_size(nullptr);
    EXPECT_EQ(vs, 0);
}

TEST(CApi, ModelContextLengthNull) {
    int64_t cl = kaguya_model_context_length(nullptr);
    EXPECT_EQ(cl, 0);
}

TEST(CApi, ModelEmbDimNull) {
    int64_t ed = kaguya_model_emb_dim(nullptr);
    EXPECT_EQ(ed, 0);
}

TEST(CApi, ModelNumLayersNull) {
    int64_t nl = kaguya_model_num_layers(nullptr);
    EXPECT_EQ(nl, 0);
}

TEST(CApi, ModelNumHeadsNull) {
    int64_t nh = kaguya_model_num_heads(nullptr);
    EXPECT_EQ(nh, 0);
}

// ============================================================================
// Context functions (null context)
// ============================================================================

TEST(CApi, ContextResetNull) {
    // Should not crash
    kaguya_context_reset(nullptr);
}

TEST(CApi, ContextPositionNull) {
    int64_t pos = kaguya_context_position(nullptr);
    EXPECT_EQ(pos, -1);
}

TEST(CApi, ContextDecodeNull) {
    int32_t tok = kaguya_context_decode(nullptr, 0.8f, 40, 0.95f, 1.0f);
    EXPECT_EQ(tok, -1);
}

TEST(CApi, ContextGenerateNull) {
    int64_t count = 0;
    int32_t* tokens = kaguya_context_generate(nullptr, 10, 0.8f, 40, 0.95f, 1.0f, &count);
    EXPECT_EQ(tokens, nullptr);
}

TEST(CApi, ContextLogitsNull) {
    int64_t count = 0;
    const float* logits = kaguya_context_logits(nullptr, &count);
    EXPECT_EQ(logits, nullptr);
}

TEST(CApi, ContextPromptTokensNull) {
    int rc = kaguya_context_prompt_tokens(nullptr, nullptr, 0);
    EXPECT_EQ(rc, -1);
}
