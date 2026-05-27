// Kaguya — GGUF parser and loader tests
#include <gtest/gtest.h>
#include "kaguya/gguf_loader.h"
#include "kaguya/model_loader.h"
#include "kaguya/model.h"
#include "kaguya/tensor.h"

using namespace kaguya;

// ---- GgmlType utility tests ----

TEST(GgufLoader, GgmlTypeNames) {
    EXPECT_STREQ(ggml_type_name(GgmlType::F32), "F32");
    EXPECT_STREQ(ggml_type_name(GgmlType::Q4_0), "Q4_0");
    EXPECT_STREQ(ggml_type_name(GgmlType::BF16), "BF16");
    EXPECT_STREQ(ggml_type_name(GgmlType::Q4_K), "Q4_K");
    EXPECT_STREQ(ggml_type_name(GgmlType::UNKNOWN), "UNKNOWN");
}

TEST(GgufLoader, BlockSizes) {
    EXPECT_EQ(ggml_block_size(GgmlType::F32), 1);
    EXPECT_EQ(ggml_block_size(GgmlType::F16), 1);
    EXPECT_EQ(ggml_block_size(GgmlType::BF16), 1);
    EXPECT_EQ(ggml_block_size(GgmlType::Q4_0), 32);
    EXPECT_EQ(ggml_block_size(GgmlType::Q8_0), 32);
    EXPECT_EQ(ggml_block_size(GgmlType::Q4_K), 256);
    EXPECT_EQ(ggml_block_size(GgmlType::Q8_K), 256);
}

TEST(GgufLoader, TypeSizes) {
    EXPECT_EQ(ggml_type_size(GgmlType::F32), 4u);
    EXPECT_EQ(ggml_type_size(GgmlType::F16), 2u);
    EXPECT_EQ(ggml_type_size(GgmlType::BF16), 2u);
    EXPECT_EQ(ggml_type_size(GgmlType::Q4_0), 18u);
    EXPECT_EQ(ggml_type_size(GgmlType::Q8_0), 34u);
    EXPECT_EQ(ggml_type_size(GgmlType::Q4_K), 144u);
}

TEST(GgufLoader, Nbytes) {
    GgufTensorInfo info;
    info.type = GgmlType::F32;
    info.dims = {1024};
    EXPECT_EQ(ggml_nbytes(info), 4096u);

    info.type = GgmlType::Q4_0;
    info.dims = {1024};
    // 1024 / 32 = 32 blocks * 18 bytes = 576
    EXPECT_EQ(ggml_nbytes(info), 576u);

    info.type = GgmlType::F32;
    info.dims = {4096, 4096};
    EXPECT_EQ(ggml_nbytes(info), 4096u * 4096 * 4);
}

// ---- DataType conversion ----

TEST(GgufLoader, GgmlToDataType) {
    EXPECT_EQ(ggml_to_data_type(GgmlType::F32), DataType::F32);
    EXPECT_EQ(ggml_to_data_type(GgmlType::Q4_0), DataType::Q4_0);
    EXPECT_EQ(ggml_to_data_type(GgmlType::BF16), DataType::BF16);
    EXPECT_EQ(ggml_to_data_type(GgmlType::Q4_K), DataType::Q4_K);
    EXPECT_EQ(ggml_to_data_type(GgmlType::IQ4_NL), DataType::IQ4_NL);
    EXPECT_EQ(ggml_to_data_type(GgmlType::TQ1_0), DataType::TQ1_0);
}

// ---- GgufValue tests ----

TEST(GgufValue, IntTypes) {
    GgufValue v;
    v.data = int32_t(42);
    EXPECT_TRUE(v.is_int());
    EXPECT_EQ(v.as_int(), 42);
}

TEST(GgufValue, FloatTypes) {
    GgufValue v;
    v.data = 3.14f;
    EXPECT_TRUE(v.is_float());
    EXPECT_FLOAT_EQ(static_cast<float>(v.as_float()), 3.14f);
}

TEST(GgufValue, StringType) {
    GgufValue v;
    v.data = std::string("llama");
    EXPECT_TRUE(v.is_string());
    EXPECT_EQ(v.as_string(), "llama");
}

TEST(GgufValue, ArrayType) {
    GgufValue v;
    GgufValue::ArrayType arr;
    arr.push_back(GgufValue{.data = int32_t(1)});
    arr.push_back(GgufValue{.data = int32_t(2)});
    arr.push_back(GgufValue{.data = int32_t(3)});
    v.data = std::move(arr);
    EXPECT_TRUE(v.is_array());
    EXPECT_EQ(v.as_array().size(), 3u);
}

// ---- Architecture detection ----

TEST(HyperParams, ArchFromString) {
    EXPECT_EQ(arch_from_string("llama"), ModelArch::LLAMA);
    EXPECT_EQ(arch_from_string("qwen2"), ModelArch::QWEN2);
    EXPECT_EQ(arch_from_string("mistral"), ModelArch::MISTRAL);
    EXPECT_EQ(arch_from_string("mixtral"), ModelArch::MIXTRAL);
    EXPECT_EQ(arch_from_string("phi3"), ModelArch::PHI3);
    EXPECT_EQ(arch_from_string("gemma"), ModelArch::GEMMA);
    EXPECT_EQ(arch_from_string("deepseek2"), ModelArch::DEEPSEEK);
    EXPECT_EQ(arch_from_string("command-r"), ModelArch::COMMAND_R);
    EXPECT_EQ(arch_from_string("unknown_arch"), ModelArch::UNKNOWN);
}

TEST(HyperParams, ArchToString) {
    EXPECT_STREQ(arch_to_string(ModelArch::LLAMA), "llama");
    EXPECT_STREQ(arch_to_string(ModelArch::QWEN2), "qwen2");
    EXPECT_STREQ(arch_to_string(ModelArch::UNKNOWN), "unknown");
}

// ---- GGUF Loader with invalid file ----

TEST(GgufLoader, LoadNonexistentFile) {
    GgufLoader loader;
    EXPECT_FALSE(loader.load("/nonexistent/path/model.gguf"));
}

// ---- HyperParams validation ----

TEST(HyperParams, ValidCheck) {
    HyperParams hp;
    EXPECT_FALSE(hp.valid()); // All zeros

    hp.vocab_size = 32000;
    hp.context_length = 4096;
    hp.emb_dim = 4096;
    hp.num_layers = 32;
    hp.num_heads = 32;
    EXPECT_TRUE(hp.valid());
}

// ---- DataType utilities ----

TEST(DataType, ExtendedTypes) {
    // IQ types
    EXPECT_EQ(data_type_block_size(DataType::IQ4_NL), 32);
    EXPECT_EQ(data_type_block_size(DataType::IQ4_XS), 256);
    EXPECT_EQ(data_type_block_size(DataType::IQ3_XXS), 256);
    EXPECT_EQ(data_type_element_size(DataType::IQ4_NL), 18u);
    EXPECT_EQ(data_type_element_size(DataType::IQ4_XS), 136u);
    EXPECT_STREQ(data_type_name(DataType::IQ4_NL), "IQ4_NL");

    // TQ types
    EXPECT_EQ(data_type_block_size(DataType::TQ1_0), 256);
    EXPECT_EQ(data_type_block_size(DataType::TQ2_0), 256);
    EXPECT_EQ(data_type_element_size(DataType::TQ1_0), 4u);
    EXPECT_EQ(data_type_element_size(DataType::TQ2_0), 64u);
    EXPECT_STREQ(data_type_name(DataType::TQ1_0), "TQ1_0");
}
