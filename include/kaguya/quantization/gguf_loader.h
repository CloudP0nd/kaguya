#pragma once
// Kaguya — gguf_loader
// GGUF format parser for loading model weights and metadata

#include <cstdint>
#include <cstddef>
#include <string>
#include "kaguya/core/tensor.h"
#include <vector>
#include <unordered_map>
#include <variant>
#include <memory>
#include <optional>

namespace kaguya {

// Forward declaration
class Tensor;

/// GGUF tensor data types (matches ggml_type)
enum class GgmlType : int32_t {
    F32       = 0,
    F16       = 1,
    Q4_0      = 2,
    Q4_1      = 3,
    Q5_0      = 6,
    Q5_1      = 7,
    Q8_0      = 8,
    Q8_1      = 9,
    Q2_K      = 10,
    Q3_K      = 11,
    Q4_K      = 12,
    Q5_K      = 13,
    Q6_K      = 14,
    Q8_K      = 15,
    IQ2_XXS   = 16,
    IQ2_XS    = 17,
    IQ3_XXS   = 18,
    IQ1_S     = 19,
    IQ4_NL    = 20,
    IQ3_S     = 21,
    IQ2_S     = 22,
    IQ4_XS    = 23,
    IQ1_M     = 24,
    BF16      = 30,
    Q4_0_4_4  = 31,
    Q4_0_4_8  = 32,
    Q4_0_8_8  = 33,
    TQ1_0     = 34,
    TQ2_0     = 35,
    IQ1_M_NL  = 36,
    UNKNOWN   = 999,
};

/// GGUF metadata value types
enum class GgufValueType : uint32_t {
    UINT8    = 0,
    INT8     = 1,
    UINT16   = 2,
    INT16    = 3,
    UINT32   = 4,
    INT32    = 5,
    FLOAT32  = 6,
    BOOL     = 7,
    STRING   = 8,
    ARRAY    = 9,
    UINT64   = 10,
    INT64    = 11,
    FLOAT64  = 12,
};

/// A metadata value — can hold any GGUF value type
struct GgufValue {
    using ArrayType = std::vector<GgufValue>;

    std::variant<
        uint8_t, int8_t,
        uint16_t, int16_t,
        uint32_t, int32_t,
        uint64_t, int64_t,
        float, double,
        bool,
        std::string,
        ArrayType
    > data;

    // Convenience accessors
    bool is_string() const { return std::holds_alternative<std::string>(data); }
    bool is_array() const { return std::holds_alternative<ArrayType>(data); }
    bool is_int() const;
    bool is_float() const;

    const std::string& as_string() const { return std::get<std::string>(data); }
    int64_t as_int() const;
    double as_float() const;
    const ArrayType& as_array() const { return std::get<ArrayType>(data); }
};

/// Describes a single tensor in the GGUF file
struct GgufTensorInfo {
    std::string name;          ///< Tensor name (e.g., "blk.0.attn_q.weight")
    uint32_t    n_dims;        ///< Number of dimensions
    std::vector<uint64_t> dims; ///< Dimension sizes
    GgmlType    type;          ///< Data type
    uint64_t    offset;        ///< Byte offset into data section
};

/// Loaded GGUF model data
struct GgufModel {
    // Header
    uint32_t version = 0;
    uint64_t tensor_count = 0;
    uint64_t metadata_kv_count = 0;

    // Metadata
    std::unordered_map<std::string, GgufValue> metadata;

    // Tensor descriptors
    std::vector<GgufTensorInfo> tensor_infos;

    // Tensor data (mmap'd or loaded)
    void*  tensor_data = nullptr;
    size_t tensor_data_size = 0;

    // Data section offset in the file
    uint64_t data_offset = 0;
};

/// GGUF file parser and loader
class GgufLoader {
public:
    GgufLoader();
    ~GgufLoader();

    /// Load a GGUF file
    /// @param path File path
    /// @param mmap Use memory-mapped I/O (default: true)
    /// @return true on success
    bool load(const std::string& path, bool mmap = true);

    /// Get the parsed model data
    const GgufModel& model() const { return model_; }

    /// Get a metadata value by key
    const GgufValue* metadata(const std::string& key) const;

    /// Get a metadata string by key
    std::optional<std::string> metadata_string(const std::string& key) const;

    /// Get a metadata integer by key
    std::optional<int64_t> metadata_int(const std::string& key) const;

    /// Get a metadata float by key
    std::optional<double> metadata_float(const std::string& key) const;

    /// Get tensor info by name
    const GgufTensorInfo* tensor_info(const std::string& name) const;

    /// Get pointer to tensor data by tensor info
    const void* tensor_data(const GgufTensorInfo& info) const;

    /// Get the GGUF version
    uint32_t version() const { return model_.version; }

    /// Print model summary
    void print_summary() const;

private:
    GgufModel model_;
    int fd_ = -1;       ///< File descriptor for mmap

    // Reading helpers
    bool read_header();
    bool read_metadata();
    bool read_tensor_infos();
    bool read_value(GgufValueType vtype, GgufValue& val);
    std::string read_gguf_string();
    bool align_to(size_t alignment);

    // Internal reader — abstracts file I/O
public: class Reader {
    public:
        virtual ~Reader() = default;
        virtual bool open(const std::string& path) = 0;
        virtual void close() = 0;
        virtual bool read(void* buf, size_t len) = 0;
        virtual bool skip(size_t len) = 0;
        virtual bool seek(size_t pos) = 0;
        virtual size_t tell() const = 0;
        virtual const uint8_t* data_ptr() const = 0;
        virtual size_t file_size() const = 0;
    };

    std::unique_ptr<Reader> reader_;
};

/// Utility: convert GgmlType to DataType
DataType ggml_to_data_type(GgmlType gt);

/// Utility: get block size for a GgmlType
int ggml_block_size(GgmlType gt);

/// Utility: get type size (bytes per block) for a GgmlType
size_t ggml_type_size(GgmlType gt);

/// Utility: calculate total bytes needed for a tensor
size_t ggml_nbytes(const GgufTensorInfo& info);

/// Utility: get human-readable name for GgmlType
const char* ggml_type_name(GgmlType gt);

} // namespace kaguya
