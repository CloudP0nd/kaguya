#pragma once

#include <cstdint>
#include <cstddef>
#include <vector>
#include <string>
#include <span>
#include <memory>

namespace kaguya {

/// Data type for tensor elements
enum class DataType : int {
    F32     = 0,
    F16     = 1,
    BF16    = 2,
    Q4_0    = 3,
    Q4_1    = 4,
    Q5_0    = 5,
    Q5_1    = 6,
    Q8_0    = 7,
    Q8_1    = 8,
    Q2_K    = 9,
    Q3_K    = 10,
    Q4_K    = 11,
    Q5_K    = 12,
    Q6_K    = 13,
    Q8_K    = 14,
    I8      = 15,
    I16     = 16,
    I32     = 17,
    I64     = 18,
    // IQ quantization types
    IQ4_NL  = 19,
    IQ4_XS  = 20,
    IQ3_XXS = 21,
    IQ1_S   = 22,
    IQ1_M   = 23,
    IQ2_XXS = 24,
    IQ2_XS  = 25,
    IQ3_S   = 26,
    IQ2_S   = 27,
    // TQ quantization types
    TQ1_0   = 28,
    TQ2_0   = 29,
    IQ1_M_NL = 30,
    UNKNOWN = 99,
};

/// Size in bytes of each element (or block for quantized types)
size_t data_type_element_size(DataType dt);

/// Number of elements per quantization block
int data_type_block_size(DataType dt);

/// Human-readable name for data type
const char* data_type_name(DataType dt);

/// Multi-dimensional tensor
class Tensor {
public:
    Tensor();
    Tensor(DataType dtype, std::vector<int64_t> shape, const std::string& name = "");

    /// Tensor name (for model weight identification)
    const std::string& name() const { return name_; }

    /// Data type
    DataType dtype() const { return dtype_; }

    /// Shape (dimensions)
    const std::vector<int64_t>& shape() const { return shape_; }

    /// Number of dimensions
    int ndim() const { return static_cast<int>(shape_.size()); }

    /// Total number of elements
    int64_t num_elements() const;

    /// Total size in bytes (including quantization padding)
    size_t num_bytes() const;

    /// Raw data pointer (mutable)
    void* data() { return data_.data(); }

    /// Raw data pointer (const)
    const void* data() const { return data_.data(); }

    /// Typed data access
    template<typename T>
    T* data_as() { return reinterpret_cast<T*>(data_.data()); }

    template<typename T>
    const T* data_as() const { return reinterpret_cast<const T*>(data_.data()); }

    /// Allocate storage for this tensor
    void allocate();

    /// Set external data (no copy, user manages lifetime)
    void set_external(void* ptr, size_t size);

    /// Reshape tensor (total elements must match)
    Tensor& reshape(std::vector<int64_t> new_shape);

private:
    std::string name_;
    DataType dtype_;
    std::vector<int64_t> shape_;
    std::vector<uint8_t> data_;
    void* external_data_ = nullptr;
    size_t external_size_ = 0;
};

} // namespace kaguya
