#pragma once
#include <cstdint>
#include <cstddef>
#include <memory>
#include <vector>
#include <string>
#include <stdexcept>

namespace kaguya {

enum class DataType {
    F32,
    F16,
    BF16,
    Q4_0,
    Q4_1,
    Q5_0,
    Q5_1,
    Q8_0,
    Q8_1,
    Q2_K,
    Q3_K,
    IQ4_NL,
    IQ4_XS,
    IQ3_XXS,
    IQ1_S,
    IQ1_M,
    IQ2_XXS,
    IQ2_XS,
    IQ3_S,
    IQ2_S,
    IQ4_S,
    IQ1_M_NL,
    Q4_K,
    Q5_K,
    Q6_K,
    Q8_K,
    TQ1_0,
    TQ2_0,
};

class Tensor {
public:
    Tensor() = default;
    Tensor(std::vector<size_t> shape, DataType dtype);
    Tensor(void* data, std::vector<size_t> shape, DataType dtype, bool own = false);

    ~Tensor();

    // Non-copyable, movable
    Tensor(const Tensor&) = delete;
    Tensor& operator=(const Tensor&) = delete;
    Tensor(Tensor&& other) noexcept;
    Tensor& operator=(Tensor&& other) noexcept;

    // Accessors
    void* data() { return data_; }
    const void* data() const { return data_; }
    size_t ne(size_t dim) const { return shape_[dim]; }
    size_t size() const { return size_; }
    DataType dtype() const { return dtype_; }
    size_t nbytes() const { return nbytes_; }
    size_t ndim() const { return shape_.size(); }
    const std::vector<size_t>& shape() const { return shape_; }

    // Mutators
    void reshape(std::vector<size_t> new_shape);
    Tensor view(std::vector<size_t> new_shape) const;

    // Zero-fill
    void zero();

    // Type info
    static size_t type_size(DataType dtype);
    static bool is_quantized(DataType dtype);
    static bool is_float(DataType dtype);
    static std::string dtype_name(DataType dtype);

private:
    void* data_ = nullptr;
    std::vector<size_t> shape_;
    size_t size_ = 0;
    size_t nbytes_ = 0;
    DataType dtype_ = DataType::F32;
    bool own_ = false;
};

} // namespace kaguya
