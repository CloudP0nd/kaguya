#include "kaguya/tensor.h"
#include <numeric>
#include <stdexcept>

namespace kaguya {

size_t data_type_element_size(DataType dt) {
    switch (dt) {
        case DataType::F32:    return 4;
        case DataType::F16:    return 2;
        case DataType::BF16:   return 2;
        case DataType::I8:     return 1;
        case DataType::I16:    return 2;
        case DataType::I32:    return 4;
        case DataType::I64:    return 8;
        // Quantized types: block-based, return block size in bytes
        case DataType::Q4_0:   return 18;  // 32 elements in 18 bytes (2 + 16)
        case DataType::Q4_1:   return 20;  // 32 elements in 20 bytes (2 + 2 + 16)
        case DataType::Q5_0:   return 22;  // 32 elements in 22 bytes
        case DataType::Q5_1:   return 24;  // 32 elements in 24 bytes
        case DataType::Q8_0:   return 34;  // 32 elements in 34 bytes (2 + 32)
        case DataType::Q8_1:   return 36;  // 32 elements in 36 bytes
        case DataType::Q2_K:   return 64;  // 256 elements in 64 bytes
        case DataType::Q3_K:   return 110; // 256 elements in 110 bytes
        case DataType::Q4_K:   return 144; // 256 elements in 144 bytes
        case DataType::Q5_K:   return 176; // 256 elements in 176 bytes
        case DataType::Q6_K:   return 210; // 256 elements in 210 bytes
        case DataType::Q8_K:   return 292; // 256 elements in 292 bytes
        // IQ quantization types
        case DataType::IQ4_NL:  return 18;  // 32 elements in 18 bytes
        case DataType::IQ4_XS:  return 136; // 256 elements in 136 bytes
        case DataType::IQ3_XXS: return 98;  // 256 elements in 98 bytes
        case DataType::IQ1_S:   return 36;  // 256 elements in 36 bytes
        case DataType::IQ1_M:   return 56;  // 256 elements in 56 bytes
        case DataType::IQ2_XXS: return 66;  // 256 elements in 66 bytes
        case DataType::IQ2_XS:  return 74;  // 256 elements in 74 bytes
        case DataType::IQ3_S:   return 110; // 256 elements in 110 bytes
        case DataType::IQ2_S:   return 82;  // 256 elements in 82 bytes
        // TQ quantization types
        case DataType::TQ1_0:   return 4;   // 256 elements in 4 bytes
        case DataType::TQ2_0:   return 64;  // 256 elements in 64 bytes
        case DataType::IQ1_M_NL:return 32;  // 256 elements in 32 bytes
        default:               return 0;
    }
}

int data_type_block_size(DataType dt) {
    switch (dt) {
        case DataType::Q4_0: case DataType::Q4_1:
        case DataType::Q5_0: case DataType::Q5_1:
        case DataType::Q8_0: case DataType::Q8_1:
        case DataType::IQ4_NL:
            return 32;
        case DataType::Q2_K: case DataType::Q3_K:
        case DataType::Q4_K: case DataType::Q5_K:
        case DataType::Q6_K: case DataType::Q8_K:
        case DataType::IQ4_XS: case DataType::IQ3_XXS:
        case DataType::IQ1_S: case DataType::IQ1_M:
        case DataType::IQ2_XXS: case DataType::IQ2_XS:
        case DataType::IQ3_S: case DataType::IQ2_S:
        case DataType::TQ1_0: case DataType::TQ2_0:
        case DataType::IQ1_M_NL:
            return 256;
        default:
            return 1;
    }
}

const char* data_type_name(DataType dt) {
    switch (dt) {
        case DataType::F32:    return "F32";
        case DataType::F16:    return "F16";
        case DataType::BF16:   return "BF16";
        case DataType::Q4_0:   return "Q4_0";
        case DataType::Q4_1:   return "Q4_1";
        case DataType::Q5_0:   return "Q5_0";
        case DataType::Q5_1:   return "Q5_1";
        case DataType::Q8_0:   return "Q8_0";
        case DataType::Q8_1:   return "Q8_1";
        case DataType::Q2_K:   return "Q2_K";
        case DataType::Q3_K:   return "Q3_K";
        case DataType::Q4_K:   return "Q4_K";
        case DataType::Q5_K:   return "Q5_K";
        case DataType::Q6_K:   return "Q6_K";
        case DataType::Q8_K:   return "Q8_K";
        case DataType::I8:     return "I8";
        case DataType::I16:    return "I16";
        case DataType::I32:    return "I32";
        case DataType::I64:    return "I64";
        case DataType::IQ4_NL:  return "IQ4_NL";
        case DataType::IQ4_XS:  return "IQ4_XS";
        case DataType::IQ3_XXS: return "IQ3_XXS";
        case DataType::IQ1_S:   return "IQ1_S";
        case DataType::IQ1_M:   return "IQ1_M";
        case DataType::IQ2_XXS: return "IQ2_XXS";
        case DataType::IQ2_XS:  return "IQ2_XS";
        case DataType::IQ3_S:   return "IQ3_S";
        case DataType::IQ2_S:   return "IQ2_S";
        case DataType::TQ1_0:   return "TQ1_0";
        case DataType::TQ2_0:   return "TQ2_0";
        case DataType::IQ1_M_NL:return "IQ1_M_NL";
        default:               return "UNKNOWN";
    }
}

// ---- Tensor ----

Tensor::Tensor() : dtype_(DataType::F32) {}

Tensor::Tensor(DataType dtype, std::vector<int64_t> shape, const std::string& name)
    : name_(name), dtype_(dtype), shape_(std::move(shape)) {}

int64_t Tensor::num_elements() const {
    if (shape_.empty()) return 0;
    int64_t total = 1;
    for (auto d : shape_) total *= d;
    return total;
}

size_t Tensor::num_bytes() const {
    int64_t n = num_elements();
    int bs = data_type_block_size(dtype_);
    size_t es = data_type_element_size(dtype_);
    // Quantized: round up to block boundary, then compute
    int64_t num_blocks = (n + bs - 1) / bs;
    return num_blocks * es;
}

void Tensor::allocate() {
    size_t sz = num_bytes();
    data_.resize(sz, 0);
    external_data_ = nullptr;
}

void Tensor::set_external(void* ptr, size_t size) {
    external_data_ = ptr;
    external_size_ = size;
    data_.clear();
}

Tensor& Tensor::reshape(std::vector<int64_t> new_shape) {
    // Validate: total elements must match
    int64_t new_total = 1;
    int neg_idx = -1;
    for (int i = 0; i < (int)new_shape.size(); ++i) {
        if (new_shape[i] == -1) {
            if (neg_idx >= 0) throw std::runtime_error("Multiple -1 in reshape");
            neg_idx = i;
        } else {
            new_total *= new_shape[i];
        }
    }
    if (neg_idx >= 0) {
        new_shape[neg_idx] = num_elements() / new_total;
        new_total *= new_shape[neg_idx];
    }
    if (new_total != num_elements()) {
        throw std::runtime_error("Reshape: element count mismatch");
    }
    shape_ = std::move(new_shape);
    return *this;
}

} // namespace kaguya
