// Kaguya — gguf_loader
// Full GGUF format parser with mmap support

#include "kaguya/gguf_loader.h"
#include "kaguya/tensor.h"

#include <cstring>
#include <iostream>
#include <algorithm>
#include <cmath>

#include <fstream>

#ifdef __linux__
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#endif

namespace kaguya {

// ========================================================================
// GgufValue convenience methods
// ========================================================================

bool GgufValue::is_int() const {
    return std::holds_alternative<uint8_t>(data)  ||
           std::holds_alternative<int8_t>(data)   ||
           std::holds_alternative<uint16_t>(data) ||
           std::holds_alternative<int16_t>(data)  ||
           std::holds_alternative<uint32_t>(data) ||
           std::holds_alternative<int32_t>(data)  ||
           std::holds_alternative<uint64_t>(data) ||
           std::holds_alternative<int64_t>(data)  ||
           std::holds_alternative<bool>(data);
}

bool GgufValue::is_float() const {
    return std::holds_alternative<float>(data) || std::holds_alternative<double>(data);
}

int64_t GgufValue::as_int() const {
    if (auto* v = std::get_if<uint8_t>(&data))  return *v;
    if (auto* v = std::get_if<int8_t>(&data))   return *v;
    if (auto* v = std::get_if<uint16_t>(&data)) return *v;
    if (auto* v = std::get_if<int16_t>(&data))  return *v;
    if (auto* v = std::get_if<uint32_t>(&data)) return *v;
    if (auto* v = std::get_if<int32_t>(&data))  return *v;
    if (auto* v = std::get_if<uint64_t>(&data)) return static_cast<int64_t>(*v);
    if (auto* v = std::get_if<int64_t>(&data))  return *v;
    if (auto* v = std::get_if<bool>(&data))     return *v ? 1 : 0;
    return 0;
}

double GgufValue::as_float() const {
    if (auto* v = std::get_if<float>(&data))  return *v;
    if (auto* v = std::get_if<double>(&data)) return *v;
    if (is_int()) return static_cast<double>(as_int());
    return 0.0;
}

// ========================================================================
// Utility functions
// ========================================================================

const char* ggml_type_name(GgmlType gt) {
    switch (gt) {
        case GgmlType::F32:      return "F32";
        case GgmlType::F16:      return "F16";
        case GgmlType::BF16:     return "BF16";
        case GgmlType::Q4_0:     return "Q4_0";
        case GgmlType::Q4_1:     return "Q4_1";
        case GgmlType::Q5_0:     return "Q5_0";
        case GgmlType::Q5_1:     return "Q5_1";
        case GgmlType::Q8_0:     return "Q8_0";
        case GgmlType::Q8_1:     return "Q8_1";
        case GgmlType::Q2_K:     return "Q2_K";
        case GgmlType::Q3_K:     return "Q3_K";
        case GgmlType::Q4_K:     return "Q4_K";
        case GgmlType::Q5_K:     return "Q5_K";
        case GgmlType::Q6_K:     return "Q6_K";
        case GgmlType::Q8_K:     return "Q8_K";
        case GgmlType::IQ2_XXS:  return "IQ2_XXS";
        case GgmlType::IQ2_XS:   return "IQ2_XS";
        case GgmlType::IQ3_XXS:  return "IQ3_XXS";
        case GgmlType::IQ1_S:    return "IQ1_S";
        case GgmlType::IQ4_NL:   return "IQ4_NL";
        case GgmlType::IQ3_S:    return "IQ3_S";
        case GgmlType::IQ2_S:    return "IQ2_S";
        case GgmlType::IQ4_XS:   return "IQ4_XS";
        case GgmlType::IQ1_M:    return "IQ1_M";
        case GgmlType::TQ1_0:    return "TQ1_0";
        case GgmlType::TQ2_0:    return "TQ2_0";
        case GgmlType::IQ1_M_NL: return "IQ1_M_NL";
        default:                  return "UNKNOWN";
    }
}

int ggml_block_size(GgmlType gt) {
    switch (gt) {
        case GgmlType::Q4_0:
        case GgmlType::Q4_1:
        case GgmlType::Q5_0:
        case GgmlType::Q5_1:
        case GgmlType::Q8_0:
        case GgmlType::Q8_1:
        case GgmlType::IQ4_NL:
            return 32;
        case GgmlType::Q2_K:
        case GgmlType::Q3_K:
        case GgmlType::Q4_K:
        case GgmlType::Q5_K:
        case GgmlType::Q6_K:
        case GgmlType::Q8_K:
        case GgmlType::IQ2_XXS:
        case GgmlType::IQ2_XS:
        case GgmlType::IQ3_XXS:
        case GgmlType::IQ1_S:
        case GgmlType::IQ3_S:
        case GgmlType::IQ2_S:
        case GgmlType::IQ4_XS:
        case GgmlType::IQ1_M:
        case GgmlType::TQ1_0:
        case GgmlType::TQ2_0:
        case GgmlType::IQ1_M_NL:
            return 256;
        default:
            return 1; // F32, F16, BF16
    }
}

size_t ggml_type_size(GgmlType gt) {
    switch (gt) {
        case GgmlType::F32:      return 4;
        case GgmlType::F16:      return 2;
        case GgmlType::BF16:     return 2;
        case GgmlType::Q4_0:     return 18;   // 32 elements in 18 bytes
        case GgmlType::Q4_1:     return 20;   // 32 elements in 20 bytes
        case GgmlType::Q5_0:     return 22;   // 32 elements in 22 bytes
        case GgmlType::Q5_1:     return 24;   // 32 elements in 24 bytes
        case GgmlType::Q8_0:     return 34;   // 32 elements in 34 bytes
        case GgmlType::Q8_1:     return 36;   // 32 elements in 36 bytes
        case GgmlType::Q2_K:     return 64;   // 256 elements in 64 bytes
        case GgmlType::Q3_K:     return 110;  // 256 elements in 110 bytes
        case GgmlType::Q4_K:     return 144;  // 256 elements in 144 bytes
        case GgmlType::Q5_K:     return 176;  // 256 elements in 176 bytes
        case GgmlType::Q6_K:     return 210;  // 256 elements in 210 bytes
        case GgmlType::Q8_K:     return 292;  // 256 elements in 292 bytes
        case GgmlType::IQ2_XXS:  return 66;
        case GgmlType::IQ2_XS:   return 74;
        case GgmlType::IQ3_XXS:  return 98;
        case GgmlType::IQ1_S:    return 36;
        case GgmlType::IQ4_NL:   return 18;
        case GgmlType::IQ3_S:    return 110;
        case GgmlType::IQ2_S:    return 82;
        case GgmlType::IQ4_XS:   return 136;
        case GgmlType::IQ1_M:    return 56;
        case GgmlType::TQ1_0:    return 4;    // per 256 elements
        case GgmlType::TQ2_0:    return 64;   // per 256 elements
        case GgmlType::IQ1_M_NL: return 32;
        default:                  return 0;
    }
}

DataType ggml_to_data_type(GgmlType gt) {
    switch (gt) {
        case GgmlType::F32:      return DataType::F32;
        case GgmlType::F16:      return DataType::F16;
        case GgmlType::BF16:     return DataType::BF16;
        case GgmlType::Q4_0:     return DataType::Q4_0;
        case GgmlType::Q4_1:     return DataType::Q4_1;
        case GgmlType::Q5_0:     return DataType::Q5_0;
        case GgmlType::Q5_1:     return DataType::Q5_1;
        case GgmlType::Q8_0:     return DataType::Q8_0;
        case GgmlType::Q8_1:     return DataType::Q8_1;
        case GgmlType::Q2_K:     return DataType::Q2_K;
        case GgmlType::Q3_K:     return DataType::Q3_K;
        case GgmlType::Q4_K:     return DataType::Q4_K;
        case GgmlType::Q5_K:     return DataType::Q5_K;
        case GgmlType::Q6_K:     return DataType::Q6_K;
        case GgmlType::Q8_K:     return DataType::Q8_K;
        case GgmlType::IQ4_NL:   return DataType::IQ4_NL;
        case GgmlType::IQ4_XS:   return DataType::IQ4_XS;
        case GgmlType::IQ3_XXS:  return DataType::IQ3_XXS;
        case GgmlType::IQ1_S:    return DataType::IQ1_S;
        case GgmlType::IQ1_M:    return DataType::IQ1_M;
        case GgmlType::IQ2_XXS:  return DataType::IQ2_XXS;
        case GgmlType::IQ2_XS:   return DataType::IQ2_XS;
        case GgmlType::IQ3_S:    return DataType::IQ3_S;
        case GgmlType::IQ2_S:    return DataType::IQ2_S;
        case GgmlType::TQ1_0:    return DataType::TQ1_0;
        case GgmlType::TQ2_0:    return DataType::TQ2_0;
        case GgmlType::IQ1_M_NL: return DataType::IQ1_M_NL;
        default:                  return DataType::F32;
    }
}

size_t ggml_nbytes(const GgufTensorInfo& info) {
    if (info.dims.empty()) return 0;

    // Total number of elements
    size_t n_elements = 1;
    for (auto d : info.dims) n_elements *= d;

    int bs = ggml_block_size(info.type);
    size_t ts = ggml_type_size(info.type);

    if (bs <= 1) {
        return n_elements * ts;
    }

    // Round up to block boundary
    size_t n_blocks = (n_elements + bs - 1) / bs;
    return n_blocks * ts;
}

// ========================================================================
// Reader — abstracts file I/O (mmap vs read)
// ========================================================================

class GgufLoader::Reader {
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

#ifdef __linux__
/// Mmap-based reader (preferred for large files)
class MmapReader : public GgufLoader::Reader {
public:
    MmapReader() = default;
    ~MmapReader() override { close(); }

    bool open(const std::string& path) override {
        fd_ = ::open(path.c_str(), O_RDONLY);
        if (fd_ < 0) {
            std::cerr << "Kaguya: Failed to open file: " << path << "\n";
            return false;
        }

        struct stat st;
        if (fstat(fd_, &st) != 0) {
            std::cerr << "Kaguya: Failed to stat file: " << path << "\n";
            ::close(fd_);
            fd_ = -1;
            return false;
        }
        file_size_ = st.st_size;

        data_ = static_cast<uint8_t*>(mmap(nullptr, file_size_, PROT_READ, MAP_PRIVATE, fd_, 0));
        if (data_ == MAP_FAILED) {
            std::cerr << "Kaguya: Failed to mmap file: " << path << "\n";
            ::close(fd_);
            fd_ = -1;
            data_ = nullptr;
            return false;
        }

        // Advise sequential access initially
        madvise(data_, file_size_, MADV_SEQUENTIAL);

        pos_ = 0;
        return true;
    }

    void close() override {
        if (data_) {
            munmap(data_, file_size_);
            data_ = nullptr;
        }
        if (fd_ >= 0) {
            ::close(fd_);
            fd_ = -1;
        }
        pos_ = 0;
        file_size_ = 0;
    }

    bool read(void* buf, size_t len) override {
        if (pos_ + len > file_size_) return false;
        memcpy(buf, data_ + pos_, len);
        pos_ += len;
        return true;
    }

    bool skip(size_t len) override {
        if (pos_ + len > file_size_) return false;
        pos_ += len;
        return true;
    }

    bool seek(size_t pos) override {
        if (pos > file_size_) return false;
        pos_ = pos;
        return true;
    }

    size_t tell() const override { return pos_; }
    const uint8_t* data_ptr() const override { return data_; }
    size_t file_size() const override { return file_size_; }

private:
    int fd_ = -1;
    uint8_t* data_ = nullptr;
    size_t file_size_ = 0;
    size_t pos_ = 0;
};
#endif

/// Stream-based reader (fallback for non-Linux)
class StreamReader : public GgufLoader::Reader {
public:
    StreamReader() = default;
    ~StreamReader() override { close(); }

    bool open(const std::string& path) override {
        file_.open(path, std::ios::binary);
        if (!file_.is_open()) {
            std::cerr << "Kaguya: Failed to open file: " << path << "\n";
            return false;
        }
        file_.seekg(0, std::ios::end);
        file_size_ = file_.tellg();
        file_.seekg(0, std::ios::beg);
        // Load entire file into memory
        buffer_.resize(file_size_);
        file_.read(reinterpret_cast<char*>(buffer_.data()), file_size_);
        if (!file_.good()) {
            std::cerr << "Kaguya: Failed to read file into memory\n";
            return false;
        }
        pos_ = 0;
        return true;
    }

    void close() override {
        file_.close();
        buffer_.clear();
        buffer_.shrink_to_fit();
        pos_ = 0;
        file_size_ = 0;
    }

    bool read(void* buf, size_t len) override {
        if (pos_ + len > buffer_.size()) return false;
        memcpy(buf, buffer_.data() + pos_, len);
        pos_ += len;
        return true;
    }

    bool skip(size_t len) override {
        if (pos_ + len > buffer_.size()) return false;
        pos_ += len;
        return true;
    }

    bool seek(size_t pos) override {
        if (pos > buffer_.size()) return false;
        pos_ = pos;
        return true;
    }

    size_t tell() const override { return pos_; }
    const uint8_t* data_ptr() const override { return buffer_.data(); }
    size_t file_size() const override { return file_size_; }

private:
    std::ifstream file_;
    std::vector<uint8_t> buffer_;
    size_t pos_ = 0;
    size_t file_size_ = 0;
};

// ========================================================================
// GgufLoader implementation
// ========================================================================

GgufLoader::GgufLoader() = default;

GgufLoader::~GgufLoader() = default;

bool GgufLoader::load(const std::string& path, bool mmap) {
    // Create appropriate reader
#ifdef __linux__
    if (mmap) {
        reader_ = std::make_unique<MmapReader>();
    } else {
        reader_ = std::make_unique<StreamReader>();
    }
#else
    (void)mmap;
    reader_ = std::make_unique<StreamReader>();
#endif

    if (!reader_->open(path)) {
        return false;
    }

    std::cout << "Kaguya: Loading GGUF file: " << path << "\n";

    // Parse
    if (!read_header()) return false;
    if (!read_metadata()) return false;
    if (!read_tensor_infos()) return false;

    // Calculate data section offset (aligned to 32 or 64 bytes)
    size_t alignment = 32;
    // GGUFv3 uses 64-byte alignment for data section
    if (model_.version >= 3) alignment = 64;

    if (!align_to(alignment)) return false;
    model_.data_offset = reader_->tell();

    // Set tensor data pointer (points into mmap'd memory)
    model_.tensor_data = const_cast<uint8_t*>(reader_->data_ptr()) + model_.data_offset;
    model_.tensor_data_size = reader_->file_size() - model_.data_offset;

    std::cout << "Kaguya: GGUF v" << model_.version
              << " | " << model_.tensor_count << " tensors"
              << " | " << model_.metadata_kv_count << " metadata keys"
              << " | data offset: " << model_.data_offset << "\n";

    return true;
}

bool GgufLoader::read_header() {
    constexpr uint32_t GGUF_MAGIC = 0x46475547; // "GGUF"

    uint32_t magic;
    if (!reader_->read(&magic, sizeof(magic))) {
        std::cerr << "Kaguya: Failed to read GGUF magic\n";
        return false;
    }

    if (magic != GGUF_MAGIC) {
        std::cerr << "Kaguya: Invalid GGUF magic: 0x" << std::hex << magic << std::dec
                  << " (expected 0x" << std::hex << GGUF_MAGIC << std::dec << ")\n";
        return false;
    }

    if (!reader_->read(&model_.version, sizeof(model_.version))) {
        std::cerr << "Kaguya: Failed to read GGUF version\n";
        return false;
    }

    if (model_.version < 2 || model_.version > 3) {
        std::cerr << "Kaguya: Unsupported GGUF version: " << model_.version << "\n";
        return false;
    }

    if (!reader_->read(&model_.tensor_count, sizeof(model_.tensor_count))) {
        std::cerr << "Kaguya: Failed to read tensor count\n";
        return false;
    }

    if (!reader_->read(&model_.metadata_kv_count, sizeof(model_.metadata_kv_count))) {
        std::cerr << "Kaguya: Failed to read metadata count\n";
        return false;
    }

    return true;
}

bool GgufLoader::read_metadata() {
    for (uint64_t i = 0; i < model_.metadata_kv_count; ++i) {
        // Read key
        std::string key = read_gguf_string();
        if (key.empty() && reader_->tell() == 0) {
            std::cerr << "Kaguya: Failed to read metadata key #" << i << "\n";
            return false;
        }

        // Read value type
        uint32_t vtype_raw;
        if (!reader_->read(&vtype_raw, sizeof(vtype_raw))) {
            std::cerr << "Kaguya: Failed to read metadata value type for key: " << key << "\n";
            return false;
        }

        GgufValueType vtype = static_cast<GgufValueType>(vtype_raw);

        // Read value
        GgufValue val;
        if (!read_value(vtype, val)) {
            std::cerr << "Kaguya: Failed to read metadata value for key: " << key << "\n";
            return false;
        }

        model_.metadata[key] = std::move(val);
    }
    return true;
}

bool GgufLoader::read_tensor_infos() {
    model_.tensor_infos.reserve(model_.tensor_count);

    for (uint64_t i = 0; i < model_.tensor_count; ++i) {
        GgufTensorInfo ti;

        // Name
        ti.name = read_gguf_string();

        // Number of dimensions
        if (!reader_->read(&ti.n_dims, sizeof(ti.n_dims))) {
            std::cerr << "Kaguya: Failed to read n_dims for tensor #" << i << "\n";
            return false;
        }

        // Dimensions
        ti.dims.resize(ti.n_dims);
        for (uint32_t d = 0; d < ti.n_dims; ++d) {
            if (!reader_->read(&ti.dims[d], sizeof(ti.dims[d]))) {
                std::cerr << "Kaguya: Failed to read dim for tensor #" << i << "\n";
                return false;
            }
        }

        // Type
        int32_t type_raw;
        if (!reader_->read(&type_raw, sizeof(type_raw))) {
            std::cerr << "Kaguya: Failed to read type for tensor #" << i << "\n";
            return false;
        }
        ti.type = static_cast<GgmlType>(type_raw);

        // Offset
        if (!reader_->read(&ti.offset, sizeof(ti.offset))) {
            std::cerr << "Kaguya: Failed to read offset for tensor #" << i << "\n";
            return false;
        }

        model_.tensor_infos.push_back(std::move(ti));
    }
    return true;
}

bool GgufLoader::read_value(GgufValueType vtype, GgufValue& val) {
    switch (vtype) {
        case GgufValueType::UINT8: {
            uint8_t v; if (!reader_->read(&v, 1)) return false;
            val.data = v; break;
        }
        case GgufValueType::INT8: {
            int8_t v; if (!reader_->read(&v, 1)) return false;
            val.data = v; break;
        }
        case GgufValueType::UINT16: {
            uint16_t v; if (!reader_->read(&v, 2)) return false;
            val.data = v; break;
        }
        case GgufValueType::INT16: {
            int16_t v; if (!reader_->read(&v, 2)) return false;
            val.data = v; break;
        }
        case GgufValueType::UINT32: {
            uint32_t v; if (!reader_->read(&v, 4)) return false;
            val.data = v; break;
        }
        case GgufValueType::INT32: {
            int32_t v; if (!reader_->read(&v, 4)) return false;
            val.data = v; break;
        }
        case GgufValueType::FLOAT32: {
            float v; if (!reader_->read(&v, 4)) return false;
            val.data = v; break;
        }
        case GgufValueType::BOOL: {
            uint8_t v; if (!reader_->read(&v, 1)) return false;
            val.data = static_cast<bool>(v); break;
        }
        case GgufValueType::STRING: {
            std::string s = read_gguf_string();
            val.data = std::move(s); break;
        }
        case GgufValueType::UINT64: {
            uint64_t v; if (!reader_->read(&v, 8)) return false;
            val.data = v; break;
        }
        case GgufValueType::INT64: {
            int64_t v; if (!reader_->read(&v, 8)) return false;
            val.data = v; break;
        }
        case GgufValueType::FLOAT64: {
            double v; if (!reader_->read(&v, 8)) return false;
            val.data = v; break;
        }
        case GgufValueType::ARRAY: {
            uint32_t elem_type_raw;
            if (!reader_->read(&elem_type_raw, sizeof(elem_type_raw))) return false;
            GgufValueType elem_type = static_cast<GgufValueType>(elem_type_raw);

            uint64_t count;
            if (!reader_->read(&count, sizeof(count))) return false;

            GgufValue::ArrayType arr;
            arr.reserve(count);
            for (uint64_t i = 0; i < count; ++i) {
                GgufValue elem;
                if (!read_value(elem_type, elem)) return false;
                arr.push_back(std::move(elem));
            }
            val.data = std::move(arr); break;
        }
        default:
            std::cerr << "Kaguya: Unknown GGUF value type: " << static_cast<uint32_t>(vtype) << "\n";
            return false;
    }
    return true;
}

std::string GgufLoader::read_gguf_string() {
    uint64_t len = 0;
    if (!reader_->read(&len, sizeof(len))) return "";

    std::string s(len, '\0');
    if (len > 0 && !reader_->read(s.data(), len)) return "";
    return s;
}

bool GgufLoader::align_to(size_t alignment) {
    size_t pos = reader_->tell();
    size_t aligned = (pos + alignment - 1) & ~(alignment - 1);
    if (aligned != pos) {
        return reader_->seek(aligned);
    }
    return true;
}

// ========================================================================
// Accessors
// ========================================================================

const GgufValue* GgufLoader::metadata(const std::string& key) const {
    auto it = model_.metadata.find(key);
    return (it != model_.metadata.end()) ? &it->second : nullptr;
}

std::optional<std::string> GgufLoader::metadata_string(const std::string& key) const {
    auto* v = metadata(key);
    if (v && v->is_string()) return v->as_string();
    return std::nullopt;
}

std::optional<int64_t> GgufLoader::metadata_int(const std::string& key) const {
    auto* v = metadata(key);
    if (v && v->is_int()) return v->as_int();
    return std::nullopt;
}

std::optional<double> GgufLoader::metadata_float(const std::string& key) const {
    auto* v = metadata(key);
    if (v && v->is_float()) return v->as_float();
    return std::nullopt;
}

const GgufTensorInfo* GgufLoader::tensor_info(const std::string& name) const {
    for (const auto& ti : model_.tensor_infos) {
        if (ti.name == name) return &ti;
    }
    return nullptr;
}

const void* GgufLoader::tensor_data(const GgufTensorInfo& info) const {
    if (!model_.tensor_data) return nullptr;
    return static_cast<const uint8_t*>(model_.tensor_data) + info.offset;
}

// ========================================================================
// Summary
// ========================================================================

void GgufLoader::print_summary() const {
    std::cout << "=== GGUF Model Summary ===\n";
    std::cout << "Version:    " << model_.version << "\n";
    std::cout << "Tensors:    " << model_.tensor_count << "\n";
    std::cout << "Metadata:   " << model_.metadata_kv_count << " keys\n";
    std::cout << "Data offset: " << model_.data_offset << "\n\n";

    // Architecture
    if (auto arch = metadata_string("general.architecture")) {
        std::cout << "Architecture: " << *arch << "\n";
    }
    if (auto name = metadata_string("general.name")) {
        std::cout << "Model name:   " << *name << "\n";
    }

    // Hyperparameters
    std::cout << "\n--- Hyperparameters ---\n";
    const char* int_keys[] = {
        "llama.context_length",  "llama.embedding_length",
        "llama.block_count",     "llama.attention.head_count",
        "llama.attention.head_count_kv", "llama.feed_forward_length",
        "llama.rope.freq_base",  "llama.attention.layer_norm_rms_epsilon",
    };
    for (const char* key : int_keys) {
        if (auto val = metadata_int(key)) {
            std::cout << "  " << key << " = " << *val << "\n";
        }
    }
    // Also check qwen2/mistral/phi3/gemma/deepseek2 prefixes
    const char* alt_prefixes[] = {"qwen2", "mistral", "phi3", "gemma", "deepseek2", "command-r"};
    for (const char* prefix : alt_prefixes) {
        std::string ctx_key = std::string(prefix) + ".context_length";
        if (auto val = metadata_int(ctx_key)) {
            std::cout << "  " << ctx_key << " = " << *val << "\n";
        }
    }

    // Quantization
    if (auto ft = metadata_int("general.file_type")) {
        std::cout << "\nFile type: " << *ft << "\n";
    }

    // Tensor types summary
    std::cout << "\n--- Tensor Types ---\n";
    std::unordered_map<GgmlType, int> type_counts;
    for (const auto& ti : model_.tensor_infos) {
        type_counts[ti.type]++;
    }
    for (const auto& [type, count] : type_counts) {
        std::cout << "  " << ggml_type_name(type) << ": " << count << " tensors\n";
    }

    // Total model size
    size_t total_bytes = 0;
    for (const auto& ti : model_.tensor_infos) {
        total_bytes += ggml_nbytes(ti);
    }
    std::cout << "\nTotal tensor data: " << (total_bytes / (1024.0 * 1024.0)) << " MiB\n";
}

} // namespace kaguya
