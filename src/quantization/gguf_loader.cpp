// Kaguya — gguf_loader
// Full GGUF format parser with mmap support

#include "kaguya/quantization/gguf_loader.h"
#include "kaguya/core/tensor.h"

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
        case GgmlType::Q4_0: case GgmlType::Q4_1:
        case GgmlType::Q5_0: case GgmlType::Q5_1:
        case GgmlType::Q8_0: case GgmlType::Q8_1:
        case GgmlType::IQ4_NL:
            return 32;
        case GgmlType::Q2_K: case GgmlType::Q3_K:
        case GgmlType::Q4_K: case GgmlType::Q5_K:
        case GgmlType::Q6_K: case GgmlType::Q8_K:
        case GgmlType::IQ2_XXS: case GgmlType::IQ2_XS:
        case GgmlType::IQ3_XXS: case GgmlType::IQ1_S:
        case GgmlType::IQ3_S: case GgmlType::IQ2_S:
        case GgmlType::IQ4_XS: case GgmlType::IQ1_M:
        case GgmlType::TQ1_0: case GgmlType::TQ2_0:
        case GgmlType::IQ1_M_NL:
            return 256;
        default:
            return 1;
    }
}

size_t ggml_type_size(GgmlType gt) {
    switch (gt) {
        case GgmlType::F32:      return 4;
        case GgmlType::F16:      return 2;
        case GgmlType::BF16:     return 2;
        case GgmlType::Q4_0:     return 18;
        case GgmlType::Q4_1:     return 20;
        case GgmlType::Q5_0:     return 22;
        case GgmlType::Q5_1:     return 24;
        case GgmlType::Q8_0:     return 34;
        case GgmlType::Q8_1:     return 36;
        case GgmlType::Q2_K:     return 64;
        case GgmlType::Q3_K:     return 110;
        case GgmlType::Q4_K:     return 144;
        case GgmlType::Q5_K:     return 176;
        case GgmlType::Q6_K:     return 210;
        case GgmlType::Q8_K:     return 292;
        case GgmlType::IQ2_XXS:  return 66;
        case GgmlType::IQ2_XS:   return 74;
        case GgmlType::IQ3_XXS:  return 98;
        case GgmlType::IQ1_S:    return 36;
        case GgmlType::IQ4_NL:   return 18;
        case GgmlType::IQ3_S:    return 110;
        case GgmlType::IQ2_S:    return 82;
        case GgmlType::IQ4_XS:   return 136;
        case GgmlType::IQ1_M:    return 56;
        case GgmlType::TQ1_0:    return 4;
        case GgmlType::TQ2_0:    return 64;
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
    size_t n_elements = 1;
    for (auto d : info.dims) n_elements *= d;
    int bs = ggml_block_size(info.type);
    size_t ts = ggml_type_size(info.type);
    if (bs <= 1) return n_elements * ts;
    size_t n_blocks = (n_elements + bs - 1) / bs;
    return n_blocks * ts;
}

// ========================================================================
// Reader — abstracts file I/O
// ========================================================================

#ifdef __linux__
class MmapReader final : public GgufLoader::Reader {
public:
    MmapReader() = default;
    ~MmapReader() override { close(); }

    bool open(const std::string& path) override {
        fd_ = ::open(path.c_str(), O_RDONLY);
        if (fd_ < 0) return false;
        struct stat st;
        if (fstat(fd_, &st) != 0) { ::close(fd_); fd_ = -1; return false; }
        file_size_ = st.st_size;
        data_ = static_cast<uint8_t*>(mmap(nullptr, file_size_, PROT_READ, MAP_PRIVATE, fd_, 0));
        if (data_ == MAP_FAILED) { ::close(fd_); fd_ = -1; data_ = nullptr; return false; }
        madvise(data_, file_size_, MADV_SEQUENTIAL);
        pos_ = 0;
        return true;
    }

    void close() override {
        if (data_) { munmap(data_, file_size_); data_ = nullptr; }
        if (fd_ >= 0) { ::close(fd_); fd_ = -1; }
        pos_ = 0; file_size_ = 0;
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

class StreamReader final : public GgufLoader::Reader {
public:
    StreamReader() = default;
    ~StreamReader() override { close(); }

    bool open(const std::string& path) override {
        std::ifstream file(path, std::ios::binary | std::ios::ate);
        if (!file.is_open()) return false;
        file_size_ = file.tellg();
        file.seekg(0, std::ios::beg);
        buffer_.resize(file_size_);
        file.read(reinterpret_cast<char*>(buffer_.data()), file_size_);
        if (!file.good()) return false;
        pos_ = 0;
        return true;
    }

    void close() override {
        buffer_.clear();
        buffer_.shrink_to_fit();
        pos_ = 0; file_size_ = 0;
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
        std::cerr << "Kaguya: Failed to open GGUF file: " << path << "\n";
        return false;
    }

    std::cout << "Kaguya: Loading GGUF file: " << path << "\n";

    if (!read_header()) return false;
    if (!read_metadata()) return false;
    if (!read_tensor_infos()) return false;

    // Align data section
    size_t alignment = (model_.version >= 3) ? 64 : 32;
    if (!align_to(alignment)) return false;
    model_.data_offset = reader_->tell();

    // Set tensor data pointer
    model_.tensor_data = const_cast<uint8_t*>(reader_->data_ptr()) + model_.data_offset;
    model_.tensor_data_size = reader_->file_size() - model_.data_offset;

    std::cout << "Kaguya: GGUF v" << model_.version
              << " | " << model_.tensor_count << " tensors"
              << " | " << model_.metadata_kv_count << " metadata keys"
              << " | data offset: " << model_.data_offset << "\n";
    return true;
}

bool GgufLoader::read_header() {
    constexpr uint32_t GGUF_MAGIC = 0x46475547;
    uint32_t magic;
    if (!reader_->read(&magic, sizeof(magic))) { std::cerr << "Kaguya: Failed to read magic\n"; return false; }
    if (magic != GGUF_MAGIC) { std::cerr << "Kaguya: Invalid magic: 0x" << std::hex << magic << "\n"; return false; }
    if (!reader_->read(&model_.version, sizeof(model_.version))) return false;
    if (model_.version < 2 || model_.version > 3) { std::cerr << "Kaguya: Unsupported version: " << model_.version << "\n"; return false; }
    if (!reader_->read(&model_.tensor_count, sizeof(model_.tensor_count))) return false;
    if (!reader_->read(&model_.metadata_kv_count, sizeof(model_.metadata_kv_count))) return false;
    return true;
}

bool GgufLoader::read_metadata() {
    for (uint64_t i = 0; i < model_.metadata_kv_count; ++i) {
        std::string key = read_gguf_string();
        uint32_t vtype_raw;
        if (!reader_->read(&vtype_raw, sizeof(vtype_raw))) return false;
        GgufValueType vtype = static_cast<GgufValueType>(vtype_raw);
        GgufValue val;
        if (!read_value(vtype, val)) return false;
        model_.metadata[key] = std::move(val);
    }
    return true;
}

bool GgufLoader::read_tensor_infos() {
    model_.tensor_infos.reserve(model_.tensor_count);
    for (uint64_t i = 0; i < model_.tensor_count; ++i) {
        GgufTensorInfo ti;
        ti.name = read_gguf_string();
        if (!reader_->read(&ti.n_dims, sizeof(ti.n_dims))) return false;
        ti.dims.resize(ti.n_dims);
        for (uint32_t d = 0; d < ti.n_dims; ++d) {
            if (!reader_->read(&ti.dims[d], sizeof(ti.dims[d]))) return false;
        }
        int32_t type_raw;
        if (!reader_->read(&type_raw, sizeof(type_raw))) return false;
        ti.type = static_cast<GgmlType>(type_raw);
        if (!reader_->read(&ti.offset, sizeof(ti.offset))) return false;
        model_.tensor_infos.push_back(std::move(ti));
    }
    return true;
}

bool GgufLoader::read_value(GgufValueType vtype, GgufValue& val) {
    switch (vtype) {
        case GgufValueType::UINT8:   { uint8_t v; if (!reader_->read(&v, 1)) return false; val.data = v; break; }
        case GgufValueType::INT8:    { int8_t v;  if (!reader_->read(&v, 1)) return false; val.data = v; break; }
        case GgufValueType::UINT16:  { uint16_t v; if (!reader_->read(&v, 2)) return false; val.data = v; break; }
        case GgufValueType::INT16:   { int16_t v;  if (!reader_->read(&v, 2)) return false; val.data = v; break; }
        case GgufValueType::UINT32:  { uint32_t v; if (!reader_->read(&v, 4)) return false; val.data = v; break; }
        case GgufValueType::INT32:   { int32_t v;  if (!reader_->read(&v, 4)) return false; val.data = v; break; }
        case GgufValueType::FLOAT32: { float v;    if (!reader_->read(&v, 4)) return false; val.data = v; break; }
        case GgufValueType::BOOL:    { uint8_t v;  if (!reader_->read(&v, 1)) return false; val.data = static_cast<bool>(v); break; }
        case GgufValueType::STRING:  { val.data = read_gguf_string(); break; }
        case GgufValueType::UINT64:  { uint64_t v; if (!reader_->read(&v, 8)) return false; val.data = v; break; }
        case GgufValueType::INT64:   { int64_t v;  if (!reader_->read(&v, 8)) return false; val.data = v; break; }
        case GgufValueType::FLOAT64: { double v;   if (!reader_->read(&v, 8)) return false; val.data = v; break; }
        case GgufValueType::ARRAY: {
            uint32_t elem_type_raw;
            if (!reader_->read(&elem_type_raw, sizeof(elem_type_raw))) return false;
            uint64_t count;
            if (!reader_->read(&count, sizeof(count))) return false;
            GgufValue::ArrayType arr;
            arr.reserve(count);
            for (uint64_t i = 0; i < count; ++i) {
                GgufValue elem;
                if (!read_value(static_cast<GgufValueType>(elem_type_raw), elem)) return false;
                arr.push_back(std::move(elem));
            }
            val.data = std::move(arr);
            break;
        }
        default: return false;
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
    if (aligned != pos) return reader_->seek(aligned);
    return true;
}

// Accessors
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

void GgufLoader::print_summary() const {
    std::cout << "=== GGUF Model Summary ===\n";
    std::cout << "Version:    " << model_.version << "\n";
    std::cout << "Tensors:    " << model_.tensor_count << "\n";
    std::cout << "Metadata:   " << model_.metadata_kv_count << " keys\n\n";
    if (auto arch = metadata_string("general.architecture"))
        std::cout << "Architecture: " << *arch << "\n";
    if (auto name = metadata_string("general.name"))
        std::cout << "Model name:   " << *name << "\n";

    size_t total_bytes = 0;
    std::unordered_map<GgmlType, int> type_counts;
    for (const auto& ti : model_.tensor_infos) {
        type_counts[ti.type]++;
        total_bytes += ggml_nbytes(ti);
    }
    std::cout << "\nTensor types:\n";
    for (const auto& [type, count] : type_counts)
        std::cout << "  " << ggml_type_name(type) << ": " << count << "\n";
    std::cout << "\nTotal tensor data: " << (total_bytes / (1024.0 * 1024.0)) << " MiB\n";
}

} // namespace kaguya
