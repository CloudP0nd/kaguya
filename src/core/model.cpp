#include "kaguya/model.h"
#include "kaguya/model_loader.h"
#include <stdexcept>
#include <algorithm>

namespace kaguya {

ModelArch arch_from_string(const std::string& name) {
    if (name == "llama" || name == "llama2" || name == "llama3") return ModelArch::LLAMA;
    if (name == "qwen2") return ModelArch::QWEN2;
    if (name == "mistral") return ModelArch::MISTRAL;
    if (name == "mixtral") return ModelArch::MIXTRAL;
    if (name == "phi2" || name == "phi3") return ModelArch::PHI3;
    if (name == "gemma" || name == "gemma2") return ModelArch::GEMMA;
    if (name == "falcon") return ModelArch::FALCON;
    if (name == "deepseek" || name == "deepseek2" || name == "deepseek3") return ModelArch::DEEPSEEK;
    if (name == "command-r") return ModelArch::COMMAND_R;
    return ModelArch::UNKNOWN;
}

const char* arch_to_string(ModelArch arch) {
    switch (arch) {
        case ModelArch::LLAMA:     return "llama";
        case ModelArch::QWEN2:     return "qwen2";
        case ModelArch::MISTRAL:   return "mistral";
        case ModelArch::MIXTRAL:   return "mixtral";
        case ModelArch::PHI3:      return "phi3";
        case ModelArch::GEMMA:     return "gemma";
        case ModelArch::FALCON:    return "falcon";
        case ModelArch::DEEPSEEK:  return "deepseek";
        case ModelArch::COMMAND_R: return "command-r";
        default:                   return "unknown";
    }
}

Model::Model() = default;

bool Model::load(const std::string& path) {
    // Use ModelLoader for actual loading
    ModelLoader loader;
    if (!loader.load(path)) {
        return false;
    }
    // Transfer loaded data
    set_weights(ModelWeights{loader.model().weights()});
    return true;
}

const GgufTensorInfo* Model::tensor_info(const std::string& name) const {
    (void)name;
    // Model no longer stores GgufTensorInfo directly
    // Use ModelLoader for tensor info access
    return nullptr;
}

} // namespace kaguya
