// Kaguya — hparams implementation

#include "kaguya/model/hparams.h"

namespace kaguya {

ModelArch arch_from_string(const std::string& name) {
    if (name == "llama" || name == "llama2" || name == "llama3" || name == "llama4")
        return ModelArch::LLAMA;
    if (name == "qwen2" || name == "qwen2moe" || name == "qwen3")
        return ModelArch::QWEN2;
    if (name == "mistral" || name == "mistral-nemo")
        return ModelArch::MISTRAL;
    if (name == "mixtral")
        return ModelArch::MIXTRAL;
    if (name == "phi3" || name == "phi2")
        return ModelArch::PHI3;
    if (name == "gemma" || name == "gemma2")
        return ModelArch::GEMMA;
    if (name == "falcon")
        return ModelArch::FALCON;
    if (name == "deepseek2" || name == "deepseek")
        return ModelArch::DEEPSEEK;
    if (name == "command-r")
        return ModelArch::COMMAND_R;
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

} // namespace kaguya
