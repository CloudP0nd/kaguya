#pragma once

#include <string>
#include <functional>

namespace kaguya {

enum class KernelTarget {
    AMX,
    AVX512,
    AVX2,
    Scalar,
};

/// Select best kernel target at runtime
KernelTarget select_kernel_target();

} // namespace kaguya
