#include "kaguya/kernels/dispatcher.h"
#include "kaguya/kernels/gemm.h"
#include "kaguya/cpu_features.h"

namespace kaguya {

/// Select best kernel target at runtime based on CPU feature detection.
/// Follows the fallback chain: AMX → AVX-512 BF16/VNNI → AVX2+FMA → AVX2 → Scalar
///
/// IMPORTANT: AMX is currently disabled in KVM environments due to SIGSEGV
/// on LDTILECFG. The dispatcher will skip AMX and fall back to AVX-512.
KernelTarget select_kernel_target() {
    const auto& info = CpuFeatureDetector::get();
    const auto& f = info.flags;

    // AMX: only enable on bare metal where OS properly supports TILE state.
    // KVM environments report AMX via CPUID but LDTILECFG causes SIGSEGV.
    // Also require both AMX_BF16 and AMX_INT8 for a complete AMX GEMM.
    if (f.amx_bf16 && f.amx_int8 && f.amx_tile && f.os_amx) {
        return KernelTarget::AMX;
    }

    // AVX-512 BF16 + VNNI: PRIMARY strategy (confirmed working in KVM).
    // This provides the best GEMM performance for BF16 and INT8 workloads.
    if (f.avx512f && f.avx512bf16 && f.avx512vnni && f.os_avx512) {
        return KernelTarget::AVX512;
    }

    // AVX-512F only (without BF16/VNNI): still significantly better than
    // AVX2 for FP32 GEMM due to 512-bit registers and 32 ZMM registers.
    if (f.avx512f && f.os_avx512) {
        return KernelTarget::AVX512;
    }

    // AVX2 + FMA: good FP32 performance with 256-bit registers.
    // Note: In some KVM environments, FMA CPUID bit may not be set even
    // though the hardware supports it. The cpu_features.cpp deduces FMA
    // from AVX-512 availability, so this check should pass on AVX-512 CPUs.
    if (f.avx2 && f.fma && f.os_avx) {
        return KernelTarget::AVX2;
    }

    // AVX2 without FMA: still faster than scalar due to 256-bit SIMD.
    // The gemm_avx2() function provides a non-FMA fallback kernel.
    if (f.avx2 && f.os_avx) {
        return KernelTarget::AVX2;
    }

    return KernelTarget::Scalar;
}

} // namespace kaguya

namespace kaguya::kernels {

/// Unified GEMM dispatch: selects the best available FP32 GEMM kernel
/// based on CPU feature detection at runtime.
///
/// Dispatch priority:
///   1. AMX  → stub-only (falls through to AVX-512)
///   2. AVX-512 → gemm_avx512() with 4x16 micro-kernel
///   3. AVX2   → gemm_avx2() with 4x8 micro-kernel (FMA or non-FMA)
///   4. Scalar → gemm_scalar() with 64x64 cache-friendly tiling
void gemm_dispatch(const GemmParams& params) {
    const auto target = kaguya::select_kernel_target();

    switch (target) {
        case kaguya::KernelTarget::AMX:
            // AMX GEMM not yet implemented (bare-metal only).
            // Fall through to AVX-512.
            [[fallthrough]];
        case kaguya::KernelTarget::AVX512:
            gemm_avx512(params);
            break;
        case kaguya::KernelTarget::AVX2:
            gemm_avx2(params);
            break;
        case kaguya::KernelTarget::Scalar:
        default:
            gemm_scalar(params);
            break;
    }
}

} // namespace kaguya::kernels
