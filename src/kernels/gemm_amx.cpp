#include "kaguya/kernels/gemm.h"
#include "kaguya/cpu_features.h"

namespace kaguya::kernels {

// ============================================================================
// AMX GEMM kernel — STUB ONLY
//
// DESIGN NOTE: AMX (Advanced Matrix Extensions) provides tile-based matrix
// operations (TDPBF16PS for BF16, TDPBSSD for INT8) that can achieve
// significantly higher throughput than AVX-512 for matrix multiply.
//
// HOWEVER: AMX is NOT usable in KVM virtual machines at this time.
//
// KVM Limitation Details:
// -----------------------
// 1. CPUID reports AMX support (AMX_TILE, AMX_INT8, AMX_BF16 bits are set)
// 2. XCR0[17:18] (AMX_TILECFG + AMX_TILEDATA) may be reported as available
// 3. BUT: LDTILECFG instruction causes SIGSEGV in KVM guests
// 4. Root cause: KVM hypervisor does not properly virtualize AMX TILE state
//    - The AMX tile configuration register (TILECFG) is not context-switched
//    - Guest attempts to execute LDTILECFG trap and fail
// 5. This affects all major KVM versions as of 2024
//
// Safe Execution Environments:
// ----------------------------
// - Bare metal (no virtualization)
// - VMware with proper AMX passthrough (requires ESXi 8.0+)
// - Future KVM versions once AMX virtualization is implemented
//
// Fallback Strategy:
// ------------------
// The dispatcher will skip AMX and fall back to AVX-512 BF16/VNNI,
// which provides excellent performance for both BF16 and INT8 GEMM.
//
// The fallback chain is: AMX → AVX-512 → AVX2 → Scalar
// Since AMX is stub-only, the effective chain is: AVX-512 → AVX2 → Scalar
//
// When AMX virtualization becomes available in KVM, this stub should be
// replaced with actual AMX GEMM implementations using:
//   - _tile_loadconfig() / _tile_storeconfig() for tile management
//   - _tile_loadd() for loading tile data
//   - _tile_dpbssd() / _tile_dpbf16ps() for INT8/BF16 dot products
//   - _tile_stored() for storing results
// ============================================================================

} // namespace kaguya::kernels
