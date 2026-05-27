#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace kaguya {

/// CPU cache level descriptor
struct CacheInfo {
    int level;          ///< L1, L2, L3
    int size_kb;        ///< Size in KiB
    int line_size;      ///< Cache line size in bytes
    int associativity;  ///< Ways of associativity
    int num_sharing;    ///< Number of logical CPUs sharing this cache
};

/// NUMA node descriptor
struct NumaNode {
    int node_id;
    std::vector<int> cpu_list;  ///< Logical CPUs in this node
    uint64_t memory_total_mb;
    uint64_t memory_free_mb;
};

/// Comprehensive CPU feature flags
struct CpuFeatureFlags {
    // x87 / SSE
    bool fpu       = false;
    bool sse       = false;
    bool sse2      = false;
    bool sse3      = false;
    bool ssse3     = false;
    bool sse4_1    = false;
    bool sse4_2    = false;

    // AVX family
    bool avx       = false;
    bool avx2      = false;
    bool fma       = false;
    bool f16c      = false;

    // AVX-512 foundation
    bool avx512f       = false;
    bool avx512dq      = false;
    bool avx512ifma    = false;
    bool avx512cd      = false;
    bool avx512bw      = false;
    bool avx512vl      = false;
    bool avx512vbmi    = false;
    bool avx512vbmi2   = false;
    bool avx512vnni    = false;
    bool avx512bf16    = false;
    bool avx512fp16    = false;
    bool avx512bitalg  = false;
    bool avx512vpopcntdq = false;

    // AVX-VNNI (non-512 VNNI)
    bool avx_vnni   = false;

    // AMX
    bool amx_tile   = false;
    bool amx_int8   = false;
    bool amx_bf16   = false;

    // Other
    bool popcnt     = false;
    bool bmi1       = false;
    bool bmi2       = false;
    bool aes        = false;
    bool sha        = false;
    bool rdrand     = false;
    bool rdseed     = false;
    bool clflushopt = false;
    bool clwb       = false;
    bool movbe      = false;
    bool serialize  = false;

    // OS support for extended state
    bool os_avx     = false;  ///< OS supports AVX (YMM state)
    bool os_avx512  = false;  ///< OS supports AVX-512 (ZMM state)
    bool os_amx     = false;  ///< OS supports AMX (TILE state)
};

/// Detected CPU information
struct CpuInfo {
    std::string vendor;          ///< "GenuineIntel" or "AuthenticAMD"
    std::string brand_string;    ///< Full CPU brand string
    int family                   = 0;
    int model                    = 0;
    int stepping                 = 0;
    int num_logical_cpus         = 0;
    int num_physical_cpus        = 0;
    int threads_per_core         = 0;
    uint64_t frequency_hz        = 0;  ///< Nominal frequency

    CpuFeatureFlags flags;
    std::vector<CacheInfo> caches;
    std::vector<NumaNode> numa_nodes;
};

/// CPU feature detector — queries CPUID and OS support at runtime
class CpuFeatureDetector {
public:
    /// Detect all CPU features (call once at startup)
    static CpuInfo detect();

    /// Get the cached detection result
    static const CpuInfo& get();

    /// Get the optimal kernel target based on detected features
    static std::string best_kernel_target();

    /// Print detected CPU info (for logging/debug)
    static std::string summary();

private:
    static CpuInfo cached_info_;
    static bool detected_;

    // CPUID wrappers
    static void cpuid(uint32_t leaf, uint32_t subleaf,
                      uint32_t& eax, uint32_t& ebx, uint32_t& ecx, uint32_t& edx);

    // OS XSAVE support check
    static bool check_os_xsave();
    static bool check_os_ymm();
    static bool check_os_zmm();
    static bool check_os_tile();

    // Cache info via CPUID leaf 4
    static std::vector<CacheInfo> detect_caches();

    // NUMA topology (Linux only)
    static std::vector<NumaNode> detect_numa();

    // Topology via CPUID leaf 0xB
    static void detect_topology(CpuInfo& info);
};

} // namespace kaguya
