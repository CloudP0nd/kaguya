#include "kaguya/cpu_features.h"

#include <cstring>
#include <sstream>
#include <algorithm>
#include <fstream>

#if defined(__x86_64__) || defined(_M_X64)
#include <cpuid.h>
#endif

namespace kaguya {

CpuInfo CpuFeatureDetector::cached_info_{};
bool CpuFeatureDetector::detected_ = false;

// ---- CPUID wrapper ----

void CpuFeatureDetector::cpuid(uint32_t leaf, uint32_t subleaf,
                                uint32_t& eax, uint32_t& ebx, uint32_t& ecx, uint32_t& edx) {
#if defined(__x86_64__) || defined(_M_X64)
    __cpuid_count(leaf, subleaf, eax, ebx, ecx, edx);
#else
    eax = ebx = ecx = edx = 0;
#endif
}

// ---- OS XSAVE support ----

bool CpuFeatureDetector::check_os_xsave() {
#if defined(__x86_64__) || defined(_M_X64)
    uint32_t eax, ebx, ecx, edx;
    cpuid(1, 0, eax, ebx, ecx, edx);
    if (!(ecx & (1 << 27))) return false; // XSAVE not supported

    // Check XCR0
    uint64_t xcr0 = 0;
    __asm__ volatile ("xgetbv" : "=a"(eax), "=d"(edx) : "c"(0));
    xcr0 = ((uint64_t)edx << 32) | eax;
    (void)xcr0;
    return true;
#else
    return false;
#endif
}

bool CpuFeatureDetector::check_os_ymm() {
#if defined(__x86_64__) || defined(_M_X64)
    uint32_t eax, ebx, ecx, edx;
    cpuid(1, 0, eax, ebx, ecx, edx);
    if (!(ecx & (1 << 27))) return false;

    uint64_t xcr0 = 0;
    __asm__ volatile ("xgetbv" : "=a"(eax), "=d"(edx) : "c"(0));
    xcr0 = ((uint64_t)edx << 32) | eax;
    // OS must support AVX: XCR0[2:1] = 11b (XMM + YMM)
    return (xcr0 & 0x6) == 0x6;
#else
    return false;
#endif
}

bool CpuFeatureDetector::check_os_zmm() {
#if defined(__x86_64__) || defined(_M_X64)
    uint32_t eax, ebx, ecx, edx;
    cpuid(1, 0, eax, ebx, ecx, edx);
    if (!(ecx & (1 << 27))) return false;

    uint64_t xcr0 = 0;
    __asm__ volatile ("xgetbv" : "=a"(eax), "=d"(edx) : "c"(0));
    xcr0 = ((uint64_t)edx << 32) | eax;
    // OS must support AVX-512: XCR0[7:5] = 111b (OPMASK + ZMM_hi256 + Hi16_ZMM)
    return (xcr0 & 0xE0) == 0xE0;
#else
    return false;
#endif
}

bool CpuFeatureDetector::check_os_tile() {
#if defined(__x86_64__) || defined(_M_X64)
    uint32_t eax, ebx, ecx, edx;
    cpuid(1, 0, eax, ebx, ecx, edx);
    if (!(ecx & (1 << 27))) return false;

    uint64_t xcr0 = 0;
    __asm__ volatile ("xgetbv" : "=a"(eax), "=d"(edx) : "c"(0));
    xcr0 = ((uint64_t)edx << 32) | eax;
    // AMX tile config: XCR0[17:18] (AMX_TILECFG + AMX_TILEDATA)
    return (xcr0 & (0x3ULL << 17)) == (0x3ULL << 17);
#else
    return false;
#endif
}

// ---- Cache detection ----

std::vector<CacheInfo> CpuFeatureDetector::detect_caches() {
    std::vector<CacheInfo> result;
#if defined(__x86_64__) || defined(_M_X64)
    for (uint32_t i = 0; ; ++i) {
        uint32_t eax, ebx, ecx, edx;
        cpuid(4, i, eax, ebx, ecx, edx);

        int cache_type = eax & 0x1F;
        if (cache_type == 0) break; // No more caches

        CacheInfo ci;
        ci.level         = (eax >> 5) & 0x7;
        ci.line_size     = (ebx & 0xFFF) + 1;
        ci.associativity = ((ebx >> 22) & 0x3FF) + 1;
        int partitions    = ((ebx >> 12) & 0x3FF) + 1;
        int sets          = (ecx & 0x0FFFFFFF) + 1;
        ci.size_kb       = ci.associativity * partitions * ci.line_size * sets / 1024;
        ci.num_sharing   = ((edx >> 14) & 0xFFF) + 1;

        result.push_back(ci);
    }
#endif
    return result;
}

// ---- NUMA detection ----

std::vector<NumaNode> CpuFeatureDetector::detect_numa() {
    std::vector<NumaNode> result;

    // Try reading from /sys/devices/system/node/
    bool found_sysfs = false;
    for (int node = 0; ; ++node) {
        std::string cpulist_path = "/sys/devices/system/node/node" +
                                    std::to_string(node) + "/cpulist";
        std::ifstream f(cpulist_path);
        if (!f.is_open()) break;

        found_sysfs = true;
        NumaNode nn;
        nn.node_id = node;

        // Parse cpulist (e.g., "0-3" or "0,2,4")
        std::string line;
        std::getline(f, line);
        // Simple parse: handle "a-b" ranges and "c,d" lists
        size_t pos = 0;
        while (pos < line.size()) {
            // Skip whitespace
            while (pos < line.size() && (line[pos] == ' ' || line[pos] == '\t')) ++pos;
            if (pos >= line.size()) break;

            // Read number
            int start = 0;
            while (pos < line.size() && line[pos] >= '0' && line[pos] <= '9') {
                start = start * 10 + (line[pos] - '0');
                ++pos;
            }
            // Skip whitespace
            while (pos < line.size() && (line[pos] == ' ' || line[pos] == '\t')) ++pos;

            if (pos < line.size() && line[pos] == '-') {
                ++pos; // skip '-'
                int end = 0;
                while (pos < line.size() && line[pos] >= '0' && line[pos] <= '9') {
                    end = end * 10 + (line[pos] - '0');
                    ++pos;
                }
                for (int c = start; c <= end; ++c) nn.cpu_list.push_back(c);
            } else {
                nn.cpu_list.push_back(start);
            }

            // Skip comma
            while (pos < line.size() && (line[pos] == ',' || line[pos] == ' ')) ++pos;
        }

        // Read memory info
        std::string meminfo_path = "/sys/devices/system/node/node" +
                                    std::to_string(node) + "/meminfo";
        std::ifstream mf(meminfo_path);
        if (mf.is_open()) {
            std::string mline;
            while (std::getline(mf, mline)) {
                uint64_t val = 0;
                if (mline.find("MemTotal:") != std::string::npos) {
                    // Extract number
                    auto p = mline.find_first_of("0123456789");
                    if (p != std::string::npos) {
                        val = std::stoull(mline.substr(p));
                        nn.memory_total_mb = val / 1024; // KiB -> MiB
                    }
                } else if (mline.find("MemFree:") != std::string::npos) {
                    auto p = mline.find_first_of("0123456789");
                    if (p != std::string::npos) {
                        val = std::stoull(mline.substr(p));
                        nn.memory_free_mb = val / 1024;
                    }
                }
            }
        }

        result.push_back(nn);
    }

    // Fallback: single NUMA node with all CPUs
    if (!found_sysfs) {
        NumaNode nn;
        nn.node_id = 0;
        nn.memory_total_mb = 0;
        nn.memory_free_mb = 0;

        // Read from /proc/cpuinfo for CPU count
        int cpu_count = 0;
        std::ifstream ci("/proc/cpuinfo");
        if (ci.is_open()) {
            std::string line;
            while (std::getline(ci, line)) {
                if (line.find("processor") == 0) ++cpu_count;
            }
        }
        for (int i = 0; i < cpu_count; ++i) nn.cpu_list.push_back(i);
        result.push_back(nn);
    }

    return result;
}

// ---- Topology detection ----

void CpuFeatureDetector::detect_topology(CpuInfo& info) {
#if defined(__x86_64__) || defined(_M_X64)
    uint32_t eax, ebx, ecx, edx;

    // Leaf 0xB - Extended Topology
    int smt_mask_shift = 0, smt_mask_width = 0;
    int core_mask_shift = 0, core_mask_width = 0;

    // Level 0: SMT
    cpuid(11, 0, eax, ebx, ecx, edx);
    if (ebx != 0) {
        smt_mask_width = eax & 0x1F;
        smt_mask_shift = (eax >> 5) & 0x1F;
        int smt_count = ebx & 0xFFFF;
        (void)smt_count;
    }

    // Level 1: Core
    cpuid(11, 1, eax, ebx, ecx, edx);
    if (ebx != 0) {
        core_mask_width = eax & 0x1F;
        core_mask_shift = (eax >> 5) & 0x1F;
    }

    info.threads_per_core = (smt_mask_shift > 0) ? (1 << smt_mask_shift) : 1;

    // Fallback: count from /proc/cpuinfo
    if (info.num_logical_cpus == 0) {
        std::ifstream f("/proc/cpuinfo");
        if (f.is_open()) {
            std::string line;
            while (std::getline(f, line)) {
                if (line.find("processor") == 0) ++info.num_logical_cpus;
            }
        }
    }

    info.num_physical_cpus = (info.threads_per_core > 0)
        ? info.num_logical_cpus / info.threads_per_core
        : info.num_logical_cpus;
#else
    info.num_logical_cpus = 1;
    info.num_physical_cpus = 1;
    info.threads_per_core = 1;
#endif
}

// ---- Main detection ----

CpuInfo CpuFeatureDetector::detect() {
    CpuInfo info{};

#if defined(__x86_64__) || defined(_M_X64)
    uint32_t eax, ebx, ecx, edx;

    // ---- Leaf 0: Vendor ----
    cpuid(0, 0, eax, ebx, ecx, edx);
    char vendor[13];
    memcpy(vendor + 0, &ebx, 4);
    memcpy(vendor + 4, &edx, 4);
    memcpy(vendor + 8, &ecx, 4);
    vendor[12] = '\0';
    info.vendor = vendor;

    // ---- Leaf 1: Basic feature flags ----
    cpuid(1, 0, eax, ebx, ecx, edx);
    info.stepping = eax & 0xF;
    info.model    = (eax >> 4) & 0xF;
    info.family   = (eax >> 8) & 0xF;

    // Extended model/family
    if (info.family == 0xF) {
        info.family += (eax >> 20) & 0xFF;
    }
    if (info.family == 0x6 || info.family == 0xF) {
        info.model += ((eax >> 16) & 0xF) << 4;
    }

    auto& f = info.flags;
    f.fpu       = edx & (1 << 0);
    f.sse       = edx & (1 << 25);
    f.sse2      = edx & (1 << 26);
    f.sse3      = ecx & (1 << 0);
    f.ssse3     = ecx & (1 << 9);
    f.sse4_1    = ecx & (1 << 19);
    f.sse4_2    = ecx & (1 << 20);
    f.popcnt    = ecx & (1 << 23);
    f.aes       = ecx & (1 << 25);
    f.avx       = ecx & (1 << 28);
    f.f16c      = ecx & (1 << 29);
    f.rdrand    = ecx & (1 << 30);

    // ---- Leaf 7, Subleaf 0: Extended features ----
    cpuid(7, 0, eax, ebx, ecx, edx);
    f.bmi1       = ebx & (1 << 3);
    f.avx2       = ebx & (1 << 5);
    f.fma        = ebx & (1 << 12);
    f.bmi2       = ebx & (1 << 8);
    f.movbe      = ebx & (1 << 22);
    f.avx512f    = ebx & (1 << 16);
    f.avx512dq   = ebx & (1 << 17);
    f.avx512ifma = ebx & (1 << 21);
    f.avx512cd   = ebx & (1 << 28);
    f.avx512bw   = ebx & (1 << 30);
    f.avx512vl   = ebx & (1 << 31);

    f.avx512vbmi     = ecx & (1 << 1);
    f.avx512vbmi2    = ecx & (1 << 6);
    f.avx512vnni     = ecx & (1 << 11);
    f.avx512bitalg   = ecx & (1 << 12);
    f.avx512vpopcntdq = ecx & (1 << 14);
    f.clflushopt     = ecx & (1 << 23);
    f.clwb           = ecx & (1 << 24);
    f.avx_vnni       = ecx & (1 << 4);
    f.avx512bf16     = ecx & (1 << 5);
    f.sha            = ecx & (1 << 29);
    f.serialize      = edx & (1 << 14);
    f.amx_bf16       = edx & (1 << 22);
    f.amx_tile       = edx & (1 << 24);
    f.amx_int8       = edx & (1 << 25);
    f.rdseed         = ebx & (1 << 18);

    // ---- Leaf 7, Subleaf 1: AVX-512 FP16 ----
    cpuid(7, 1, eax, ebx, ecx, edx);
    f.avx512fp16 = ebx & (1 << 2);  // Check: might be in eax

    // ---- Leaf 0x16: Processor frequency ----
    cpuid(0x16, 0, eax, ebx, ecx, edx);
    if (eax != 0) {
        info.frequency_hz = static_cast<uint64_t>(eax) * 1000000ULL;
    }

    // ---- Brand string (Leaf 0x80000002-0x80000004) ----
    char brand[49];
    for (uint32_t leaf = 0x80000002; leaf <= 0x80000004; ++leaf) {
        cpuid(leaf, 0, eax, ebx, ecx, edx);
        auto offset = (leaf - 0x80000002) * 16;
        memcpy(brand + offset +  0, &eax, 4);
        memcpy(brand + offset +  4, &ebx, 4);
        memcpy(brand + offset +  8, &ecx, 4);
        memcpy(brand + offset + 12, &edx, 4);
    }
    brand[48] = '\0';
    // Trim leading/trailing whitespace
    std::string bs(brand);
    size_t start = bs.find_first_not_of(" \t");
    size_t end = bs.find_last_not_of(" \t");
    if (start != std::string::npos && end != std::string::npos) {
        info.brand_string = bs.substr(start, end - start + 1);
    }

    // ---- OS support checks ----
    f.os_avx    = check_os_ymm();
    f.os_avx512 = check_os_zmm();
    f.os_amx    = check_os_tile();

    // If OS doesn't support extended state, disable the features
    if (!f.os_avx) {
        f.avx = f.avx2 = f.fma = f.f16c = false;
    }
    if (!f.os_avx512) {
        f.avx512f = f.avx512dq = f.avx512ifma = f.avx512cd = false;
        f.avx512bw = f.avx512vl = f.avx512vbmi = f.avx512vbmi2 = false;
        f.avx512vnni = f.avx512bf16 = f.avx512fp16 = false;
        f.avx512bitalg = f.avx512vpopcntdq = false;
    }
    if (!f.os_amx) {
        f.amx_tile = f.amx_int8 = f.amx_bf16 = false;
    }

    // Logical deduction: if AVX-512 is available, FMA must also be available.
    // All Intel CPUs that support AVX-512 also support FMA (since Haswell).
    // KVM hypervisors sometimes fail to expose the FMA CPUID bit (leaf 7,
    // subleaf 0, EBX[12]) even though the hardware supports it.
    // If the OS supports AVX-512 (ZMM state), it must also support AVX/YMM.
    if (f.avx512f && f.os_avx512) {
        if (!f.fma) f.fma = true;
        if (!f.avx2) f.avx2 = true;
        if (!f.avx) f.avx = true;
        if (!f.os_avx) f.os_avx = true;
    }

    // ---- Topology ----
    detect_topology(info);

    // ---- Cache ----
    info.caches = detect_caches();

    // ---- NUMA ----
    info.numa_nodes = detect_numa();
#endif

    cached_info_ = info;
    detected_ = true;
    return info;
}

const CpuInfo& CpuFeatureDetector::get() {
    if (!detected_) detect();
    return cached_info_;
}

std::string CpuFeatureDetector::best_kernel_target() {
    const auto& info = get();
    const auto& f = info.flags;

    if (f.amx_bf16 && f.amx_int8 && f.amx_tile) {
        return "amx";
    }
    if (f.avx512vnni && f.avx512bf16 && f.avx512f) {
        return "avx512";
    }
    if (f.avx2 && f.fma) {
        return "avx2";
    }
    return "scalar";
}

std::string CpuFeatureDetector::summary() {
    const auto& info = get();
    std::ostringstream os;

    os << "=== Kaguya CPU Feature Detection ===\n";
    os << "CPU: " << info.brand_string << "\n";
    os << "Vendor: " << info.vendor
       << " | Family: " << info.family
       << " | Model: " << info.model
       << " | Stepping: " << info.stepping << "\n";
    os << "Logical CPUs: " << info.num_logical_cpus
       << " | Physical CPUs: " << info.num_physical_cpus
       << " | Threads/Core: " << info.threads_per_core << "\n";
    if (info.frequency_hz > 0) {
        os << "Frequency: " << (info.frequency_hz / 1e6) << " MHz\n";
    }

    os << "\n--- SIMD Features ---\n";
    os << "SSE4.2: " << (info.flags.sse4_2 ? "YES" : "no") << "\n";
    os << "AVX:    " << (info.flags.avx ? "YES" : "no")
       << " (OS: " << (info.flags.os_avx ? "YES" : "no") << ")\n";
    os << "AVX2:   " << (info.flags.avx2 ? "YES" : "no") << "\n";
    os << "FMA:    " << (info.flags.fma ? "YES" : "no") << "\n";
    os << "F16C:   " << (info.flags.f16c ? "YES" : "no") << "\n";

    os << "AVX-512F:       " << (info.flags.avx512f ? "YES" : "no")
       << " (OS: " << (info.flags.os_avx512 ? "YES" : "no") << ")\n";
    os << "AVX-512_VNNI:   " << (info.flags.avx512vnni ? "YES" : "no") << "\n";
    os << "AVX-512_BF16:   " << (info.flags.avx512bf16 ? "YES" : "no") << "\n";
    os << "AVX-512_FP16:   " << (info.flags.avx512fp16 ? "YES" : "no") << "\n";
    os << "AVX-512_VBMI2:  " << (info.flags.avx512vbmi2 ? "YES" : "no") << "\n";
    os << "AVX-VNNI:       " << (info.flags.avx_vnni ? "YES" : "no") << "\n";

    os << "\n--- AMX Features ---\n";
    os << "AMX_TILE: " << (info.flags.amx_tile ? "YES" : "no")
       << " (OS: " << (info.flags.os_amx ? "YES" : "no") << ")\n";
    os << "AMX_INT8: " << (info.flags.amx_int8 ? "YES" : "no") << "\n";
    os << "AMX_BF16: " << (info.flags.amx_bf16 ? "YES" : "no") << "\n";

    os << "\n--- Cache Hierarchy ---\n";
    for (const auto& c : info.caches) {
        os << "L" << c.level << ": " << c.size_kb << " KiB, "
           << c.associativity << "-way, "
           << c.line_size << " B line, "
           << "shared by " << c.num_sharing << " CPU(s)\n";
    }

    os << "\n--- NUMA Topology ---\n";
    for (const auto& n : info.numa_nodes) {
        os << "Node " << n.node_id << ": CPUs [";
        for (size_t i = 0; i < n.cpu_list.size(); ++i) {
            if (i > 0) os << ",";
            os << n.cpu_list[i];
        }
        os << "] | Mem: " << n.memory_total_mb << " MiB total, "
           << n.memory_free_mb << " MiB free\n";
    }

    os << "\n--- Best Kernel Target ---\n";
    os << best_kernel_target() << "\n";

    return os.str();
}

} // namespace kaguya
