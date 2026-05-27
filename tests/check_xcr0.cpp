/// Check OS XCR0 support for AMX
#include <cstdint>
#include <iostream>

int main() {
    uint32_t eax, edx;
    __asm__ volatile ("xgetbv" : "=a"(eax), "=d"(edx) : "c"(0));
    uint64_t xcr0 = ((uint64_t)edx << 32) | eax;

    std::cout << "XCR0 = 0x" << std::hex << xcr0 << std::dec << "\n";
    std::cout << "XMM/YMM (bits 1:2): " << ((xcr0 >> 1) & 3) << " (expect 3)\n";
    std::cout << "OPMASK  (bit 5):     " << ((xcr0 >> 5) & 1) << "\n";
    std::cout << "ZMM_hi (bit 6):      " << ((xcr0 >> 6) & 1) << "\n";
    std::cout << "Hi16_ZMM (bit 7):    " << ((xcr0 >> 7) & 1) << "\n";
    std::cout << "AMX_TILECFG (bit 17): " << ((xcr0 >> 17) & 1) << "\n";
    std::cout << "AMX_TILEDATA (bit 18): " << ((xcr0 >> 18) & 1) << "\n";

    bool avx_ok = (xcr0 & 0x6) == 0x6;
    bool avx512_ok = (xcr0 & 0xE0) == 0xE0;
    bool amx_ok = (xcr0 & (0x3ULL << 17)) == (0x3ULL << 17);

    std::cout << "\nOS AVX support:     " << (avx_ok ? "YES" : "NO") << "\n";
    std::cout << "OS AVX-512 support: " << (avx512_ok ? "YES" : "NO") << "\n";
    std::cout << "OS AMX support:     " << (amx_ok ? "YES" : "NO") << "\n";

    return 0;
}
