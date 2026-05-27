/// AMX smoke test — with XCR0 AMX initialization attempt
#include <cstdint>
#include <cstring>
#include <cmath>
#include <iostream>

#if defined(__x86_64__)
#include <immintrin.h>

/// Read current XCR0
static uint64_t read_xcr0() {
    uint32_t eax, edx;
    __asm__ volatile ("xgetbv" : "=a"(eax), "=d"(edx) : "c"(0));
    return ((uint64_t)edx << 32) | eax;
}

/// Try to enable AMX in XCR0 (requires CPL0, may not work in userspace)
static bool try_enable_amx_xcr0() {
    uint64_t xcr0 = read_xcr0();
    std::cout << "Current XCR0: 0x" << std::hex << xcr0 << std::dec << "\n";

    // Check if AMX bits (17, 18) are already set
    uint64_t amx_bits = (0x3ULL << 17);
    if ((xcr0 & amx_bits) == amx_bits) {
        std::cout << "AMX bits already set in XCR0\n";
        return true;
    }

    std::cout << "AMX bits NOT set in XCR0 — this is unusual\n";
    std::cout << "XCR0 bit 17 (TILECFG): " << ((xcr0 >> 17) & 1) << "\n";
    std::cout << "XCR0 bit 18 (TILEDATA): " << ((xcr0 >> 18) & 1) << "\n";

    // Try to set them (will likely fail in userspace)
    uint64_t new_xcr0 = xcr0 | amx_bits;
    __asm__ volatile (
        "xsetbv"
        :
        : "a" ((uint32_t)new_xcr0), "d" ((uint32_t)(new_xcr0 >> 32)), "c" (0)
    );

    xcr0 = read_xcr0();
    std::cout << "After XSETBV, XCR0: 0x" << std::hex << xcr0 << std::dec << "\n";
    return (xcr0 & amx_bits) == amx_bits;
}

static bool test_amx_bf16() {
    // First check XCR0
    try_enable_amx_xcr0();

    // --- Tile configuration ---
    alignas(64) uint8_t cfg_buf[64];
    memset(cfg_buf, 0, 64);

    cfg_buf[0] = 1;  // palette_id = 1

    // colsb[8] at offset 16
    *reinterpret_cast<uint16_t*>(&cfg_buf[16]) = 64;  // TMM0: 64 bytes/row
    *reinterpret_cast<uint16_t*>(&cfg_buf[18]) = 64;  // TMM1: 64 bytes/row
    *reinterpret_cast<uint16_t*>(&cfg_buf[20]) = 64;  // TMM2: 64 bytes/row

    // rows[8] at offset 32
    cfg_buf[32] = 16;  // TMM0
    cfg_buf[33] = 16;  // TMM1
    cfg_buf[34] = 16;  // TMM2

    // --- Input data ---
    alignas(64) uint16_t a_data[16 * 32];
    alignas(64) uint16_t b_data[16 * 32];
    alignas(64) float    c_data[16 * 16];

    for (int i = 0; i < 16 * 32; ++i) {
        a_data[i] = 0x3f80;
        b_data[i] = 0x3f80;
    }
    memset(c_data, 0, sizeof(c_data));

    std::cout << "Calling _tile_loadconfig...\n" << std::flush;
    _tile_loadconfig(cfg_buf);
    std::cout << "_tile_loadconfig returned\n" << std::flush;

    _tile_zero(2);
    _tile_loadd(0, a_data, 64);
    _tile_loadd(1, b_data, 64);
    _tile_dpbf16ps(2, 1, 0);
    _tile_stored(2, c_data, 64);
    _tile_release();

    bool ok = true;
    for (int i = 0; i < 16; ++i) {
        for (int j = 0; j < 16; ++j) {
            float val = c_data[i * 16 + j];
            if (std::abs(val - 32.0f) > 0.5f) {
                std::cerr << "Mismatch at [" << i << "][" << j << "]: "
                          << val << "\n";
                ok = false;
            }
        }
    }

    return ok;
}

int main() {
    std::cout << "=== Kaguya AMX Smoke Test ===\n\n";

    if (test_amx_bf16()) {
        std::cout << "PASS!\n";
        return 0;
    } else {
        std::cout << "FAIL\n";
        return 1;
    }
}

#else
int main() {
    std::cout << "AMX not available.\n";
    return 1;
}
#endif
