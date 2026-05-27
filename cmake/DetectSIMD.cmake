# DetectSIMD.cmake — Auto-detect and configure SIMD instruction sets

message(STATUS "=== Kaguya SIMD Detection ===")

include(CheckCXXSourceRuns)

set(SIMD_COMPILE_FLAGS "")
set(KAGUYA_HAS_AVX2 0)
set(KAGUYA_HAS_AVX512 0)
set(KAGUYA_HAS_AVX512_VNNI 0)
set(KAGUYA_HAS_AVX512_BF16 0)
set(KAGUYA_HAS_AVX512_FP16 0)
set(KAGUYA_HAS_FMA 0)

# AVX2
check_cxx_source_runs("
    #include <immintrin.h>
    int main() { __m256i a = _mm256_setzero_si256(); return 0; }
" HAVE_AVX2)
if(HAVE_AVX2)
    set(KAGUYA_HAS_AVX2 1)
    string(APPEND SIMD_COMPILE_FLAGS " -mavx2")
    message(STATUS "  AVX2:      YES")
else()
    message(STATUS "  AVX2:      NO")
endif()

# FMA
check_cxx_source_runs("
    #include <immintrin.h>
    int main() { __m256 a = _mm256_fmadd_ps(_mm256_set1_ps(1.0f), _mm256_set1_ps(2.0f), _mm256_set1_ps(3.0f)); return 0; }
" HAVE_FMA)
if(HAVE_FMA)
    set(KAGUYA_HAS_FMA 1)
    string(APPEND SIMD_COMPILE_FLAGS " -mfma")
    message(STATUS "  FMA:       YES")
else()
    message(STATUS "  FMA:       NO")
endif()

# AVX-512 Foundation
check_cxx_source_runs("
    #include <immintrin.h>
    int main() { __m512i a = _mm512_setzero_si512(); return 0; }
" HAVE_AVX512F)
if(HAVE_AVX512F)
    set(KAGUYA_HAS_AVX512 1)
    string(APPEND SIMD_COMPILE_FLAGS " -mavx512f")
    message(STATUS "  AVX-512F:  YES")

    # AVX-512 VNNI
    check_cxx_source_runs("
        #include <immintrin.h>
        int main() { __m512i a = _mm512_setzero_si512(); __m512i b = _mm512_setzero_si512(); __m512i c = _mm512_setzero_si512();
        __m512i r = _mm512_dpbusd_epi32(a, b, c); return 0; }
    " HAVE_AVX512_VNNI)
    if(HAVE_AVX512_VNNI)
        set(KAGUYA_HAS_AVX512_VNNI 1)
        string(APPEND SIMD_COMPILE_FLAGS " -mavx512vnni")
        message(STATUS "  AVX-512 VNNI: YES")
    else()
        message(STATUS "  AVX-512 VNNI: NO")
    endif()

    # AVX-512 BF16
    check_cxx_source_runs("
        #include <immintrin.h>
        int main() { __m512i a = _mm512_setzero_si512(); __m512 b = _mm512_cvtneps_pbh(_mm512_set1_ps(1.0f)); return 0; }
    " HAVE_AVX512_BF16)
    if(HAVE_AVX512_BF16)
        set(KAGUYA_HAS_AVX512_BF16 1)
        string(APPEND SIMD_COMPILE_FLAGS " -mavx512bf16")
        message(STATUS "  AVX-512 BF16: YES")
    else()
        message(STATUS "  AVX-512 BF16: NO")
    endif()

    # AVX-512 FP16
    check_cxx_source_runs("
        #include <immintrin.h>
        int main() { __m512h a = _mm512_setzero_ph(); return 0; }
    " HAVE_AVX512_FP16)
    if(HAVE_AVX512_FP16)
        set(KAGUYA_HAS_AVX512_FP16 1)
        string(APPEND SIMD_COMPILE_FLAGS " -mavx512fp16")
        message(STATUS "  AVX-512 FP16: YES")
    else()
        message(STATUS "  AVX-512 FP16: NO")
    endif()
else()
    message(STATUS "  AVX-512F:  NO")
endif()

# Configure preprocessor definitions
set(SIMD_DEFINITIONS "")
if(KAGUYA_HAS_AVX2)
    string(APPEND SIMD_DEFINITIONS " -DKAGUYA_AVX2=1")
endif()
if(KAGUYA_HAS_AVX512)
    string(APPEND SIMD_DEFINITIONS " -DKAGUYA_AVX512=1")
endif()
if(KAGUYA_HAS_AVX512_VNNI)
    string(APPEND SIMD_DEFINITIONS " -DKAGUYA_AVX512_VNNI=1")
endif()
if(KAGUYA_HAS_AVX512_BF16)
    string(APPEND SIMD_DEFINITIONS " -DKAGUYA_AVV512_BF16=1")
endif()
if(KAGUYA_HAS_AVX512_FP16)
    string(APPEND SIMD_DEFINITIONS " -DKAGUYA_AVX512_FP16=1")
endif()

message(STATUS "  Compile flags: ${SIMD_COMPILE_FLAGS}")
message(STATUS "=== SIMD Detection Complete ===")
