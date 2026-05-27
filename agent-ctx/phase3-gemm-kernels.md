# Phase 3: GEMM Compute Kernels — Work Record

## Task
Implement Phase 3 compute kernels for Kaguya CPU inference engine: GEMM header, all GEMM variants (scalar/AVX2/AVX-512 BF16/VNNI/FP32/AMX stub), dispatcher, and tests.

## Files Created/Modified

### Created/Updated Header
- `include/kaguya/kernels/gemm.h` — GemmParams, GemmParamsBF16, GemmParamsVNNI structs with comprehensive documentation; function declarations for all GEMM variants

### Implementation Files
- `src/kernels/gemm_scalar.cpp` — FP32 scalar GEMM with 64x64 cache-friendly micro-tiling
- `src/kernels/gemm_avx2.cpp` — AVX2+FMA FP32 GEMM with 4x8 micro-kernel; added non-FMA AVX2 fallback kernel for KVM environments
- `src/kernels/gemm_avx512.cpp` — AVX-512 BF16 GEMM (_mm512_dpbf16_ps), VNNI INT8 GEMM (_mm512_dpbusd_epi32), FP32 GEMM (4x16 micro-kernel)
- `src/kernels/gemm_amx.cpp` — Stub with detailed AMX KVM limitation documentation
- `src/kernels/dispatcher.cpp` — select_kernel_target() and gemm_dispatch() with full fallback chain

### Bug Fix
- `src/core/cpu_features.cpp` — Fixed FMA detection in KVM: logically deduce FMA+AVX2+AVX+os_avx from AVX-512 availability (KVM hypervisors sometimes fail to expose FMA CPUID bit even though hardware supports it)

### Tests
- `tests/unit/test_gemm.cpp` — 39 GEMM tests covering scalar correctness, AVX2 vs scalar, AVX-512 FP32 vs scalar, BF16 precision, VNNI exactness, edge cases (GEMV, K=1, odd dimensions, alpha/beta, empty matrices, non-multiple-of-4 K for VNNI, odd K for BF16)

## Build & Test Results
- **Compiler**: GCC 14.2.0, C++23, `-march=native -O3`
- **CPU**: Intel Xeon 6982P-C (Granite Rapids-D) with AVX-512 BF16/VNNI confirmed working
- **All 61 tests PASS** (0 failures, 0 skips)
  - Previously 2 AVX2 tests were SKIPPED due to FMA not detected; fixed by logical deduction in CPU feature detection
- **22 non-GEMM tests** still pass (CpuFeatures, GgufLoader, etc.)

## Key Design Decisions
1. AVX-512 BF16 is primary kernel (confirmed working in KVM)
2. AVX2 has dual-path: FMA kernel when available, non-FMA fallback with separate mul+add
3. AMX stub-only due to LDTILECFG SIGSEGV in KVM — dispatcher falls through to AVX-512
4. BF16 GEMM fallback: converts BF16→FP32 and uses scalar when AVX-512 BF16 unavailable
5. VNNI GEMM fallback: scalar INT8 when AVX-512 VNNI unavailable
