cmake_minimum_required(VERSION 3.25)
project(kaguya VERSION 0.1.0 LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# Optimization flags for Release
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -mtune=native -flto=auto")
elseif(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g3 -fsanitize=address,undefined")
endif()

# Warning flags
add_compile_options(-Wall -Wextra -Wpedantic -Wno-unused-parameter)

# ---- Platform detection ----

# Check for NUMA support
include(CheckIncludeFileCXX)
check_include_file_cxx("numa.h" KAGUYA_HAS_NUMA)
if(KAGUYA_HAS_NUMA)
    message(STATUS "NUMA support: YES")
    add_compile_definitions(KAGUYA_HAS_NUMA)
else()
    message(STATUS "NUMA support: NO (install libnuma-dev for NUMA support)")
endif()

# ---- Dependencies ----

# Google Test
include(FetchContent)
FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG        v1.14.0
)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

# Google Benchmark
FetchContent_Declare(
    googlebenchmark
    GIT_REPOSITORY https://github.com/google/benchmark.git
    GIT_TAG        v1.8.3
)
set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googlebenchmark)

# ---- Library ----

set(KAGUYA_SOURCES
    src/core/cpu_features.cpp
    src/core/tensor.cpp
    src/core/model.cpp
    src/core/context.cpp
    src/loaders/gguf_loader.cpp
    src/loaders/model_loader.cpp
    src/loaders/gguf_parser.cpp
    src/loaders/gguf_tensor.cpp
    src/kernels/dispatcher.cpp
    src/kernels/gemm_amx.cpp
    src/kernels/gemm_avx512.cpp
    src/kernels/gemm_avx2.cpp
    src/kernels/gemm_scalar.cpp
    src/kernels/quantize.cpp
    src/kernels/softmax.cpp
    src/kernels/norm.cpp
    src/kernels/rope.cpp
    src/kernels/activation.cpp
    src/runtime/memory_manager.cpp
    src/runtime/thread_pool.cpp
    src/runtime/thread_affinity.cpp
    src/inference/pipeline.cpp
    src/inference/kv_cache.cpp
    src/inference/attention.cpp
    src/inference/batch.cpp
    src/sampling/sampler.cpp
)

add_library(kaguya STATIC ${KAGUYA_SOURCES})
target_include_directories(kaguya PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_BINARY_DIR}
)
target_link_libraries(kaguya PRIVATE pthread)

# ---- CLI ----

add_executable(kaguya-cli src/cli/main.cpp)
target_link_libraries(kaguya-cli PRIVATE kaguya)

# ---- Tests ----

enable_testing()

add_executable(kaguya_tests
    tests/unit/test_cpu_features.cpp
    tests/unit/test_gguf_parser.cpp
    tests/unit/test_gemm.cpp
    tests/unit/test_quantize.cpp
    tests/unit/test_kv_cache.cpp
    tests/integration/test_inference.cpp
)
target_link_libraries(kaguya_tests PRIVATE kaguya GTest::gtest_main)
include(GoogleTest)
gtest_discover_tests(kaguya_tests)

# ---- Benchmarks ----

add_executable(kaguya_bench
    benchmarks/bench_gemm.cpp
    benchmarks/bench_quantize.cpp
    benchmarks/bench_inference.cpp
)
target_link_libraries(kaguya_bench PRIVATE kaguya benchmark::benchmark)
