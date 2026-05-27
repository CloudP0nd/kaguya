cmake_minimum_required(VERSION 3.20)
project(kaguya
    VERSION 0.1.0
    DESCRIPTION "Kaguya — High-performance CPU LLM inference engine"
    LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Compiler flags
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Platform-specific SIMD detection
include(CheckCXXSourceCompiles)
include(cmake/DetectSIMD.cmake)

# Third-party dependencies
# ggml (for GGUF model loading compatibility)
# add_subdirectory(third_party/ggml)

# Kaguya core library
add_library(kaguya_core
    src/core/engine.cpp
    src/core/tensor.cpp
    src/core/context.cpp
    src/core/cpu_features.cpp
    src/memory/memory_manager.cpp
    src/memory/thread_pool.cpp
    src/memory/thread_affinity.cpp
)

target_include_directories(kaguya_core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_compile_options(kaguya_core PRIVATE ${SIMD_COMPILE_FLAGS})

# Kaguya kernels library
add_library(kaguya_kernels
    src/kernels/transformer.cpp
    src/kernels/attention.cpp
    src/kernels/ffn.cpp
    src/kernels/rmsnorm.cpp
    src/kernels/rope.cpp
    src/kernels/mul_mat.cpp
)

target_link_libraries(kaguya_kernels PUBLIC kaguya_core)

# Kaguya quantization library
add_library(kaguya_quant
    src/quantization/gguf_loader.cpp
    src/quantization/quantizer.cpp
)

target_link_libraries(kaguya_quant PUBLIC kaguya_core)

# Kaguya sampling library
add_library(kaguya_sampling
    src/sampling/sampler.cpp
    src/sampling/top_k.cpp
    src/sampling/top_p.cpp
    src/sampling/temperature.cpp
    src/sampling/repetition_penalty.cpp
)

target_link_libraries(kaguya_sampling PUBLIC kaguya_core)

# Kaguya memory management
add_library(kaguya_memory
    src/memory/buf_manager.cpp
    src/memory/kv_cache.cpp
)

target_link_libraries(kaguya_memory PUBLIC kaguya_core)

# Kaguya model loader
add_library(kaguya_model
    src/model/loader.cpp
    src/model/hparams.cpp
    src/model/weights.cpp
)

target_link_libraries(kaguya_model PUBLIC kaguya_core kaguya_quant)

# CLI executable
add_executable(kaguya-cli src/main.cpp)
target_link_libraries(kaguya-cli
    PRIVATE kaguya_core kaguya_kernels kaguya_quant
    kaguya_sampling kaguya_memory kaguya_model
)

# Testing
option(KAGUYA_BUILD_TESTS "Build tests" ON)
if(KAGUYA_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

# Benchmarks
option(KAGUYA_BUILD_BENCHMARKS "Build benchmarks" ON)
if(KAGUYA_BUILD_BENCHMARKS)
    add_subdirectory(benchmarks)
endif()
