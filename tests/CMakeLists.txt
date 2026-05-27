include(FetchContent)
FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG        v1.14.0
)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

add_executable(kaguya_tests
    test_cpu_features.cpp
    test_gguf_loader.cpp
)

target_link_libraries(kaguya_tests PRIVATE
    kaguya_core kaguya_kernels kaguya_quant
    kaguya_sampling kaguya_memory kaguya_model
    GTest::gtest_main
)

include(GoogleTest)
gtest_discover_tests(kaguya_tests)
