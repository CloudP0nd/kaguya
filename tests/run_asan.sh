#!/bin/bash
# Run tests with AddressSanitizer and UndefinedBehaviorSanitizer
# Uses Debug build type which enables -fsanitize=address,undefined
set -e

cd "$(dirname "$0")/../build"

cmake .. -DCMAKE_BUILD_TYPE=Debug
make -j$(nproc)
./kaguya_tests
