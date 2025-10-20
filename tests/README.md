# CELM Test Suite

This directory contains the test suite for CELM using Google Test.

## Running Tests

### Using CLion/IntelliJ

1. **Reload CMake Project**: Tools → CMake → Reload CMake Project
2. **Select Debug Configuration**: Choose "Debug" from the configuration dropdown
3. **Run Tests**:
   - Run all tests: Select `run_tests` target and click Run
   - Run specific test: Select test target (e.g., `test_example_gtest`) and click Run

### Using Command Line

#### Configure and Build (Debug)
```bash
# Windows
configure.bat --build-type=Debug
build.bat

# Linux/macOS
./configure --build-type=Debug
make
```

#### Run Tests
```bash
# Windows
build.bat test

# Linux/macOS
make test

# Or run CTest directly
cd build
ctest --output-on-failure
```

### Using CMake Presets

```bash
# Configure
cmake --preset debug

# Build
cmake --build --preset debug

# Test
ctest --preset debug
```

## Test Structure

- `test_main.cpp` - Main entry point for all tests
- `core/` - Core functionality tests
- `io/` - Input/Output tests
- `test_instrumentation/` - Test framework utilities

## Writing Tests

Example test file:

```cpp
#include <gtest/gtest.h>

TEST(MyTest, BasicTest) {
    EXPECT_EQ(1 + 1, 2);
}
```

Add to `CMakeLists.txt`:
```cmake
add_instrumented_test(test_mytest_gtest test_mytest.cpp)
```

## Test Instrumentation

Set `CELM_ENABLE_TEST_INSTRUMENTATION=ON` for detailed test diagnostics.
This is automatically enabled in Debug builds.

## Troubleshooting

### Tests not showing up in CLion
- Reload CMake: Tools → CMake → Reload CMake Project
- Check CMake output for errors
- Verify BUILD_TESTS is ON in CMakeLists.txt

### Tests fail to build
- Ensure Google Test is properly fetched (check cmake-build-debug/_deps/)
- Verify C++23 compiler is available
- Check CMake version >= 3.20
