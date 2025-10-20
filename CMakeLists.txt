cmake_minimum_required(VERSION 3.20)
project(CELM VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Options
option(BUILD_TESTS "Build tests" ON)
option(ENABLE_ASAN "Enable AddressSanitizer for memory debugging" OFF)
option(CELM_ENABLE_TEST_INSTRUMENTATION
       "Enable detailed test diagnostics (adds overhead)"
       OFF)

# AddressSanitizer configuration
if(ENABLE_ASAN)
    if(MSVC)
        add_compile_options(/fsanitize=address)
        add_link_options(/fsanitize=address)
    else()
        add_compile_options(-fsanitize=address -fno-omit-frame-pointer -g)
        add_link_options(-fsanitize=address)
    endif()
    message(STATUS "AddressSanitizer enabled")
endif()

# Test Instrumentation Configuration
if(CELM_ENABLE_TEST_INSTRUMENTATION)
    add_compile_definitions(CELM_ENABLE_TEST_INSTRUMENTATION)
    message(STATUS "CELM: Test instrumentation ENABLED")
else()
    message(STATUS "CELM: Test instrumentation DISABLED")
endif()

# Automatically enable in Debug builds
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(CELM_ENABLE_TEST_INSTRUMENTATION ON CACHE BOOL "" FORCE)
    add_compile_definitions(CELM_ENABLE_TEST_INSTRUMENTATION)
endif()

# Find Google Test
if(BUILD_TESTS)
    find_package(GTest QUIET)
    if(NOT GTest_FOUND)
        include(FetchContent)
        FetchContent_Declare(
            googletest
            GIT_REPOSITORY https://github.com/google/googletest.git
            GIT_TAG v1.15.2  # Use stable version
            GIT_SHALLOW TRUE
        )
        # For Windows: Prevent overriding the parent project's compiler/linker settings
        set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
        FetchContent_MakeAvailable(googletest)
        # Include Google Test's module for gtest_discover_tests
        include(GoogleTest)
    endif()
endif()

# Add source subdirectories
add_subdirectory(src)

# Build main executable
add_executable(CELM main.cpp)
target_link_libraries(CELM PRIVATE celm_lib)

# Set default working directory for debugging
set_target_properties(CELM PROPERTIES
    VS_DEBUGGER_WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
)

# Tests
if(BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

# Installation configuration
include(GNUInstallDirs)

# Install binary to bin/ directory
install(TARGETS CELM
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    COMPONENT runtime
)

# Optional: Install documentation
install(FILES README.md LICENSE
    DESTINATION ${CMAKE_INSTALL_DOCDIR}
    COMPONENT documentation
    OPTIONAL
)
