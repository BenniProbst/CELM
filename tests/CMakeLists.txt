# Test Instrumentation Library
add_library(test_instrumentation INTERFACE)
target_include_directories(test_instrumentation INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}
)

# Store tests directory for use in subdirectories
set(TESTS_DIR ${CMAKE_CURRENT_SOURCE_DIR} CACHE INTERNAL "Tests directory")

# Helper function to add instrumented tests
function(add_instrumented_test test_name test_source)
    add_executable(${test_name} ${test_source} ${TESTS_DIR}/test_main.cpp)
    target_link_libraries(${test_name} PRIVATE
        core_lib
        io_lib
        test_instrumentation
        GTest::gtest
    )
    gtest_discover_tests(${test_name} DISCOVERY_TIMEOUT 30)
endfunction()

# Test subdirectories
add_subdirectory(core)
add_subdirectory(io)
