# Core library
add_subdirectory(core)
add_subdirectory(io)

# Main library target
add_library(celm_lib INTERFACE)
target_link_libraries(celm_lib INTERFACE
    core_lib
    io_lib
)
