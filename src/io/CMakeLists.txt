add_library(io_lib INTERFACE)

target_include_directories(io_lib INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/include
)

target_link_libraries(io_lib INTERFACE
    core_lib
)
