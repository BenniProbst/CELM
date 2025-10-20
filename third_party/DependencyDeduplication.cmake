# ==============================================================================
# BEP Dependency Deduplication System
# ==============================================================================
# Prevents code bloat by ensuring each dependency is loaded only once across
# the entire recursive project tree.
#
# Usage in CMakeLists.txt:
#   include(${CMAKE_SOURCE_DIR}/third_party/DependencyDeduplication.cmake)
#   bep_find_or_fetch_package(PackageName
#       GIT_REPOSITORY https://github.com/...
#       GIT_TAG v1.0.0
#   )
#
# Features:
# - Tracks all loaded packages globally
# - Warns if package is already loaded
# - Automatically forwards interface links to existing target
# - Works recursively across any depth of subprojects
# ==============================================================================

# Global property to track loaded packages across all subprojects
define_property(GLOBAL PROPERTY BEP_LOADED_PACKAGES
    BRIEF_DOCS "List of all loaded packages"
    FULL_DOCS "Tracks which packages have been loaded to prevent duplicates"
)

# Global property to track package versions
define_property(GLOBAL PROPERTY BEP_PACKAGE_VERSIONS
    BRIEF_DOCS "Versions of loaded packages"
    FULL_DOCS "Maps package names to their loaded versions"
)

# Global property to track package source locations
define_property(GLOBAL PROPERTY BEP_PACKAGE_SOURCES
    BRIEF_DOCS "Source locations of loaded packages"
    FULL_DOCS "Maps package names to their source directories"
)

# ==============================================================================
# bep_register_package
# ==============================================================================
# Internal function to register a package as loaded
function(bep_register_package PACKAGE_NAME VERSION SOURCE_DIR)
    get_property(loaded_packages GLOBAL PROPERTY BEP_LOADED_PACKAGES)
    list(APPEND loaded_packages ${PACKAGE_NAME})
    set_property(GLOBAL PROPERTY BEP_LOADED_PACKAGES "${loaded_packages}")

    set_property(GLOBAL PROPERTY BEP_PACKAGE_VERSION_${PACKAGE_NAME} "${VERSION}")
    set_property(GLOBAL PROPERTY BEP_PACKAGE_SOURCE_${PACKAGE_NAME} "${SOURCE_DIR}")

    message(STATUS "[BEP] Registered package: ${PACKAGE_NAME} v${VERSION}")
    message(STATUS "      Source: ${SOURCE_DIR}")
endfunction()

# ==============================================================================
# bep_is_package_loaded
# ==============================================================================
# Check if a package is already loaded
function(bep_is_package_loaded PACKAGE_NAME OUT_VAR)
    get_property(loaded_packages GLOBAL PROPERTY BEP_LOADED_PACKAGES)
    if("${PACKAGE_NAME}" IN_LIST loaded_packages)
        set(${OUT_VAR} TRUE PARENT_SCOPE)
    else()
        set(${OUT_VAR} FALSE PARENT_SCOPE)
    endif()
endfunction()

# ==============================================================================
# bep_get_package_info
# ==============================================================================
# Get information about a loaded package
function(bep_get_package_info PACKAGE_NAME OUT_VERSION OUT_SOURCE)
    get_property(version GLOBAL PROPERTY BEP_PACKAGE_VERSION_${PACKAGE_NAME})
    get_property(source GLOBAL PROPERTY BEP_PACKAGE_SOURCE_${PACKAGE_NAME})
    set(${OUT_VERSION} "${version}" PARENT_SCOPE)
    set(${OUT_SOURCE} "${source}" PARENT_SCOPE)
endfunction()

# ==============================================================================
# bep_find_or_fetch_package
# ==============================================================================
# Main function: find package locally or fetch via FetchContent
# Automatically handles deduplication across all subprojects
#
# Parameters:
#   PACKAGE_NAME      - Name of the package
#   GIT_REPOSITORY    - Git repository URL
#   GIT_TAG          - Git tag/branch/commit
#   GIT_SHALLOW      - (Optional) Use shallow clone
#   FIND_PACKAGE     - (Optional) Try find_package first
#   VERSION          - (Optional) Package version
#   TARGET_NAME      - (Optional) Custom target name if different from package
# ==============================================================================
function(bep_find_or_fetch_package PACKAGE_NAME)
    set(options GIT_SHALLOW FIND_PACKAGE)
    set(oneValueArgs GIT_REPOSITORY GIT_TAG VERSION TARGET_NAME)
    set(multiValueArgs)
    cmake_parse_arguments(ARG "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    # Default values
    if(NOT DEFINED ARG_VERSION)
        set(ARG_VERSION "unknown")
    endif()

    if(NOT DEFINED ARG_TARGET_NAME)
        set(ARG_TARGET_NAME ${PACKAGE_NAME})
    endif()

    # Check if package is already loaded
    bep_is_package_loaded(${PACKAGE_NAME} IS_LOADED)

    if(IS_LOADED)
        # Package already loaded - get info and warn
        bep_get_package_info(${PACKAGE_NAME} LOADED_VERSION LOADED_SOURCE)

        message(STATUS "[BEP] Package '${PACKAGE_NAME}' already loaded:")
        message(STATUS "      Loaded version: ${LOADED_VERSION}")
        message(STATUS "      Loaded from: ${LOADED_SOURCE}")
        message(STATUS "      Requested version: ${ARG_VERSION}")
        message(STATUS "      → Using existing package (preventing code bloat)")

        # Check version mismatch
        if(NOT "${ARG_VERSION}" STREQUAL "${LOADED_VERSION}" AND NOT "${ARG_VERSION}" STREQUAL "unknown")
            message(WARNING "[BEP] Version mismatch for ${PACKAGE_NAME}:")
            message(WARNING "      Requested: ${ARG_VERSION}")
            message(WARNING "      Using: ${LOADED_VERSION}")
            message(WARNING "      This may cause compatibility issues!")
        endif()

        return()
    endif()

    # Package not loaded yet - fetch it
    message(STATUS "[BEP] Loading new package: ${PACKAGE_NAME} v${ARG_VERSION}")

    # Try find_package first if requested
    if(ARG_FIND_PACKAGE)
        find_package(${PACKAGE_NAME} QUIET)
        if(${PACKAGE_NAME}_FOUND)
            message(STATUS "[BEP] Found ${PACKAGE_NAME} via find_package")
            bep_register_package(${PACKAGE_NAME} "${ARG_VERSION}" "system")
            return()
        endif()
    endif()

    # Fetch via FetchContent
    if(NOT DEFINED ARG_GIT_REPOSITORY)
        message(FATAL_ERROR "[BEP] GIT_REPOSITORY required for package ${PACKAGE_NAME}")
    endif()

    if(NOT DEFINED ARG_GIT_TAG)
        message(FATAL_ERROR "[BEP] GIT_TAG required for package ${PACKAGE_NAME}")
    endif()

    include(FetchContent)

    set(FETCH_ARGS
        GIT_REPOSITORY ${ARG_GIT_REPOSITORY}
        GIT_TAG ${ARG_GIT_TAG}
    )

    if(ARG_GIT_SHALLOW)
        list(APPEND FETCH_ARGS GIT_SHALLOW TRUE)
    endif()

    FetchContent_Declare(
        ${PACKAGE_NAME}
        ${FETCH_ARGS}
    )

    FetchContent_MakeAvailable(${PACKAGE_NAME})

    # Get source directory
    FetchContent_GetProperties(${PACKAGE_NAME})
    string(TOLOWER ${PACKAGE_NAME} PACKAGE_NAME_LOWER)
    set(SOURCE_DIR "${${PACKAGE_NAME_LOWER}_SOURCE_DIR}")

    # Register package
    bep_register_package(${PACKAGE_NAME} "${ARG_VERSION}" "${SOURCE_DIR}")
endfunction()

# ==============================================================================
# bep_print_dependency_tree
# ==============================================================================
# Debug function: Print all loaded packages
function(bep_print_dependency_tree)
    get_property(loaded_packages GLOBAL PROPERTY BEP_LOADED_PACKAGES)

    message(STATUS "")
    message(STATUS "========================================")
    message(STATUS "BEP Dependency Tree")
    message(STATUS "========================================")

    if(NOT loaded_packages)
        message(STATUS "No packages loaded")
    else()
        foreach(pkg ${loaded_packages})
            bep_get_package_info(${pkg} version source)
            message(STATUS "  ${pkg} v${version}")
            message(STATUS "    └─ ${source}")
        endforeach()
    endif()

    message(STATUS "========================================")
    message(STATUS "")
endfunction()
