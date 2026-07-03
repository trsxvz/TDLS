# TDLS is header-only: this file compiles nothing for the library itself.
# It declares the tdls::tdls INTERFACE target consumed through
# add_subdirectory()/FetchContent (include path + C++17 floor), installs
# the headers and a relocatable package config so a system install and
# find_package(tdls) work too, and orchestrates the tests.
cmake_minimum_required(VERSION 3.21)

project(tdls VERSION 0.1.0 LANGUAGES CXX)

# The version declared here and the one in include/tdls/core/version.hpp
# must never drift apart.
file(READ "${CMAKE_CURRENT_SOURCE_DIR}/include/tdls/core/version.hpp" TDLS_VERSION_HEADER)
string(REGEX MATCH "TDLS_VERSION_STRING \"([0-9.]+)\"" _ "${TDLS_VERSION_HEADER}")
if(NOT CMAKE_MATCH_1 STREQUAL PROJECT_VERSION)
    message(FATAL_ERROR "version mismatch: version.hpp says '${CMAKE_MATCH_1}', "
                        "project() says '${PROJECT_VERSION}'")
endif()

include(GNUInstallDirs)

add_library(tdls INTERFACE)
add_library(tdls::tdls ALIAS tdls)
target_include_directories(tdls INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
target_compile_features(tdls INTERFACE cxx_std_17)

# Tests are built when tdls is the top-level project, and stay out of the
# way when tdls is consumed by another build.
if(PROJECT_IS_TOP_LEVEL)
    option(TDLS_BUILD_TESTS "Build the tdls test suite" ON)
else()
    option(TDLS_BUILD_TESTS "Build the tdls test suite" OFF)
endif()

if(PROJECT_IS_TOP_LEVEL)
    option(TDLS_BUILD_EXAMPLES "Build the tdls examples" ON)
else()
    option(TDLS_BUILD_EXAMPLES "Build the tdls examples" OFF)
endif()

if(TDLS_BUILD_TESTS OR TDLS_BUILD_EXAMPLES)
    enable_testing()
endif()
if(TDLS_BUILD_TESTS)
    add_subdirectory(tests)
endif()
if(TDLS_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

# Installation and packaging. Header-only, so this installs the headers,
# the INTERFACE target (its include path and C++17 floor) and a
# relocatable package config, so that `cmake --install` / `make install`
# and a downstream find_package(tdls) all work. On by default only when
# tdls is the top-level project, so a parent add_subdirectory() or
# FetchContent build is never forced to install it.
option(TDLS_INSTALL "Generate the tdls install and package-config rules"
       ${PROJECT_IS_TOP_LEVEL})

if(TDLS_INSTALL)
    include(CMakePackageConfigHelpers)

    install(TARGETS tdls EXPORT tdlsTargets)
    install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/include/tdls"
            DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}")

    set(TDLS_INSTALL_CMAKEDIR "${CMAKE_INSTALL_LIBDIR}/cmake/tdls")
    install(EXPORT tdlsTargets
            FILE tdlsTargets.cmake
            NAMESPACE tdls::
            DESTINATION "${TDLS_INSTALL_CMAKEDIR}")

    configure_package_config_file(
        "${CMAKE_CURRENT_SOURCE_DIR}/cmake/tdlsConfig.cmake.in"
        "${CMAKE_CURRENT_BINARY_DIR}/tdlsConfig.cmake"
        INSTALL_DESTINATION "${TDLS_INSTALL_CMAKEDIR}")
    write_basic_package_version_file(
        "${CMAKE_CURRENT_BINARY_DIR}/tdlsConfigVersion.cmake"
        VERSION "${PROJECT_VERSION}"
        COMPATIBILITY SameMajorVersion
        ARCH_INDEPENDENT)
    install(FILES
            "${CMAKE_CURRENT_BINARY_DIR}/tdlsConfig.cmake"
            "${CMAKE_CURRENT_BINARY_DIR}/tdlsConfigVersion.cmake"
            DESTINATION "${TDLS_INSTALL_CMAKEDIR}")
endif()
