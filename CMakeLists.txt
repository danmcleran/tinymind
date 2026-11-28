# Copyright (c) 2026 Dan McLeran
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# tinymind as a CMake package, and as an ESP-IDF component.
#
# This file only packages the library for other projects. tinymind's own
# tests, examples and gates are built by the Makefiles (`make check`), and
# nothing here builds or runs them.
#
# CMake (3.14 or later): add_subdirectory(), FetchContent, or
# find_package(tinymind) after `cmake --install`. Link tinymind::tinymind:
#
#   find_package(tinymind 19.1 REQUIRED)
#   target_link_libraries(app PRIVATE tinymind::tinymind)
#   target_compile_definitions(app PRIVATE TINYMIND_ENABLE_QUANTIZATION=1)
#
# The feature macros (TINYMIND_ENABLE_*, see cpp/include/tinymind_platform.hpp)
# all default to 0, a freestanding build. Set them on your own target, or turn
# on the TINYMIND_ENABLE_* options below to have tinymind::tinymind carry them
# to every target that links it. The fixed-point activation lookup tables
# (TINYMIND_USE_<FUNCTION>_<I>_<F>) live in cpp/lookupTables.cpp: list the
# ones you use in TINYMIND_LOOKUP_TABLES and link tinymind::lookup_tables,
# which compiles that file with them and passes them on, so the tables and
# the code that reads them always agree. With no tables listed, that target
# does not exist.
#
# ESP-IDF: put this directory in your project's components/ (or name it in
# EXTRA_COMPONENT_DIRS) and REQUIRES tinymind; see the README.

if(ESP_PLATFORM)
    # An ESP-IDF component. lookupTables.cpp compiles to nothing unless a
    # TINYMIND_USE_* flag is set, so it costs nothing when no table is used.
    # Set feature macros and table flags for the whole project with
    # idf_build_set_property(COMPILE_DEFINITIONS ... APPEND) before project().
    idf_component_register(SRCS "cpp/lookupTables.cpp"
                           INCLUDE_DIRS "cpp" "cpp/include")
    return()
endif()

cmake_minimum_required(VERSION 3.14)

project(tinymind
        VERSION 19.1.0
        DESCRIPTION "Header-only C++ neural network and Q-learning library for embedded systems"
        HOMEPAGE_URL "https://danmcleran.github.io/tinymind/"
        LANGUAGES CXX)

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

set(TINYMIND_FEATURES
    QUANTIZATION FLOAT STD HOSTED_IO OSTREAMS HOSTED_RAND FP16 INT16_ACCUM OPENMP
    SIMD_NEON SIMD_NEON_DOTPROD SIMD_NEON_FP16 SIMD_SVE SIMD_SVE2
    SIMD_HELIUM_MVE_I SIMD_HELIUM_MVE_F SIMD_ARM_DSP
    SIMD_AVX2 SIMD_AVX_VNNI SIMD_AVX512F SIMD_AVX512_VNNI)
foreach(feature IN LISTS TINYMIND_FEATURES)
    option(TINYMIND_ENABLE_${feature}
           "Define TINYMIND_ENABLE_${feature}=1 for every target that links tinymind::tinymind" OFF)
endforeach()
set(TINYMIND_LOOKUP_TABLES "" CACHE STRING
    "Lookup tables tinymind::lookup_tables compiles, as a list of TINYMIND_USE_* names (for example TINYMIND_USE_TANH_8_8)")

add_library(tinymind INTERFACE)
add_library(tinymind::tinymind ALIAS tinymind)
target_include_directories(tinymind INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/cpp>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/cpp/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/tinymind>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/tinymind/include>)
target_compile_features(tinymind INTERFACE cxx_std_14)
foreach(feature IN LISTS TINYMIND_FEATURES)
    if(TINYMIND_ENABLE_${feature})
        target_compile_definitions(tinymind INTERFACE TINYMIND_ENABLE_${feature}=1)
    endif()
endforeach()

set(tinymind_targets tinymind)
if(TINYMIND_LOOKUP_TABLES)
    add_library(tinymind_lookup_tables STATIC cpp/lookupTables.cpp)
    add_library(tinymind::lookup_tables ALIAS tinymind_lookup_tables)
    set_target_properties(tinymind_lookup_tables PROPERTIES EXPORT_NAME lookup_tables)
    target_link_libraries(tinymind_lookup_tables PUBLIC tinymind)
    foreach(table IN LISTS TINYMIND_LOOKUP_TABLES)
        target_compile_definitions(tinymind_lookup_tables PUBLIC ${table}=1)
    endforeach()
    list(APPEND tinymind_targets tinymind_lookup_tables)
endif()

install(DIRECTORY cpp/
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/tinymind
        FILES_MATCHING PATTERN "*.hpp" PATTERN "*.h")
install(TARGETS ${tinymind_targets}
        EXPORT tinymindTargets
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(EXPORT tinymindTargets
        NAMESPACE tinymind::
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/tinymind)
configure_package_config_file(cmake/tinymindConfig.cmake.in
                              ${CMAKE_CURRENT_BINARY_DIR}/tinymindConfig.cmake
                              INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/tinymind)
# Any 19.x satisfies a request for 19.y with y <= x; a new major version may
# change the API. Headers alone fit any architecture; a lookup-table archive
# does not.
if(TINYMIND_LOOKUP_TABLES)
    write_basic_package_version_file(${CMAKE_CURRENT_BINARY_DIR}/tinymindConfigVersion.cmake
                                     COMPATIBILITY SameMajorVersion)
else()
    write_basic_package_version_file(${CMAKE_CURRENT_BINARY_DIR}/tinymindConfigVersion.cmake
                                     COMPATIBILITY SameMajorVersion
                                     ARCH_INDEPENDENT)
endif()
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/tinymindConfig.cmake
              ${CMAKE_CURRENT_BINARY_DIR}/tinymindConfigVersion.cmake
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/tinymind)
