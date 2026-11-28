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

# A project that uses tinymind the two ways CMake offers, for CI:
#
#   -DTINYMIND_FROM=subdirectory  add_subdirectory() on this checkout (the
#                                 path FetchContent_MakeAvailable takes too)
#   -DTINYMIND_FROM=package       find_package(tinymind) on an installed copy
#                                 (installed with
#                                 -DTINYMIND_LOOKUP_TABLES=TINYMIND_USE_TANH_8_8)
#
# It runs the shared self-test (packaging/selftest) and checks that the
# lookup table tinymind::lookup_tables was asked for is compiled and linked.

cmake_minimum_required(VERSION 3.14)
project(tinymind_consumer LANGUAGES CXX)

set(TINYMIND_FROM "subdirectory" CACHE STRING "subdirectory or package")
if(TINYMIND_FROM STREQUAL "subdirectory")
    set(TINYMIND_LOOKUP_TABLES "TINYMIND_USE_TANH_8_8" CACHE STRING "" FORCE)
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../.. tinymind)
elseif(TINYMIND_FROM STREQUAL "package")
    find_package(tinymind 19.1 REQUIRED)
else()
    message(FATAL_ERROR "TINYMIND_FROM must be subdirectory or package, not ${TINYMIND_FROM}")
endif()

add_executable(tinymind_consumer main.cpp)
target_include_directories(tinymind_consumer PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../selftest)
target_compile_definitions(tinymind_consumer PRIVATE TINYMIND_ENABLE_QUANTIZATION=1)
target_compile_options(tinymind_consumer PRIVATE -Wall -Wextra -Werror -Wpedantic)
target_link_libraries(tinymind_consumer PRIVATE tinymind::tinymind tinymind::lookup_tables)

enable_testing()
add_test(NAME tinymind_selftest COMMAND tinymind_consumer)
