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

# An ESP-IDF project that uses this checkout as the tinymind component, for
# CI: it runs the shared self-test. The repository root is the component (its
# CMakeLists.txt calls idf_component_register under ESP-IDF); the component
# takes the name of its directory, so the checkout must be in a directory
# named tinymind. A project of your own would put tinymind in components/ or
# list it in idf_component.yml.

cmake_minimum_required(VERSION 3.16)

set(EXTRA_COMPONENT_DIRS ${CMAKE_CURRENT_LIST_DIR}/../..)
include($ENV{IDF_PATH}/tools/cmake/project.cmake)

# For every component, tinymind's own source (the lookup tables) included,
# so they and the application agree.
idf_build_set_property(COMPILE_DEFINITIONS "TINYMIND_ENABLE_QUANTIZATION=1" APPEND)
idf_build_set_property(COMPILE_DEFINITIONS "TINYMIND_USE_TANH_8_8=1" APPEND)

project(tinymind_esp_idf_selftest)
