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

# tinymind's Zephyr module (zephyr/module.yml). The feature macros and lookup
# table flags are defined for the whole build, so tinymind's one source file
# and the application always agree.

if(CONFIG_TINYMIND)
    set(TINYMIND_DIR ${ZEPHYR_CURRENT_MODULE_DIR})
    zephyr_include_directories(${TINYMIND_DIR}/cpp ${TINYMIND_DIR}/cpp/include)

    if(CONFIG_TINYMIND_QUANTIZATION)
        zephyr_compile_definitions(TINYMIND_ENABLE_QUANTIZATION=1)
    endif()
    if(CONFIG_TINYMIND_FLOAT)
        zephyr_compile_definitions(TINYMIND_ENABLE_FLOAT=1)
    endif()
    if(CONFIG_TINYMIND_STD)
        zephyr_compile_definitions(TINYMIND_ENABLE_STD=1)
    endif()
    if(CONFIG_TINYMIND_SIMD_ARM_DSP)
        zephyr_compile_definitions(TINYMIND_ENABLE_SIMD_ARM_DSP=1)
    endif()

    if(NOT "${CONFIG_TINYMIND_LOOKUP_TABLES}" STREQUAL "")
        string(REPLACE " " ";" tinymind_tables "${CONFIG_TINYMIND_LOOKUP_TABLES}")
        foreach(table IN LISTS tinymind_tables)
            if(NOT table STREQUAL "")
            endif()
        endforeach()
        zephyr_library_named(tinymind)
        zephyr_library_sources(${TINYMIND_DIR}/cpp/lookupTables.cpp)
    endif()
endif()
