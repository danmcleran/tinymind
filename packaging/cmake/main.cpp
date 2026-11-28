/**
* Copyright (c) 2026 Dan McLeran
*
* Permission is hereby granted, free of charge, to any person obtaining a copy
* of this software and associated documentation files (the "Software"), to deal
* in the Software without restriction, including without limitation the rights
* to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
* copies of the Software, and to permit persons to whom the Software is
* furnished to do so, subject to the following conditions:
*
* The above copyright notice and this permission notice shall be included in all
* copies or substantial portions of the Software.
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
* IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
* FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
* AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
* LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
* OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
* SOFTWARE.
*/

// The CMake packaging check: the shared self-test, and the Q8.8 tanh lookup
// table that tinymind::lookup_tables was asked to compile.

#include <cstdio>

#include "tanhValues16Bit.hpp"
#include "tinymind_selftest.hpp"

#if !TINYMIND_USE_TANH_8_8
#error "tinymind::lookup_tables did not pass TINYMIND_USE_TANH_8_8 on to its consumer"
#endif

int main()
{
    const tinymind_selftest::Result r = tinymind_selftest::run();
    // The table runs from tanh(-6) to just below tanh(6): its last entry is
    // close to +1, 0x100 in Q8.8. A missing table fails to link.
    const unsigned top = tinymind::TanhValuesTableQ8_8::values[NUMBER_OF_ACTIVATION_TABLE_VALUES - 1];
    std::printf("tinymind self-test: %u of %u outputs differ from TFLM\n",
                static_cast<unsigned>(r.differing), static_cast<unsigned>(r.checked));
    std::printf("Q8.8 tanh table, last entry: 0x%03X\n", top);
    const bool pass = r.checked > 0 && r.differing == 0 && top > 0xF0u;
    std::printf("tinymind self-test: %s\n", pass ? "PASS" : "FAIL");
    return pass ? 0 : 1;
}
