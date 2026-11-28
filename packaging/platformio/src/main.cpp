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

// The PlatformIO packaging check: the shared self-test, and the Q8.8 tanh
// lookup table the library compiled from its src/ (build_flags set
// TINYMIND_USE_TANH_8_8 for the whole build). On the host it prints the
// result and returns it; on the F411 it is only built.

#include <cstdint>

#include <tinymind.h>
#include "tanhValues16Bit.hpp"
#include "../../selftest/tinymind_selftest.hpp"

#if !defined(__arm__)
#include <cstdio>
#define TINYMIND_HOSTED_REPORT 1
#endif

volatile unsigned gDiffering;

int main()
{
    const tinymind_selftest::Result r = tinymind_selftest::run();
    const unsigned top = tinymind::TanhValuesTableQ8_8::values[NUMBER_OF_ACTIVATION_TABLE_VALUES - 1];
    const bool pass = r.checked > 0 && r.differing == 0 && top > 0xF0u;
    gDiffering = static_cast<unsigned>(r.differing);
#if defined(TINYMIND_HOSTED_REPORT)
    std::printf("tinymind self-test: %u of %u outputs differ from TFLM\n",
                static_cast<unsigned>(r.differing), static_cast<unsigned>(r.checked));
    std::printf("tinymind self-test: %s\n", pass ? "PASS" : "FAIL");
#endif
    return pass ? 0 : 1;
}
