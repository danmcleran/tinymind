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

// The ESP-IDF packaging check: the shared self-test, and the Q8.8 tanh table
// the tinymind component compiled (the project sets TINYMIND_USE_TANH_8_8 for
// every component).

#include <cstdio>

#include "tanhValues16Bit.hpp"
#include "tinymind_selftest.hpp"

#if !TINYMIND_ENABLE_QUANTIZATION || !TINYMIND_USE_TANH_8_8
#error "the project's compile definitions did not reach the application"
#endif

extern "C" void app_main(void)
{
    const tinymind_selftest::Result r = tinymind_selftest::run();
    const unsigned top = tinymind::TanhValuesTableQ8_8::values[NUMBER_OF_ACTIVATION_TABLE_VALUES - 1];
    std::printf("tinymind self-test: %u of %u outputs differ from TFLM\n",
                static_cast<unsigned>(r.differing), static_cast<unsigned>(r.checked));
    const bool pass = r.checked > 0 && r.differing == 0 && top > 0xF0u;
    std::printf("tinymind self-test: %s\n", pass ? "PASS" : "FAIL");
}
