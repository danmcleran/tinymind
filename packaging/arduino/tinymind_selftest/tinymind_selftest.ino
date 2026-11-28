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

// The Arduino packaging check: the shared self-test, run from a sketch. In
// the Arduino IDE there is no project-wide flag, so the sketch sets the
// feature macro itself, before the first tinymind header; the library's own
// source (the lookup tables) needs none of them.

#define TINYMIND_ENABLE_QUANTIZATION 1

#include <tinymind.h>
// src/ is on the include path, so the self-test (outside the library's src/)
// is reached through it.
#include <../packaging/selftest/tinymind_selftest.hpp>

void setup()
{
    Serial.begin(115200);
    while (!Serial)
    {
    }
    const tinymind_selftest::Result r = tinymind_selftest::run();
    Serial.print("tinymind self-test: ");
    Serial.print(static_cast<unsigned long>(r.differing));
    Serial.print(" of ");
    Serial.print(static_cast<unsigned long>(r.checked));
    Serial.println(" outputs differ from TFLM");
    Serial.println(r.checked > 0 && r.differing == 0 ? "tinymind self-test: PASS" : "tinymind self-test: FAIL");
}

void loop()
{
}
