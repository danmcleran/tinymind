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

// Empty backend: harness + board only, for flash/RAM subtraction.
#include "backend.hpp"

extern "C"
{

const char* const be_name = "baseline";
const int be_n_test = 1;
const uint8_t* const be_labels = nullptr;
const uint32_t be_expect_hash = 0;

int be_setup(void)
{
    return 0;
}

void be_load(int)
{
}

void be_invoke(void)
{
    __asm volatile ("" ::: "memory");
}

int be_argmax(void)
{
    return 0;
}

uint32_t be_hash(uint32_t h)
{
    return h;
}

uint32_t be_arena_used(void)
{
    return 0;
}

int be_profile(uint32_t*, const char**, int)
{
    return 0;
}

} // extern "C"
