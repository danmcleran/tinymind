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

// Interface every framework backend implements; harness.cpp times it.
#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C"
{
#endif

extern const char* const be_name;
extern const int be_n_test;
extern const uint8_t* const be_labels;   // nullptr when the model has no labels
extern const uint32_t be_expect_hash;    // 0 = float backend, hash not comparable

int      be_setup(void);                 // 0 on success
void     be_load(int sample);            // copy one test input into the model's input
void     be_invoke(void);                // one inference (the timed part)
int      be_argmax(void);
uint32_t be_hash(uint32_t h);            // fold outputs into an FNV-1a hash
uint32_t be_arena_used(void);            // TFLM tensor arena bytes, 0 otherwise
// One inference with per-layer DWT timing (only built with -DPROFILE). Returns layer count.
int      be_profile(uint32_t* cycles, const char** tags, int max);

#ifdef __cplusplus
}
#endif

// tinymind int8 output hashes over the whole test set, computed on the host
// by host/tm_check.cpp. TFLM must reproduce them on the board. (The pip
// TFLite interpreter differs by 1 LSB on 35/3600 MLP outputs because its
// fully connected kernel uses single rounding; see README.)
#define TM_MLP_HASH 0x85BDD037u
// Over the first 32 MLP samples only: the test set boards with 32 KB of flash
// carry (gen/subset.py).
#define TM_MLP_HASH_32 0x06CE0530u
#define TM_KWS_HASH 0x6B1854D3u
