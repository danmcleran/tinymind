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

// Shared by the host generator (gen_tw.cpp) and the board image
// (fft_bench.cpp), so both compute the outputs they compare the same way.
#pragma once

#include "tinymind_platform.hpp"
#include "qfft1d.hpp"

#include <stddef.h>
#include <stdint.h>

namespace fftbench
{

inline uint32_t lcg(uint32_t& state)
{
    state = state * 1664525u + 1013904223u;
    return state;
}

// Full-scale pseudo-random int16 samples; seed selects the frame.
inline void makeInput(int16_t* x, size_t n, uint32_t seed)
{
    uint32_t state = seed;
    for (size_t i = 0; i < n; ++i)
    {
        x[i] = static_cast<int16_t>(lcg(state) >> 16);
    }
}

// FNV-1a over the bytes of v, low byte first.
inline uint32_t fnv(uint32_t h, const int16_t* v, size_t n)
{
    for (size_t i = 0; i < n; ++i)
    {
        const uint16_t u = static_cast<uint16_t>(v[i]);
        h = (h ^ (u & 0xFFu)) * 16777619u;
        h = (h ^ (u >> 8)) * 16777619u;
    }
    return h;
}

// What a caller with real samples does before QFFT1D: copy them into the
// real buffer and zero the imaginary one. Timed separately.
template<size_t N>
__attribute__((noinline)) void loadComplex(const int16_t* in, int16_t* re, int16_t* im)
{
    for (size_t i = 0; i < N; ++i)
    {
        re[i] = in[i];
        im[i] = 0;
    }
}

// Each transform in its own noinline function, so the timed region is the
// transform and nothing the compiler merged into it.
template<size_t N>
__attribute__((noinline)) void runQfft(const tinymind::QFFT1D<N>& fft, int16_t* re, int16_t* im)
{
    fft.forward(re, im);
}

template<size_t N>
__attribute__((noinline)) void runQrfft(const tinymind::QRFFT1D<N>& fft, const int16_t* in,
                                        int16_t* re, int16_t* im)
{
    fft.forward(in, re, im);
}

// Output hash over frames 1..8, the bins both transforms produce
// (0..N/2), real then imaginary. re/im need N entries for QFFT1D.
template<size_t N>
void hashBoth(const tinymind::QFFT1D<N>& qfft, const tinymind::QRFFT1D<N>& qrfft,
              int16_t* in, int16_t* re, int16_t* im, uint32_t* hq, uint32_t* hr)
{
    *hq = 0x811C9DC5u;
    *hr = 0x811C9DC5u;
    for (uint32_t seed = 1; seed <= 8; ++seed)
    {
        makeInput(in, N, seed);
        loadComplex<N>(in, re, im);
        runQfft<N>(qfft, re, im);
        *hq = fnv(fnv(*hq, re, N / 2 + 1), im, N / 2 + 1);
        runQrfft<N>(qrfft, in, re, im);
        *hr = fnv(fnv(*hr, re, N / 2 + 1), im, N / 2 + 1);
    }
}

} // namespace fftbench
