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

// Board image: cycles per forward transform for FFT1D<Q8.8, N> (the
// QValue FFT of docs/architectures/fft.md), N = 64 .. 256, real input with
// a zero imaginary part. Same report format as fft_bench.cpp.
#include "board.hpp"
#include "fft_common.hpp"

#ifndef MAXN
#define MAXN 1024
#endif

#define WANT_FFT1D
#include "tw.h"

#ifndef REPS
#define REPS 33
#endif

namespace
{

fftbench::Q8_8 gRe[256];
fftbench::Q8_8 gIm[256];
fftbench::Q8_8 gCos[128];
fftbench::Q8_8 gSin[128];
uint32_t gT[REPS];

void sort(uint32_t* a, int n)
{
    for (int i = 1; i < n; ++i)
    {
        const uint32_t x = a[i];
        int j = i - 1;
        while (j >= 0 && a[j] > x)
        {
            a[j + 1] = a[j];
            --j;
        }
        a[j + 1] = x;
    }
}

void kv(const char* key, uint32_t value)
{
    board_puts(key);
    board_puts("=");
    board_put_u32(value);
    board_puts(" ");
}

template<size_t N>
void bench(const int16_t* cosRaw, const int16_t* sinRaw, uint32_t expect)
{
    for (size_t k = 0; k < N / 2; ++k)
    {
        gCos[k].setValue(cosRaw[k]);
        gSin[k].setValue(sinRaw[k]);
    }
    static tinymind::FFT1D<fftbench::Q8_8, N> fft;
    fft.setTwiddleFactors(gCos, gSin);

    fftbench::makeInputQ8_8(gRe, gIm, N, 1u);
    fftbench::runFft1d<N>(fft, gRe, gIm);

    for (int r = 0; r < REPS; ++r)
    {
        fftbench::makeInputQ8_8(gRe, gIm, N, 1u + static_cast<uint32_t>(r % 8));
        const uint32_t a = board_cycles();
        fftbench::runFft1d<N>(fft, gRe, gIm);
        gT[r] = board_cycles() - a;
    }
    sort(gT, REPS);

    const uint32_t h = fftbench::hashFft1d<N>(fft, gRe, gIm);

    kv("N", N);
    kv("fft1d_min", gT[0]);
    kv("fft1d_med", gT[REPS / 2]);
    kv("fft1d_max", gT[REPS - 1]);
    kv("fft1d_match", h == expect ? 1u : 0u);
    board_puts("\n");
}

[[noreturn]] void halt()
{
    board_flush();
    for (;;)
    {
        // A side-effect-free infinite loop is undefined behaviour in C++.
        __asm volatile ("" ::: "memory");
    }
}

} // namespace

int main()
{
    board_init();
    board_puts("\n# begin bench\n");
    board_puts("type=Q8.8\n");
    bench<64>(tw88_cos_64, tw88_sin_64, HASH_FFT1D_64);
    bench<128>(tw88_cos_128, tw88_sin_128, HASH_FFT1D_128);
    bench<256>(tw88_cos_256, tw88_sin_256, HASH_FFT1D_256);
    board_puts("# end bench\n");
    halt();
}
