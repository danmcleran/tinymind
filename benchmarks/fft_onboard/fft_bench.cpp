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

// Board image: cycles per forward transform for QFFT1D on real input (the
// imaginary buffer zeroed) and for QRFFT1D, N = 64 .. MAXN. Prints one
// key=value line per N over the ST-LINK virtual COM port.
#include "board.hpp"
#include "fft_common.hpp"

#ifndef MAXN
#define MAXN 1024
#endif

#include "tw.h"

#ifndef REPS
#define REPS 33
#endif

namespace
{

int16_t gIn[MAXN];
int16_t gRe[MAXN];
int16_t gIm[MAXN];
uint32_t gLoad[REPS];
uint32_t gQfft[REPS];
uint32_t gQrfft[REPS];

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
void bench(const int16_t* cosTable, const int16_t* sinTable, uint32_t expectQfft, uint32_t expectQrfft)
{
    tinymind::QFFT1D<N> qfft;
    qfft.twiddle_cos = cosTable;
    qfft.twiddle_sin = sinTable;
    tinymind::QRFFT1D<N> qrfft;
    qrfft.twiddle_cos = cosTable;
    qrfft.twiddle_sin = sinTable;

    // One untimed pass warms the flash accelerator and branch state.
    fftbench::makeInput(gIn, N, 1u);
    fftbench::loadComplex<N>(gIn, gRe, gIm);
    fftbench::runQfft<N>(qfft, gRe, gIm);
    fftbench::runQrfft<N>(qrfft, gIn, gRe, gIm);

    for (int r = 0; r < REPS; ++r)
    {
        fftbench::makeInput(gIn, N, 1u + static_cast<uint32_t>(r % 8));
        const uint32_t a = board_cycles();
        fftbench::loadComplex<N>(gIn, gRe, gIm);
        const uint32_t b = board_cycles();
        fftbench::runQfft<N>(qfft, gRe, gIm);
        const uint32_t c = board_cycles();
        fftbench::runQrfft<N>(qrfft, gIn, gRe, gIm);
        const uint32_t d = board_cycles();
        gLoad[r] = b - a;
        gQfft[r] = c - b;
        gQrfft[r] = d - c;
    }
    sort(gLoad, REPS);
    sort(gQfft, REPS);
    sort(gQrfft, REPS);

    uint32_t hq = 0;
    uint32_t hr = 0;
    fftbench::hashBoth<N>(qfft, qrfft, gIn, gRe, gIm, &hq, &hr);

    kv("N", N);
    kv("load_med", gLoad[REPS / 2]);
    kv("qfft_min", gQfft[0]);
    kv("qfft_med", gQfft[REPS / 2]);
    kv("qfft_max", gQfft[REPS - 1]);
    kv("qrfft_min", gQrfft[0]);
    kv("qrfft_med", gQrfft[REPS / 2]);
    kv("qrfft_max", gQrfft[REPS - 1]);
    kv("qfft_match", hq == expectQfft ? 1u : 0u);
    kv("qrfft_match", hr == expectQrfft ? 1u : 0u);
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
#ifdef TW_IN_RAM
    board_puts("twiddles=ram\n");
#else
    board_puts("twiddles=flash\n");
#endif
    bench<64>(tw_cos_64, tw_sin_64, HASH_QFFT_64, HASH_QRFFT_64);
    bench<128>(tw_cos_128, tw_sin_128, HASH_QFFT_128, HASH_QRFFT_128);
    bench<256>(tw_cos_256, tw_sin_256, HASH_QFFT_256, HASH_QRFFT_256);
    bench<512>(tw_cos_512, tw_sin_512, HASH_QFFT_512, HASH_QRFFT_512);
#if MAXN >= 1024
    bench<1024>(tw_cos_1024, tw_sin_1024, HASH_QFFT_1024, HASH_QRFFT_1024);
#endif
    board_puts("# end bench\n");
    halt();
}
