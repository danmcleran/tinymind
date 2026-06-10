/*
 * libFuzzer harness for the int16 QFFT1D radix-2 compute kernel.
 *
 * Rationale: the FFT butterfly stages are the densest fixed-point arithmetic
 * in the library -- Q1.15 twiddle multiplies, scaled butterfly adds, and the
 * bit-reversal permute, repeated over log2(N) stages in place. Saturation and
 * intermediate-overflow bugs only surface on adversarial spectra (full-scale
 * impulses, alternating +/-32767 rails) that fixed-input tests do not cover.
 * magnitudeSquared then squares two int16 rails into an int32 -- the classic
 * (-32768)^2 corner.
 *
 * Twiddles are built through buildQFFTTwiddles, the same calibration helper the
 * importers use, so findings are reachable in real deployments.
 *
 * Build/run: see fuzz/Makefile (`make fuzz` / `make fuzz-ci`).
 *
 * Copyright (c) 2026 Dan McLeran. MIT (see LICENSE.txt).
 */
#include <cstddef>
#include <cstdint>

#include "qfft1d.hpp"
#include "qcalibration.hpp"

using tinymind::QFFT1D;
using tinymind::buildQFFTTwiddles;

namespace {

// Linear byte cursor. Reads past the end return 0.
struct ByteReader
{
    const uint8_t* p;
    size_t n;
    size_t i;

    explicit ByteReader(const uint8_t* data, size_t size) : p(data), n(size), i(0) {}

    uint8_t u8() { return (i < n) ? p[i++] : static_cast<uint8_t>(0); }

    // Full int16 range, including the asymmetric -32768 rail.
    int16_t i16()
    {
        const uint32_t hi = u8();
        const uint32_t lo = u8();
        return static_cast<int16_t>((hi << 8) | lo);
    }
};

template<std::size_t N>
void run(ByteReader& br)
{
    int16_t real[N];
    int16_t imag[N];
    int32_t mag_sq[N];

    for (std::size_t k = 0; k < N; ++k) { real[k] = br.i16(); imag[k] = br.i16(); }

    int16_t cos_t[N / 2];
    int16_t sin_t[N / 2];
    buildQFFTTwiddles(N, cos_t, sin_t);

    QFFT1D<N> qfft;
    qfft.twiddle_cos = cos_t;
    qfft.twiddle_sin = sin_t;

    qfft.forward(real, imag);
    qfft.inverse(real, imag);
    QFFT1D<N>::magnitudeSquared(real, imag, mag_sq);

    // Touch the outputs so the calls cannot be optimized away.
    volatile int64_t sink = 0;
    for (std::size_t k = 0; k < N; ++k) { sink += real[k] + imag[k] + mag_sq[k]; }
    (void)sink;
}

} // namespace

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size)
{
    ByteReader br(data, size);

    // First byte selects a power-of-two length so the corpus can steer it.
    switch (br.u8() & 0x03)
    {
        case 0:  run<8>(br);   break;
        case 1:  run<16>(br);  break;
        case 2:  run<32>(br);  break;
        default: run<64>(br);  break;
    }
    return 0;
}
