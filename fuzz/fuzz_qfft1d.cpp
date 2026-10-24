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
 * Properties checked (a failure traps):
 *   1. Forward accuracy: when every input component is within +/-23170 (so
 *      the complex magnitude, which a scaled stage never grows, stays inside
 *      int16 and no stage saturates), forward() is within 3 units per stage
 *      of DFT(x) / N computed in double.
 *   2. Inverse accuracy: when the spectrum's L1 norm leaves headroom (so no
 *      unscaled stage saturates), inverse() is within 3 * N units of the
 *      unscaled inverse DFT of its input. The looser bound is inherent: an
 *      unscaled stage doubles the error carried into it.
 *   3. magnitudeSquared is re^2 + im^2, saturated to INT32_MAX.
 *
 * Build/run: see fuzz/Makefile (`make fuzz` / `make fuzz-ci`).
 *
 * Copyright (c) 2026 Dan McLeran. MIT (see LICENSE.txt).
 */
#include <cstddef>
#include <cstdint>

#include <cmath>

#include "qfft1d.hpp"
#include "qcalibration.hpp"
#include "fuzz_check.hpp"

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

// The DFT of (real, imag) in double, with sign -1 (forward) or +1
// (inverse) in the exponent, divided by `divisor`.
template<std::size_t N>
void referenceDFT(const int16_t* real, const int16_t* imag, int sign, double divisor,
                  double* out_real, double* out_imag)
{
    const double two_pi = 6.283185307179586476925286766559;
    for (std::size_t k = 0; k < N; ++k)
    {
        double acc_r = 0.0;
        double acc_i = 0.0;
        for (std::size_t n = 0; n < N; ++n)
        {
            const double phase = static_cast<double>(sign) * two_pi *
                                 static_cast<double>((k * n) % N) / static_cast<double>(N);
            const double c = std::cos(phase);
            const double s = std::sin(phase);
            acc_r += real[n] * c - imag[n] * s;
            acc_i += real[n] * s + imag[n] * c;
        }
        out_real[k] = acc_r / divisor;
        out_imag[k] = acc_i / divisor;
    }
}

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

    const double stages = static_cast<double>(QFFT1D<N>::NumStages);
    double want_r[N];
    double want_i[N];

    bool forward_checkable = true;
    for (std::size_t k = 0; k < N; ++k)
    {
        if (real[k] < -23170 || real[k] > 23170 || imag[k] < -23170 || imag[k] > 23170)
        {
            forward_checkable = false;
        }
    }
    referenceDFT<N>(real, imag, -1, static_cast<double>(N), want_r, want_i);

    qfft.forward(real, imag);
    tinymind_fuzz::emit(real, sizeof(real));
    tinymind_fuzz::emit(imag, sizeof(imag));
    if (forward_checkable)
    {
        for (std::size_t k = 0; k < N; ++k)
        {
            TINYMIND_FUZZ_CHECK(std::fabs(real[k] - want_r[k]) <= 3.0 * stages);
            TINYMIND_FUZZ_CHECK(std::fabs(imag[k] - want_i[k]) <= 3.0 * stages);
        }
    }

    double l1 = 0.0;
    for (std::size_t k = 0; k < N; ++k) { l1 += std::fabs(real[k]) + std::fabs(imag[k]); }
    const bool inverse_checkable = (l1 + 3.0 * static_cast<double>(N) <= 32767.0);
    referenceDFT<N>(real, imag, +1, 1.0, want_r, want_i);

    qfft.inverse(real, imag);
    tinymind_fuzz::emit(real, sizeof(real));
    tinymind_fuzz::emit(imag, sizeof(imag));
    if (inverse_checkable)
    {
        for (std::size_t k = 0; k < N; ++k)
        {
            TINYMIND_FUZZ_CHECK(std::fabs(real[k] - want_r[k]) <= 3.0 * static_cast<double>(N));
            TINYMIND_FUZZ_CHECK(std::fabs(imag[k] - want_i[k]) <= 3.0 * static_cast<double>(N));
        }
    }

    QFFT1D<N>::magnitudeSquared(real, imag, mag_sq);
    tinymind_fuzz::emit(mag_sq, sizeof(mag_sq));
    for (std::size_t k = 0; k < N; ++k)
    {
        const int64_t sq = static_cast<int64_t>(real[k]) * real[k] +
                           static_cast<int64_t>(imag[k]) * imag[k];
        TINYMIND_FUZZ_CHECK(mag_sq[k] == ((sq > INT32_MAX) ? INT32_MAX : sq));
    }
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
