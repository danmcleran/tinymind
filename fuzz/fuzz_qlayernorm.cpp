/*
 * libFuzzer harness for the int8 QLayerNorm1D compute kernel.
 *
 * Rationale: tinymind has no runtime deserialization surface -- weights are
 * baked in as const arrays by the offline importers, so there is no parser to
 * fuzz. The real input-dependent risk is *arithmetic* in the int8 layers fed
 * adversarial activation tensors and quantization parameters: the exact class
 * of bug that fixed-input unit tests miss (e.g. the int32 overflow in
 * QLayerNorm's gamma multiply that only a near-zero-variance row triggers).
 *
 * This harness drives QLayerNorm1D::forward with fuzz-derived input, gamma,
 * beta and output scale, under ASan+UBSan. Short/empty inputs decay to
 * constant (zero-variance) rows on purpose -- that is the pathological case
 * that inflates inv_stddev. Quantization parameters are built through the same
 * helpers the importers use, so findings are reachable in real deployments,
 * not artifacts of impossible parameter values.
 *
 * Properties checked on every input (a failure traps):
 *   1. Accuracy: given the kernel's integer row mean and variance (the two
 *      documented integer roundings), each output is within a derived error
 *      bound of gamma * (x - mean) / sqrt(var + eps) + beta computed in
 *      double. The bound covers the Q30 inverse square root, the Q14
 *      normalized value, the Q28 product clamp and the output rounding, so
 *      a scaling, sign or indexing error lands far outside it.
 *   2. Constant rows: a row with zero variance outputs exactly
 *      clamp(beta + zero point) for each feature.
 *   3. Shift invariance: adding a constant to a row whose sum keeps its
 *      sign leaves the output unchanged bit for bit (the mean rounds half
 *      away from zero, so the constant passes through it exactly).
 *
 * Build/run: see fuzz/Makefile (`make fuzz` / `make fuzz-ci`).
 *
 * Copyright (c) Dan McLeran. MIT (see LICENSE.txt).
 */
#include <cstddef>
#include <cstdint>

#include <cmath>

#include "qlayernorm.hpp"
#include "fuzz_check.hpp"

using tinymind::QLayerNorm1D;
using tinymind::quantizeLayerNormGamma;
using tinymind::quantizeLayerNormBeta;
using tinymind::buildQLayerNormOutputParams;

namespace {

// Linear byte cursor. Reads past the end return 0, so short inputs exercise
// constant/zero-variance rows rather than being rejected.
struct ByteReader
{
    const uint8_t* p;
    size_t n;
    size_t i;

    explicit ByteReader(const uint8_t* data, size_t size) : p(data), n(size), i(0) {}

    uint8_t u8() { return (i < n) ? p[i++] : static_cast<uint8_t>(0); }
    int8_t  i8() { return static_cast<int8_t>(u8()); }

    // Map a byte to a float in [lo, hi].
    float f(float lo, float hi) { return lo + (static_cast<float>(u8()) / 255.0f) * (hi - lo); }
};

template<std::size_t R, std::size_t F>
void run(ByteReader& br)
{
    int8_t  input[R * F];
    int8_t  output[R * F];
    int16_t gamma_int[F];
    int32_t beta_int[F];
    float   gamma_f[F];
    float   beta_f[F];

    for (std::size_t k = 0; k < R * F; ++k) { input[k] = br.i8(); output[k] = 0; }
    for (std::size_t k = 0; k < F; ++k)     { gamma_f[k] = br.f(-4.0f, 4.0f); }
    for (std::size_t k = 0; k < F; ++k)     { beta_f[k]  = br.f(-2.0f, 2.0f); }

    // A small positive output scale spanning the realistic range.
    const float output_scale = br.f(1.0f / 4096.0f, 1.0f);

    quantizeLayerNormGamma(gamma_f, F, gamma_int);
    quantizeLayerNormBeta(beta_f, F, output_scale, beta_int);

    int32_t out_mult = 0;
    int32_t out_shift = 0;
    buildQLayerNormOutputParams(output_scale, out_mult, out_shift);

    QLayerNorm1D<int8_t, int8_t, R, F> ln;
    ln.gamma = gamma_int;
    ln.beta = beta_int;
    // epsilon down to 1 keeps inv-sqrt finite while letting near-zero variance
    // drive inv_stddev as large as the fixed-point domain allows.
    ln.epsilon_q = static_cast<int32_t>(br.u8()) + 1;
    ln.output_multiplier = out_mult;
    ln.output_shift = out_shift;
    ln.output_zero_point = 0;
    ln.qmin = -128;
    ln.qmax = 127;

    // Read after every pre-existing field, so archived corpus inputs decode
    // as they did before. One bit flattens every row to its first element, so the zero-variance
    // branch is reached by design rather than only by short inputs.
    const bool constant_rows = (br.u8() & 0x01) != 0;
    if (constant_rows)
    {
        for (std::size_t k = 0; k < R * F; ++k) { input[k] = input[(k / F) * F]; }
    }

    ln.forward(input, output);
    tinymind_fuzz::emit(output, sizeof(output));

    const int32_t N = static_cast<int32_t>(F);
    for (std::size_t r = 0; r < R; ++r)
    {
        const int8_t* in  = input  + r * F;
        const int8_t* out = output + r * F;

        // The kernel's integer statistics: mean rounded half away from zero,
        // variance (about that mean) rounded half up.
        int64_t sum = 0;
        for (std::size_t i = 0; i < F; ++i) { sum += in[i]; }
        const int64_t bias = (sum >= 0) ? (N / 2) : -(N / 2);
        const int32_t mean_q = static_cast<int32_t>((sum + bias) / N);
        int64_t ssum = 0;
        for (std::size_t i = 0; i < F; ++i)
        {
            const int64_t dev = in[i] - mean_q;
            ssum += dev * dev;
        }
        const int64_t var_q = (ssum + N / 2) / N;
        const double denom = std::sqrt(static_cast<double>(var_q + ln.epsilon_q));

        for (std::size_t i = 0; i < F; ++i)
        {
            const double normalized = static_cast<double>(in[i] - mean_q) / denom;
            const double gamma_real = static_cast<double>(gamma_int[i]) / 16384.0;
            double weighted = normalized * gamma_real;
            // The kernel holds gamma * normalized as an int32 Q28 value.
            if (weighted > 8.0)  { weighted = 8.0; }
            if (weighted < -8.0) { weighted = -8.0; }
            double want = weighted / static_cast<double>(output_scale) +
                          static_cast<double>(beta_int[i]) +
                          static_cast<double>(ln.output_zero_point);
            if (want < -128.0) { want = -128.0; }
            if (want > 127.0)  { want = 127.0; }

            // Error of the Q14 normalized value: the inverse square root is
            // within 2^-13 relative, plus one unit of the floor.
            const double q14_error = (2.0 * std::fabs(normalized) + 2.0) / 16384.0;
            const double bound = q14_error * std::fabs(gamma_real) /
                                 static_cast<double>(output_scale) + 1.0;
            TINYMIND_FUZZ_CHECK(std::fabs(static_cast<double>(out[i]) - want) <= bound);

            if (constant_rows)
            {
                int32_t expect = beta_int[i] + ln.output_zero_point;
                if (expect < -128) { expect = -128; }
                if (expect > 127)  { expect = 127; }
                TINYMIND_FUZZ_CHECK(out[i] == expect);
            }
        }
    }

    // Shift every row by one constant, kept only where each row's sum keeps
    // its sign (so its mean rounds the same way) and every value stays in
    // range.
    int32_t delta = (static_cast<int32_t>(br.u8()) % 33) - 16;
    for (std::size_t r = 0; r < R && delta != 0; ++r)
    {
        int64_t sum = 0;
        for (std::size_t i = 0; i < F; ++i)
        {
            const int32_t moved = static_cast<int32_t>(input[r * F + i]) + delta;
            if (moved < -128 || moved > 127) { delta = 0; }
            sum += input[r * F + i];
        }
        const int64_t moved_sum = sum + static_cast<int64_t>(delta) * N;
        if ((sum >= 0) != (moved_sum >= 0)) { delta = 0; }
    }
    int8_t shifted_in[R * F];
    int8_t shifted_out[R * F];
    for (std::size_t k = 0; k < R * F; ++k) { shifted_in[k] = static_cast<int8_t>(input[k] + delta); }
    ln.forward(shifted_in, shifted_out);
    for (std::size_t k = 0; k < R * F; ++k)
    {
        TINYMIND_FUZZ_CHECK(shifted_out[k] == output[k]);
    }
}

} // namespace

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size)
{
    ByteReader br(data, size);

    // A few shapes: single row (variance edge cases), small and wider feature
    // counts. The first byte selects the shape so the corpus can steer it.
    switch (br.u8() & 0x03)
    {
        case 0:  run<1, 4>(br);  break;
        case 1:  run<2, 8>(br);  break;
        case 2:  run<1, 16>(br); break;
        default: run<4, 8>(br);  break;
    }
    return 0;
}
