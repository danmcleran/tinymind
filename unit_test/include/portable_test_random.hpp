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

#ifndef TINYMIND_PORTABLE_TEST_RANDOM_HPP
#define TINYMIND_PORTABLE_TEST_RANDOM_HPP

#include <cstdint>

// Test-only. Not part of the shipped library.
//
// Why this exists
// ---------------
// Several suites initialize network weights from a seeded RNG and then assert
// on what training converges to. That is only reproducible if the sequence is
// reproducible -- and std::uniform_real_distribution is NOT. The standard
// specifies the distribution, not the mapping from engine output to value, so
// libstdc++ and libc++ return different doubles from identical engine state.
// std::default_random_engine is likewise a typedef the implementation picks.
//
// The consequence was real: the nightly MSan job, which links an instrumented
// libc++, saw unit_test/nn fail two tolerance checks that pass under libstdc++
// -- not a defect, just a different draw sequence.
//
// Why it reproduces libstdc++ rather than picking something cleaner
// ----------------------------------------------------------------
// The obvious fix -- switch to std::mt19937, whose output IS specified, and do
// the scaling by hand -- was tried first and rejected on evidence. It is a
// perfectly good generator: statistically indistinguishable from the current
// one (mean ~0, mean|x| ~0.5 over 2000 draws). But it produces a different
// sequence, and two tests do not survive that:
//
//   test_case_lstm_weight_serialization   0.605 against a 0.02 bound
//   test_case_rmsprop_fixedpoint_xor      average error 9 against a bound of 4
//
// Those tolerances are calibrated against the draws the tests have always seen.
// Loosening them to accommodate a new generator would weaken two real gates to
// settle a question neither is asking about.
//
// Making them robust to any initialization is the deeper fix and a much larger
// change: it means deciding what each of those tolerances should assert, which
// risks masking genuine regressions if done in bulk. Freezing the sequence the
// tests were written against gets the portability without touching a single
// tolerance, and leaves that larger cleanup as separate work.
//
// What is frozen
// --------------
// libstdc++'s std::default_random_engine is minstd_rand0: x = 16807x mod
// (2^31 - 1). Its uniform_real_distribution<double> goes through
// generate_canonical<double, 53>, which for this engine consumes exactly two
// draws (b = 53 bits, log2(r) = 30, so k = ceil(53/30) = 2) and forms
// (d1 + d2*r) / r^2 before scaling into [low, high).
//
// Verified bit-identical to libstdc++'s output over 20,000 draws: 0 mismatches,
// max difference 0. test_case_portable_uniform_real_matches_frozen_sequence in
// nn_unit_test.cpp locks the first draws in as an assertion, so a future edit
// cannot quietly change the sequence and silently re-tune every training test
// that depends on it.
class PortableUniformReal
{
public:
    PortableUniformReal(const double low, const double high, const uint32_t s)
        : mState(s != 0u ? s : 1u), mLow(low), mHigh(high)
    {
    }

    double operator()()
    {
        // r = engine.max() - engine.min() + 1, i.e. (2^31 - 2) - 1 + 1.
        const double r = 2147483646.0;
        const double d1 = static_cast<double>(next() - 1u);
        const double d2 = static_cast<double>(next() - 1u);
        const double sum = d1 + (d2 * r);
        const double canonical = sum / (r * r);

        return (canonical * (mHigh - mLow)) + mLow;
    }

    void seed(const uint32_t s)
    {
        mState = (s != 0u) ? s : 1u;
    }

private:
    uint32_t next()
    {
        mState = static_cast<uint32_t>((16807ULL * static_cast<uint64_t>(mState)) % 2147483647ULL);

        return mState;
    }

    uint32_t mState;
    double   mLow;
    double   mHigh;
};

// Not covered here
// ----------------
// unit_test/qlearn still uses std::default_random_engine with
// std::uniform_int_distribution to pick maze states and actions. That mapping
// is implementation-defined too, so the sequence differs across libraries in
// principle. It is left alone deliberately: the suite passes under the
// instrumented libc++ in the nightly MSan job, so there is no observed problem
// to fix, and changing the draw sequence of a passing suite risks breaking its
// assertions for no benefit. Worth revisiting only if it actually diverges.

#endif // TINYMIND_PORTABLE_TEST_RANDOM_HPP
