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

// Phase 16 integration tests. Each fixture shells out to one of the four
// reference exemplar binaries with --golden and compares the emitted
// int8 byte stream to the baked-in expected string byte-for-byte.
//
// The exemplar binaries are deterministic (hand-crafted weights, fixed
// synthetic dataset, pure-integer forward), so the int8 output stream
// is invariant across SIMD gate combos by Phase 14's bit-exactness
// guarantee. A diff against the baked-in golden therefore catches any
// silent regression in the inference path regardless of which backend
// the dispatch resolves to.

#include "compiler.h"

#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <string>

#define BOOST_TEST_MODULE integration_unit_test
TINYMIND_DISABLE_WARNING_PUSH
TINYMIND_DISABLE_WARNING("-Wdangling-reference")
#include <boost/test/included/unit_test.hpp>
TINYMIND_DISABLE_WARNING_POP

namespace {

std::string runGolden(const char* binary_relpath)
{
    // The unit_test/integration binary runs from output/ (per
    // 'make run'), so the example binaries are reachable at
    // ../../../examples/<name>/output/<name>.
    std::string cmd = binary_relpath;
    cmd += " --golden 2>/dev/null";
    std::FILE* pipe = popen(cmd.c_str(), "r");
    BOOST_REQUIRE_MESSAGE(pipe != nullptr,
                          "popen failed for: " << cmd);
    std::string out;
    char buf[1024];
    while (std::fgets(buf, sizeof(buf), pipe) != nullptr)
    {
        out += buf;
    }
    const int rc = pclose(pipe);
    BOOST_REQUIRE_MESSAGE(rc == 0,
                          "non-zero exit (" << rc << ") for: " << cmd);
    return out;
}

void requireGoldenMatch(const char* binary_relpath, const char* expected)
{
    const std::string actual = runGolden(binary_relpath);
    if (actual != expected)
    {
        std::cerr << "Golden mismatch for " << binary_relpath << "\n"
                  << "---- expected ----\n" << expected
                  << "---- actual ------\n" << actual
                  << "------------------\n";
    }
    BOOST_REQUIRE_EQUAL(actual, std::string(expected));
}

// ---------------------------------------------------------------------------
// Baked-in golden int8 byte streams. Captured 2026-05-11 from a build at
// TINYMIND_ENABLE_FLOAT=1 TINYMIND_ENABLE_STD=1 TINYMIND_ENABLE_QUANTIZATION=1
// (Phase 14 gates all off). Any drift in the example pipeline, the qaffine
// requantizer, the qcalibration helpers, or any SIMD specialization that
// claims bit-exactness will trip these strings.
// ---------------------------------------------------------------------------

const char* kGoldenResnet18 =
    "# resnet18_block_int8 golden output\n"
    "# samples=4 classes=4\n"
    "sample 0: -8 102 89 -41\n"
    "sample 1: -20 71 55 -60\n"
    "sample 2: -16 89 78 -44\n"
    "sample 3: 5 124 109 -34\n";

const char* kGoldenMobileNetV2 =
    "# mobilenetv2_int8 golden output\n"
    "# samples=4 classes=4\n"
    "sample 0: -98 -43 36 99\n"
    "sample 1: -98 -43 37 99\n"
    "sample 2: -99 -43 37 100\n"
    "sample 3: -96 -42 36 97\n";

const char* kGoldenMixedPrecisionKws =
    "# mixed_precision_kws golden output\n"
    "# samples=4 classes=4\n"
    "sample 0: 65 116 48 -103\n"
    "sample 1: 62 100 27 -117\n"
    "sample 2: 51 76 4 -127\n"
    "sample 3: 57 98 30 -112\n";

const char* kGoldenMixedPrecisionMlpInt8QFormat =
    "# mixed_precision_mlp_int8_qformat golden output\n"
    "# samples=4 classes=4\n"
    "sample 0: -57 -17 -37 -105\n"
    "sample 1: 1 73 38 -86\n"
    "sample 2: -106 -81 -81 -107\n"
    "sample 3: -73 -11 -12 -74\n";

const char* kGoldenTransformerEncoder =
    "# transformer_encoder_int8 golden output\n"
    "# samples=8 cells=32\n"
    "sample 0: -108 -85 -66 -43 -11 -65 -51 -40 108 127 47 45 58 75 77 -13 0 9 3 0 -64 -42 -31 -28 -70 -127 -117 -95 -61 -29 -87 -65\n"
    "sample 1: -42 -18 0 20 45 -14 -4 3 99 113 30 21 32 49 50 -41 -28 -20 -15 -6 -70 -50 -32 -13 -42 -97 -81 -56 -19 13 -45 -27\n"
    "sample 2: 4 26 44 60 76 9 18 24 80 92 4 -6 5 23 26 -61 -56 -46 -36 -21 -78 -54 -32 -10 -16 -71 -49 -22 10 39 -20 -4\n"
    "sample 3: 48 70 79 86 98 33 39 36 55 65 -23 -31 -19 2 8 -74 -93 -80 -71 -51 -98 -68 -47 -27 8 -44 -24 0 31 56 -9 3\n"
    "sample 4: 60 76 76 74 86 23 24 14 26 36 -47 -50 -36 -12 1 -73 -108 -90 -76 -53 -96 -64 -42 -22 52 -2 20 41 63 81 15 24\n"
    "sample 5: 50 64 56 49 59 -2 -1 -14 1 11 -63 -56 -40 -17 4 -59 -97 -75 -57 -32 -73 -40 -21 -3 107 54 69 82 98 116 46 47\n"
    "sample 6: 30 41 32 22 32 -28 -25 -32 -16 -5 -73 -56 -33 -9 13 -42 -102 -78 -60 -34 -77 -47 -30 -16 122 66 71 72 87 105 31 21\n"
    "sample 7: 4 12 4 -2 9 -48 -41 -43 -70 -54 -118 -97 -70 -41 -17 -71 -78 -53 -35 -12 -61 -37 -24 -14 122 62 59 53 66 84 6 -6\n";

} // namespace

BOOST_AUTO_TEST_CASE(resnet18_block_int8_golden_match)
{
    requireGoldenMatch(
        "../../../examples/resnet18_block_int8/output/resnet18_block_int8",
        kGoldenResnet18);
}

BOOST_AUTO_TEST_CASE(mobilenetv2_int8_golden_match)
{
    requireGoldenMatch(
        "../../../examples/mobilenetv2_int8/output/mobilenetv2_int8",
        kGoldenMobileNetV2);
}

BOOST_AUTO_TEST_CASE(mixed_precision_kws_golden_match)
{
    requireGoldenMatch(
        "../../../examples/mixed_precision_kws/output/mixed_precision_kws",
        kGoldenMixedPrecisionKws);
}

BOOST_AUTO_TEST_CASE(transformer_encoder_int8_golden_match)
{
    requireGoldenMatch(
        "../../../examples/transformer_encoder_int8/output/transformer_encoder_int8",
        kGoldenTransformerEncoder);
}

BOOST_AUTO_TEST_CASE(mixed_precision_mlp_int8_qformat_golden_match)
{
    requireGoldenMatch(
        "../../../examples/mixed_precision_mlp_int8_qformat/output/mixed_precision_mlp_int8_qformat",
        kGoldenMixedPrecisionMlpInt8QFormat);
}
