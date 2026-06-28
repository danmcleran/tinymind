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

const char* kGoldenSeq2Seq =
    "# seq2seq_int8 golden output\n"
    "# samples=6 cells=48 cache_equiv=1\n"
    "sample 0: 41 122 41 117 39 122 46 124 26 1 -36 30 -50 27 -50 27 84 -16 31 91 18 101 28 109 36 -48 49 94 24 111 41 119 -73 -67 5 35 -37 44 -29 43 -87 17 29 54 -10 77 11 89\n"
    "sample 1: -52 27 -55 20 -58 23 -51 27 95 73 39 109 31 112 35 115 112 11 57 116 41 121 45 124 -26 -115 -21 22 -52 29 -46 30 -42 -28 48 82 14 102 35 113 -48 55 66 90 26 113 43 122\n"
    "sample 2: 35 114 33 109 29 108 30 107 109 86 50 118 38 116 36 114 19 -84 -40 19 -56 23 -51 26 43 -42 55 100 27 112 39 118 -16 -4 72 104 36 120 51 126 -112 -15 -7 14 -53 28 -46 29\n"
    "sample 3: 32 109 27 102 20 97 17 92 15 -8 -41 27 -48 33 -41 40 107 3 48 108 31 108 30 107 57 -30 66 110 35 117 41 117 -107 -97 -23 10 -60 24 -45 31 -41 60 70 93 29 113 42 120\n"
    "sample 4: -36 48 -32 44 -33 54 -16 65 93 70 31 97 14 91 14 89 104 -2 42 101 22 97 17 92 -40 -125 -28 18 -54 31 -39 41 -24 -13 61 93 22 104 34 107 -32 68 76 99 31 113 39 114\n"
    "sample 5: 0 82 4 80 4 91 21 100 78 55 15 78 -6 71 -6 70 35 -64 -18 42 -30 53 -17 64 37 -53 42 85 10 89 13 86 -27 -20 54 84 12 92 19 91 -128 -27 -18 7 -57 28 -40 39\n";

const char* kGoldenTinyGenerate =
    "# tiny_generate_int8 golden output\n"
    "# prompts=4 gen=12 token_match=48/48\n"
    "prompt 1 5 -> 6 7 0 1 2 3 4 5 6 7 0 1\n"
    "prompt 3 6 -> 7 0 1 2 3 4 5 6 7 0 1 2\n"
    "prompt 7 2 -> 3 4 5 6 7 0 1 2 3 4 5 6\n"
    "prompt 4 0 -> 1 2 3 4 5 6 7 0 1 2 3 4\n";

const char* kGoldenSeq2SeqSoftmax =
    "# seq2seq_softmax_int8 golden output\n"
    "# samples=6 cells=48 cache_equiv=1\n"
    "sample 0: 24 108 22 100 32 101 36 118 -2 -20 -65 6 -64 -1 -70 10 61 -28 7 70 6 77 15 98 23 -51 33 82 21 94 36 116 -78 -64 -5 30 -33 35 -27 49 -87 20 20 51 -3 72 15 97\n"
    "sample 1: -68 15 -74 4 -65 3 -60 19 68 53 11 83 15 83 15 98 87 -4 31 93 28 95 29 110 -39 -118 -38 9 -56 13 -52 26 -50 -29 34 73 15 88 32 112 -52 53 53 83 28 101 43 122\n"
    "sample 2: 13 96 10 87 17 84 18 96 75 59 16 88 17 82 13 92 -4 -96 -64 -2 -68 0 -64 16 25 -49 33 83 21 92 31 113 -25 -7 56 93 34 106 48 126 -116 -15 -17 9 -49 20 -43 35\n"
    "sample 3: 3 83 -5 72 1 65 -3 75 -11 -27 -68 4 -63 6 -56 27 79 -14 20 82 15 81 13 93 38 -38 44 92 27 95 32 110 -114 -97 -35 2 -59 13 -45 33 -49 56 55 85 30 100 42 120\n"
    "sample 4: -42 44 -43 36 -31 43 -18 65 62 44 -2 67 -5 59 -10 67 73 -22 10 73 3 66 -5 73 -54 -128 -47 2 -59 14 -45 37 -33 -16 46 81 20 90 29 106 -39 63 61 89 31 100 38 113\n"
    "sample 5: -7 79 -6 73 7 80 22 103 47 30 -17 53 -21 42 -28 51 15 -72 -39 25 -39 33 -26 58 22 -56 23 71 6 73 7 84 -34 -18 42 78 14 82 21 97 -126 -22 -23 6 -48 25 -33 49\n";

const char* kGoldenStateSpace =
    "# state_space_int8 golden output\n"
    "# seq=32 channels=4 cache_equiv=1\n"
    "lti: 0 4 -3 -1 15 9 -10 -13 26 7 -22 -28 37 15 -35 -26 61 14 -39 -1 84 5 -27 23 95 0 -7 25 105 1 9 12 122 -6 21 -3 127 -15 27 -19 119 -12 19 -26 113 -6 0 -12 108 -7 -15 14 90 -6 -22 23 64 4 -23 14 46 10 -17 2 28 7 0 -11 1 7 17 -22 -24 11 21 -18 -35 8 18 5 -48 -2 12 20 -65 -3 0 16 -70 -2 -17 6 -64 -8 -22 -4 -61 -13 -16 -15 -60 -7 -7 -19 -46 -2 3 -4 -26 -3 15 15 -15 -2 21 16 -4 8 14 8 17 10 3 0 35 5 -5 -9\n"
    "sel: -1 -1 -1 -1 12 2 1 -7 20 1 4 -20 29 11 6 -17 55 12 3 -7 80 -3 1 -6 91 -2 13 -3 103 -1 23 -1 120 0 31 -3 127 3 34 -12 115 4 27 -17 106 2 23 -8 99 2 18 -5 84 2 11 -5 72 -1 6 -2 63 3 2 -2 55 1 14 -5 45 0 27 -13 39 6 26 -10 35 2 21 -5 30 -5 15 -5 25 1 14 -2 23 -1 13 -1 21 1 6 -2 18 3 2 -8 15 3 4 -11 18 1 9 -4 28 0 18 -4 29 0 23 -3 31 0 18 -1 49 5 13 -2 63 -3 8 -4\n";

const char* kGoldenGbdtTabular =
    "# gbdt_tabular_int8 golden output\n"
    "# trees=5 classes=3 grid=48 agree=2256/2304\n"
    "probes: 0 0 1 1 2 2 0 2\n";

} // namespace

BOOST_AUTO_TEST_CASE(gbdt_tabular_int8_golden_match)
{
    requireGoldenMatch(
        "../../../examples/gbdt_tabular_int8/output/gbdt_tabular_int8",
        kGoldenGbdtTabular);
}

BOOST_AUTO_TEST_CASE(state_space_int8_golden_match)
{
    requireGoldenMatch(
        "../../../examples/state_space_int8/output/state_space_int8",
        kGoldenStateSpace);
}

BOOST_AUTO_TEST_CASE(seq2seq_softmax_int8_golden_match)
{
    requireGoldenMatch(
        "../../../examples/seq2seq_softmax_int8/output/seq2seq_softmax_int8",
        kGoldenSeq2SeqSoftmax);
}

BOOST_AUTO_TEST_CASE(tiny_generate_int8_golden_match)
{
    requireGoldenMatch(
        "../../../examples/tiny_generate_int8/output/tiny_generate_int8",
        kGoldenTinyGenerate);
}

BOOST_AUTO_TEST_CASE(seq2seq_int8_golden_match)
{
    requireGoldenMatch(
        "../../../examples/seq2seq_int8/output/seq2seq_int8",
        kGoldenSeq2Seq);
}

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
