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


// Host check of the MLPerf Tiny reference models (gen/make_mlperf.py): the
// tinymind output hash over the generated test inputs, against the hash
// fw/backend.hpp expects (which TFLM reproduces on the board) and against
// the TFLite interpreter's reference kernels.
#include "mlperf_kws_data.h"
#include "mlperf_ic_data.h"
#include "mlperf_kws_tinymind.hpp"
#include "mlperf_ic_tinymind.hpp"
#include "backend.hpp"

#include <cstdio>

namespace
{

template<typename Forward>
uint32_t hashModel(Forward forward, const int8_t* x, int nTest, int nIn, int nOut)
{
    uint32_t h = 0x811C9DC5u;
    for (int s = 0; s < nTest; ++s)
    {
        int8_t out[16];
        forward(x + s * nIn, out);
        for (int k = 0; k < nOut; ++k)
        {
            h ^= static_cast<uint8_t>(out[k]);
            h *= 0x01000193u;
        }
    }
    return h;
}

} // namespace

int main()
{
    static_assert(MLPERF_KWS_N_OUT <= 16 && MLPERF_IC_N_OUT <= 16, "output buffer");
    const uint32_t kws = hashModel(tm_mlperf_kws::forward, mlperf_kws_test_x_i8,
                                   MLPERF_KWS_N_TEST, MLPERF_KWS_N_IN, MLPERF_KWS_N_OUT);
    const uint32_t ic = hashModel(tm_mlperf_ic::forward, mlperf_ic_test_x_i8,
                                  MLPERF_IC_N_TEST, MLPERF_IC_N_IN, MLPERF_IC_N_OUT);
    std::printf("mlperf kws hash 0x%08X (expect 0x%08X, TFLite reference 0x%08X)\n",
                kws, TM_MLPERF_KWS_HASH, MLPERF_KWS_TFLITE_REF_HASH);
    std::printf("mlperf ic hash 0x%08X (expect 0x%08X, TFLite reference 0x%08X)\n",
                ic, TM_MLPERF_IC_HASH, MLPERF_IC_TFLITE_REF_HASH);
    return (kws == TM_MLPERF_KWS_HASH && ic == TM_MLPERF_IC_HASH) ? 0 : 1;
}
