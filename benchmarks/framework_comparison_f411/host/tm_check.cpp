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

// Host check: run the generated tinymind int8 models over the embedded test
// sets and print the output hashes the firmware expects (fw/backend.hpp).
// Exit status is non-zero if they drift.
#define MLP_DATA_INT8
#define KWS_DATA_INT8
#include "mlp_data.h"
#include "kws_data.h"
#include "rf_data.h"
#include "mlp_tinymind.hpp"
#include "kws_tinymind.hpp"
#include "rf_tinymind.hpp"
#include "backend.hpp"

#include <cstdio>

namespace
{

uint32_t fnv(uint32_t h, const int8_t* p, int n)
{
    for (int i = 0; i < n; ++i)
    {
        h ^= static_cast<uint8_t>(p[i]);
        h *= 0x01000193u;
    }
    return h;
}

} // namespace

int main()
{
    uint32_t mlpHash = 0x811C9DC5u;
    uint32_t mlpHash32 = 0;
    int correct = 0;
    for (int s = 0; s < MLP_N_TEST; ++s)
    {
        if (s == 32)
        {
            mlpHash32 = mlpHash;
        }
        int8_t out[MLP_N_OUT];
        tm_mlp::forward(mlp_test_x_i8 + s * MLP_N_IN, out);
        mlpHash = fnv(mlpHash, out, MLP_N_OUT);
        int best = 0;
        for (int k = 1; k < MLP_N_OUT; ++k)
        {
            if (out[k] > out[best])
            {
                best = k;
            }
        }
        correct += (best == mlp_test_y[s]) ? 1 : 0;
    }

    uint32_t kwsHash = 0x811C9DC5u;
    for (int s = 0; s < KWS_N_TEST; ++s)
    {
        int8_t out[KWS_N_OUT];
        tm_kws::forward(kws_test_x_i8 + s * KWS_N_IN, out);
        kwsHash = fnv(kwsHash, out, KWS_N_OUT);
    }

    uint32_t rfHash = 0x811C9DC5u;
    uint32_t rfHash32 = 0;
    for (int s = 0; s < RF_N_TEST; ++s)
    {
        if (s == 32)
        {
            rfHash32 = rfHash;
        }
        const int8_t cls = static_cast<int8_t>(tm_rf::predict(rf_test_x_i8 + s * RF_N_IN));
        rfHash = fnv(rfHash, &cls, 1);
    }

    std::printf("mlp hash 0x%08X (expect 0x%08X), accuracy %.4f\n",
                mlpHash, TM_MLP_HASH, correct / static_cast<double>(MLP_N_TEST));
    std::printf("mlp hash over first 32 0x%08X (expect 0x%08X)\n", mlpHash32, TM_MLP_HASH_32);
    std::printf("kws hash 0x%08X (expect 0x%08X)\n", kwsHash, TM_KWS_HASH);
    std::printf("rf hash 0x%08X (expect 0x%08X), first 32 0x%08X (expect 0x%08X)\n",
                rfHash, RF_TM_HASH_360, rfHash32, RF_TM_HASH_32);
    return (mlpHash == TM_MLP_HASH && mlpHash32 == TM_MLP_HASH_32 && kwsHash == TM_KWS_HASH
            && rfHash == RF_TM_HASH_360 && rfHash32 == RF_TM_HASH_32) ? 0 : 1;
}
