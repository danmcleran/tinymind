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

/* emlearn integer random forest (gen/make_trees.py), int16 features.
 * EML_RF_LOADABLE selects the 'loadable' method (EmlTrees data walked by
 * eml_trees_predict, which tallies votes in float); otherwise the default
 * 'inline' method (one generated if/else function per tree, int votes). */
#include "backend.hpp"
#include "rf_data.h"
/* emlearn's generated inline trees never read their features_length parameter. */
#pragma GCC diagnostic ignored "-Wunused-parameter"
#ifdef EML_RF_LOADABLE
#  include "rf_emlearn_loadable.h"
#  define RF_PREDICT rf_loadable_predict
#  define RF_TAG "loadable"
#else
#  include "rf_emlearn_inline.h"
#  define RF_PREDICT rf_inline_predict
#  define RF_TAG "inline"
#endif

#if RF_N_TEST == 360
#  define RF_EML_HASH RF_EML_HASH_360
#elif RF_N_TEST == 32
#  define RF_EML_HASH RF_EML_HASH_32
#else
#  error "no expected hash for this RF test-set size (see gen/make_trees.py)"
#endif

static int16_t gIn[RF_N_IN];
static int32_t gClass;

const char* const be_name = "emlearn_int16_" RF_TAG "/rf";
const int be_n_test = RF_N_TEST;
const uint8_t* const be_labels = rf_test_y;
const uint32_t be_expect_hash = RF_EML_HASH;

int be_setup(void)
{
    return 0;
}

void be_load(int s)
{
    for (int i = 0; i < RF_N_IN; ++i)
    {
        gIn[i] = rf_test_x_i8[s * RF_N_IN + i];
    }
}

__attribute__((noinline)) void be_invoke(void)
{
    gClass = RF_PREDICT(gIn, RF_N_IN);
}

int be_argmax(void)
{
    return (int)gClass;
}

uint32_t be_hash(uint32_t h)
{
    h ^= (uint8_t)gClass;
    h *= 0x01000193u;
    return h;
}

uint32_t be_arena_used(void)
{
    return 0;
}

int be_profile(uint32_t* c, const char** t, int m)
{
    (void)c;
    (void)t;
    (void)m;
    return 0;
}
