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

// tinymind integer random forest: tinymind::QDecisionTree per tree over a
// QTreeNode<int8_t> pool, majority vote (gen/make_trees.py). const data in
// flash, no setup step.
//
// -DTM_RF_INLINE: the same pool as tinymind::QStaticDecisionTree with the
// QTreeInlineEval policy (rf_tinymind_inline.hpp), expanded at compile time
// instead of walked.
// -DTM_RF_U8 (with TM_RF_INLINE): read the features as uint8_t. They are
// 0..32, so the value is unchanged; Thumb-1 has an immediate-offset LDRB but
// only a register-offset LDRSB.
#include "backend.hpp"
#include "rf_data.h"
#if defined(TM_RF_INLINE)
#  include "rf_tinymind_inline.hpp"
#else
#  include "rf_tinymind.hpp"
#endif
#include <string.h>

#if RF_N_TEST == 360
#  define RF_TM_HASH RF_TM_HASH_360
#elif RF_N_TEST == 32
#  define RF_TM_HASH RF_TM_HASH_32
#else
#  error "no expected hash for this RF test-set size (see gen/make_trees.py)"
#endif

namespace
{

int8_t gIn[RF_N_IN];
int gClass;

} // namespace

extern "C"
{

#if defined(TM_RF_INLINE) && defined(TM_RF_U8)
const char* const be_name = "tinymind_uint8/rf_inline";
#elif defined(TM_RF_INLINE)
const char* const be_name = "tinymind_int8/rf_inline";
#else
const char* const be_name = "tinymind_int8/rf";
#endif
const int be_n_test = RF_N_TEST;
const uint8_t* const be_labels = rf_test_y;
const uint32_t be_expect_hash = RF_TM_HASH;

int be_setup(void)
{
    return 0;
}

void be_load(int s)
{
    memcpy(gIn, rf_test_x_i8 + s * RF_N_IN, RF_N_IN);
}

__attribute__((noinline)) void be_invoke(void)
{
#if defined(TM_RF_INLINE) && defined(TM_RF_U8)
    gClass = tm_rf_inline::predict(reinterpret_cast<const uint8_t*>(gIn));
#elif defined(TM_RF_INLINE)
    gClass = tm_rf_inline::predict(gIn);
#else
    gClass = tm_rf::predict(gIn);
#endif
}

int be_argmax(void)
{
    return gClass;
}

uint32_t be_hash(uint32_t h)
{
    h ^= static_cast<uint8_t>(gClass);
    h *= 0x01000193u;
    return h;
}

uint32_t be_arena_used(void)
{
    return 0;
}

int be_profile(uint32_t*, const char**, int)
{
    return 0;
}

} // extern "C"
