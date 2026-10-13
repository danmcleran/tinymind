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

/* emlearn backend: eml_net float MLP. The generated header's last activation is
 * patched from softmax to identity (Makefile rule) so it returns logits like the
 * other backends; argmax is unchanged. */
#include "backend.hpp"
#define MLP_DATA_F32
#include "mlp_data.h"
#include "mlp_emlearn_logits.h"
#include <string.h>

static float gIn[MLP_N_IN];

const char* const be_name = "emlearn_f32/mlp";
const int be_n_test = MLP_N_TEST;
const uint8_t* const be_labels = mlp_test_y;
const uint32_t be_expect_hash = 0;

int be_setup(void)
{
    return 0;
}

void be_load(int s)
{
    memcpy(gIn, mlp_test_x_f32 + s * MLP_N_IN, sizeof(gIn));
}

__attribute__((noinline)) void be_invoke(void)
{
    eml_net_infer(&mlp, gIn, MLP_N_IN);
}

int be_argmax(void)
{
    return eml_net_argmax(mlp.activations2, MLP_N_OUT);
}

uint32_t be_hash(uint32_t h)
{
    h ^= (uint8_t)be_argmax();
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
