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

// Maps MODEL_MLP / MODEL_KWS / MODEL_MLPERF_KWS / MODEL_MLPERF_IC to generic
// names used by the backends.
#pragma once

#if defined(MODEL_MLP)
#  include "mlp_data.h"
#  define N_TEST    MLP_N_TEST
#  define N_IN      MLP_N_IN
#  define N_OUT     MLP_N_OUT
#  define LABELS    mlp_test_y
#  if MLP_N_TEST == 360
#    define TM_HASH TM_MLP_HASH
#  elif MLP_N_TEST == 32
#    define TM_HASH TM_MLP_HASH_32
#  else
#    error "no expected tinymind hash for this MLP test-set size (see fw/backend.hpp)"
#  endif
#  define X_I8      mlp_test_x_i8
#  define X_F32     mlp_test_x_f32
#  define MODEL_TAG "mlp"
#elif defined(MODEL_KWS)
#  include "kws_data.h"
#  define N_TEST    KWS_N_TEST
#  define N_IN      KWS_N_IN
#  define N_OUT     KWS_N_OUT
#  define LABELS    nullptr
#  define TM_HASH   TM_KWS_HASH
#  define X_I8      kws_test_x_i8
#  define X_F32     kws_test_x_f32
#  define MODEL_TAG "kws"
#elif defined(MODEL_MLPERF_KWS)
#  include "mlperf_kws_data.h"
#  define N_TEST    MLPERF_KWS_N_TEST
#  define N_IN      MLPERF_KWS_N_IN
#  define N_OUT     MLPERF_KWS_N_OUT
#  define LABELS    nullptr
#  define TM_HASH   TM_MLPERF_KWS_HASH
#  define X_I8      mlperf_kws_test_x_i8
#  define MODEL_TAG "mlperf_kws"
#elif defined(MODEL_MLPERF_IC)
#  include "mlperf_ic_data.h"
#  define N_TEST    MLPERF_IC_N_TEST
#  define N_IN      MLPERF_IC_N_IN
#  define N_OUT     MLPERF_IC_N_OUT
#  define LABELS    nullptr
#  define TM_HASH   TM_MLPERF_IC_HASH
#  define X_I8      mlperf_ic_test_x_i8
#  define MODEL_TAG "mlperf_ic"
#else
#  error "define MODEL_MLP, MODEL_KWS, MODEL_MLPERF_KWS or MODEL_MLPERF_IC"
#endif
