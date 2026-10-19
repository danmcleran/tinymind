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

// TFLM backend: MicroInterpreter with a MicroMutableOpResolver holding only the
// operators the model needs. -DPROFILE adds per-op DWT timing through TFLM's
// profiler hook (needs a release_with_logs lib); -DARENA_PROBE bisects the
// smallest tensor arena AllocateTensors() accepts.
#include "backend.hpp"
#include "board.hpp"
#if defined(DTYPE_F32)
#  define MLP_DATA_F32
#  define KWS_DATA_F32
#else
#  define MLP_DATA_INT8
#  define KWS_DATA_INT8
#endif
#include "model_select.hpp"

#if defined(MODEL_MLP) && defined(DTYPE_F32)
#  include "mlp_tflite_f32.h"
#  define MODEL_BLOB mlp_f32_tflite
#elif defined(MODEL_MLP)
#  include "mlp_tflite_int8.h"
#  define MODEL_BLOB mlp_int8_tflite
#elif defined(MODEL_MLPERF_KWS)
#  include "mlperf_kws_tflite_int8.h"
#  define MODEL_BLOB mlperf_kws_int8_tflite
#  define MODEL_MLPERF
#elif defined(MODEL_MLPERF_IC)
#  include "mlperf_ic_tflite_int8.h"
#  define MODEL_BLOB mlperf_ic_int8_tflite
#  define MODEL_MLPERF
#elif defined(DTYPE_F32)
#  include "kws_tflite_f32.h"
#  define MODEL_BLOB kws_f32_tflite
#else
#  include "kws_tflite_int8.h"
#  define MODEL_BLOB kws_int8_tflite
#endif

#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_profiler_interface.h"
#include "tensorflow/lite/micro/schema/schema_generated.h"
#include <string.h>
#include <new>

#ifndef ARENA_BYTES
#define ARENA_BYTES 32768
#endif
#ifdef DTYPE_F32
#  define DT_TAG "f32"
#else
#  define DT_TAG "int8"
#endif
#ifndef KERNELS_TAG
#define KERNELS_TAG ""
#endif

namespace
{

#ifdef PROFILE
class DwtProfiler : public tflite::MicroProfilerInterface
{
public:
    bool active = false;
    int n = 0;
    const char* tags[32];
    uint32_t start[32];
    uint32_t cycles[32];

    uint32_t BeginEvent(const char* tag) override
    {
        if (!active || n >= 32)
        {
            return 0;
        }
        tags[n] = tag;
        start[n] = board_cycles();
        return static_cast<uint32_t>(n++);
    }

    void EndEvent(uint32_t handle) override
    {
        if (active)
        {
            cycles[handle] = board_cycles() - start[handle];
        }
    }
};
DwtProfiler gProfiler;
#  define PROFILER_PTR (&gProfiler)
#else
#  define PROFILER_PTR nullptr
#endif

alignas(16) uint8_t gArena[ARENA_BYTES];
#ifdef ARENA_PROBE
uint32_t gArenaMin = 0;
#endif
#if defined(MODEL_MLP)
using Resolver = tflite::MicroMutableOpResolver<1>;
#elif defined(MODEL_MLPERF)
using Resolver = tflite::MicroMutableOpResolver<6>;
#else
using Resolver = tflite::MicroMutableOpResolver<5>;
#endif
alignas(Resolver) uint8_t gResolverMem[sizeof(Resolver)];
alignas(tflite::MicroInterpreter) uint8_t gInterpMem[sizeof(tflite::MicroInterpreter)];
tflite::MicroInterpreter* gInterp = nullptr;
TfLiteTensor* gIn = nullptr;
TfLiteTensor* gOut = nullptr;

} // namespace

extern "C"
{

const char* const be_name = "tflm_" DT_TAG KERNELS_TAG "/" MODEL_TAG;
const int be_n_test = N_TEST;
const uint8_t* const be_labels = LABELS;
#ifdef DTYPE_F32
const uint32_t be_expect_hash = 0;
#else
const uint32_t be_expect_hash = TM_HASH;
#endif

int be_setup(void)
{
    const tflite::Model* model = tflite::GetModel(MODEL_BLOB);
    Resolver* resolver = new (gResolverMem) Resolver();
    resolver->AddFullyConnected();
#if defined(MODEL_KWS)
    resolver->AddConv2D();
    resolver->AddMaxPool2D();
    resolver->AddDepthwiseConv2D();
    resolver->AddAveragePool2D();
#elif defined(MODEL_MLPERF)
    // Exactly the reference graph's operators, so no unused kernel is linked.
    resolver->AddConv2D();
#  if defined(MODEL_MLPERF_KWS)
    resolver->AddDepthwiseConv2D();
#  else
    resolver->AddAdd();
#  endif
    resolver->AddAveragePool2D();
    resolver->AddReshape();
    resolver->AddSoftmax();
#endif
#ifdef ARENA_PROBE
    // Bisect the smallest arena (16-byte steps) AllocateTensors accepts.
    {
        uint32_t lo = 1;
        uint32_t hi = ARENA_BYTES / 16;
        while (lo < hi)
        {
            const uint32_t mid = (lo + hi) / 2;
            tflite::MicroInterpreter* probe =
                new (gInterpMem) tflite::MicroInterpreter(model, *resolver, gArena, mid * 16u);
            const bool ok = probe->AllocateTensors() == kTfLiteOk;
            probe->~MicroInterpreter();
            if (ok)
            {
                hi = mid;
            }
            else
            {
                lo = mid + 1;
            }
        }
        gArenaMin = lo * 16u;
    }
#endif
    gInterp = new (gInterpMem) tflite::MicroInterpreter(model, *resolver, gArena, ARENA_BYTES,
                                                         nullptr, PROFILER_PTR);
    if (gInterp->AllocateTensors() != kTfLiteOk)
    {
        return 1;
    }
    gIn = gInterp->input(0);
    gOut = gInterp->output(0);
    return 0;
}

void be_load(int s)
{
#ifdef DTYPE_F32
    memcpy(gIn->data.f, X_F32 + s * N_IN, N_IN * sizeof(float));
#else
    memcpy(gIn->data.int8, X_I8 + s * N_IN, N_IN);
#endif
}

__attribute__((noinline)) void be_invoke(void)
{
    gInterp->Invoke();
}

int be_argmax(void)
{
    int best = 0;
    for (int k = 1; k < N_OUT; ++k)
    {
#ifdef DTYPE_F32
        if (gOut->data.f[k] > gOut->data.f[best])
#else
        if (gOut->data.int8[k] > gOut->data.int8[best])
#endif
        {
            best = k;
        }
    }
    return best;
}

uint32_t be_hash(uint32_t h)
{
#ifdef DTYPE_F32
    h ^= static_cast<uint8_t>(be_argmax());
    h *= 0x01000193u;
#else
    for (int k = 0; k < N_OUT; ++k)
    {
        h ^= static_cast<uint8_t>(gOut->data.int8[k]);
        h *= 0x01000193u;
    }
#endif
    return h;
}

uint32_t be_arena_used(void)
{
#ifdef ARENA_PROBE
    return gArenaMin;
#else
    return static_cast<uint32_t>(gInterp->arena_used_bytes());
#endif
}

int be_profile(uint32_t* c, const char** t, int max)
{
#ifdef PROFILE
    gProfiler.n = 0;
    gProfiler.active = true;
    gInterp->Invoke();
    gProfiler.active = false;
    int k = 0;
    for (; k < gProfiler.n && k < max; ++k)
    {
        c[k] = gProfiler.cycles[k];
        t[k] = gProfiler.tags[k];
    }
    return k;
#else
    (void)c;
    (void)t;
    (void)max;
    return 0;
#endif
}

} // extern "C"
