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

// The self-test every packaging sample runs (CMake, Zephyr, ESP-IDF,
// PlatformIO, Arduino): examples/tflite_streaming_int8's generated models,
// fed its 16-second test recording, must reproduce every output TensorFlow
// Lite for Microcontrollers computed for it -- step() one frame at a time,
// and forward() over the window ending at each frame. A sample that builds
// against a wrongly packaged library fails to compile; one whose flags
// differ from the library's (a feature macro, a lookup-table flag) fails
// here.
//
// The consumer must define TINYMIND_ENABLE_QUANTIZATION=1, the way each
// build system's documentation shows; the generated model #errors without it.

#pragma once

#include <cstddef>
#include <cstdint>

#include "../../examples/tflite_streaming_int8/model.hpp"
#include "../../examples/tflite_streaming_int8/model_stream.hpp"
#include "../../examples/tflite_streaming_int8/reference.hpp"

namespace tinymind_selftest
{
    struct Result
    {
        std::size_t checked;   // int8 outputs compared with TFLM's
        std::size_t differing; // of them, how many differ
    };

    inline Result run()
    {
        namespace W = vibration;
        namespace S = vibration::stream;
        static_assert(S::kModelInputSize * S::kModelWindowFrames == W::kModelInputSize, "frame size");

        alignas(S::kModelStateAlignment) static int8_t state[S::kModelStateSize];
        alignas(S::kModelArenaAlignment) static int8_t streamArena[S::kModelArenaSize];
        alignas(W::kModelArenaAlignment) static int8_t wholeArena[W::kModelArenaSize];
        int8_t streamed[S::kModelOutputSize];
        int8_t whole[W::kModelOutputSize];

        Result result = {0, 0};
        S::reset(state);
        for (std::size_t t = 0; t < W::kFrameCount; ++t)
        {
            S::step(&W::kFrames[t * S::kModelInputSize], streamed, state, streamArena);
            if (t + 1 < S::kModelWindowFrames)
            {
                continue;
            }
            const std::size_t n = t + 1 - S::kModelWindowFrames;
            // The frames are contiguous, so the window ending at frame t is
            // the kModelWindowFrames frames starting at frame n.
            W::forward(&W::kFrames[n * S::kModelInputSize], whole, wholeArena);
            const int8_t* tflm = &W::kTflmOutput[(n > 0 ? n - 1 : 0) * W::kModelOutputSize];
            for (std::size_t c = 0; c < W::kModelOutputSize; ++c)
            {
                result.checked += 2;
                result.differing += (streamed[c] != tflm[c]) ? 1u : 0u;
                result.differing += (whole[c] != tflm[c]) ? 1u : 0u;
            }
        }
        return result;
    }
} // namespace tinymind_selftest
