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

// A streaming int8 CNN that detects a bearing fault in a machine's vibration,
// trained in Keras, converted by TFLiteConverter, and run from two headers
// that apps/import_tflite generated from that one .tflite file:
//
//   model.hpp         forward() over a whole window of 32 frames (0.64 s)
//   model_stream.hpp  step(), which takes one new 20 ms frame and keeps, in a
//                     caller-owned state, the frames each layer still needs
//
// A device that reruns forward() every 20 ms recomputes 31 frames it already
// computed; step() computes one frame per layer. This driver feeds a 16-second
// test recording frame by frame to step(), runs forward() over the window
// ending at each frame as well, and requires both to give, for every window,
// the two int8 outputs TensorFlow Lite for Microcontrollers computed for it.
//
// Built with TINYMIND_ENABLE_FLOAT=0 and TINYMIND_ENABLE_STD=0: the generated
// models need no float and no standard library.
//
//   ./tflite_streaming_int8           agreement with TFLM, detection, CSVs
//   ./tflite_streaming_int8 --golden  step()'s int8 outputs for the first 16 full windows
//   ./tflite_streaming_int8 --bench   time per frame of step() and forward(), as CSV

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>

#include "model.hpp"
#include "model_stream.hpp"
#include "reference.hpp"

namespace
{
    namespace W = vibration;
    namespace S = vibration::stream;

    const char* const kResultsCsv = "tflite_streaming_int8.csv";
    const char* const kSummaryCsv = "tflite_streaming_int8_summary.csv";

    static_assert(S::kModelInputSize * S::kModelWindowFrames == W::kModelInputSize,
                  "a window is kModelWindowFrames frames");
    static_assert(S::kModelOutputSize == W::kModelOutputSize, "one output per window");

    constexpr std::size_t kFrame = S::kModelInputSize;
    constexpr std::size_t kWindowFrames = S::kModelWindowFrames;
    constexpr std::size_t kWindows = W::kFrameCount - kWindowFrames + 1;
    // An alarm that holds for 25 frames (0.5 s) counts as a detection.
    constexpr std::size_t kSustainFrames = 25;

    alignas(S::kModelStateAlignment) int8_t state[S::kModelStateSize];
    alignas(S::kModelArenaAlignment) int8_t streamArena[S::kModelArenaSize];
    alignas(W::kModelArenaAlignment) int8_t wholeArena[W::kModelArenaSize];

    const int8_t* frame(std::size_t t)
    {
        return &W::kFrames[t * kFrame];
    }

    // The softmax output grid is 1/256 with zero point -128.
    int faultPer256(const int8_t* outputs)
    {
        return static_cast<int>(outputs[1]) - W::kModelOutputZeroPoint;
    }

    int golden()
    {
        std::printf("# tflite_streaming_int8 golden output: step()'s int8 outputs for the first 16 full windows\n");
        int8_t outputs[S::kModelOutputSize];
        S::reset(state);
        for (std::size_t t = 0; t < kWindowFrames - 1 + 16; ++t)
        {
            S::step(frame(t), outputs, state, streamArena);
            if (t + 1 >= kWindowFrames)
            {
                std::printf("%d %d\n", static_cast<int>(outputs[0]), static_cast<int>(outputs[1]));
            }
        }
        return 0;
    }

    int bench()
    {
        int8_t outputs[S::kModelOutputSize];
        int checksum = 0;
        const int passes = 200;
        S::reset(state);
        const std::chrono::steady_clock::time_point s0 = std::chrono::steady_clock::now();
        for (int pass = 0; pass < passes; ++pass)
        {
            for (std::size_t t = 0; t < W::kFrameCount; ++t)
            {
                S::step(frame(t), outputs, state, streamArena);
                checksum += outputs[1];
            }
        }
        const double streamUs = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - s0).count();
        const std::chrono::steady_clock::time_point w0 = std::chrono::steady_clock::now();
        for (int pass = 0; pass < passes; ++pass)
        {
            for (std::size_t n = 0; n < kWindows; ++n)
            {
                // The frames are contiguous, so the window ending at frame n + 31
                // is the 32 frames starting at frame n.
                W::forward(frame(n), outputs, wholeArena);
                checksum += outputs[1];
            }
        }
        const double wholeUs = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - w0).count();
        std::printf("metric,value\n");
        std::printf("us_per_step,%.4f\n", streamUs / (passes * static_cast<double>(W::kFrameCount)));
        std::printf("us_per_forward,%.4f\n", wholeUs / (passes * static_cast<double>(kWindows)));
        std::printf("checksum,%d\n", checksum);
        return 0;
    }
} // namespace

int main(int argc, char* argv[])
{
    if (argc > 1 && std::strcmp(argv[1], "--golden") == 0)
    {
        return golden();
    }
    if (argc > 1 && std::strcmp(argv[1], "--bench") == 0)
    {
        return bench();
    }

    std::FILE* results = std::fopen(kResultsCsv, "w");
    if (results == nullptr)
    {
        std::fprintf(stderr, "cannot write %s\n", kResultsCsv);
        return 1;
    }
    std::fprintf(results, "frame,seconds,severity_percent,fault_per256,valid");
    for (std::size_t b = 0; b < kFrame; ++b)
    {
        std::fprintf(results, ",band%zu", b);
    }
    std::fprintf(results, "\n");

    // Garbage in the scratch arena before every step: a step must not depend
    // on the arena's contents from the step before. Only the state carries
    // anything between steps.
    S::reset(state);
    std::size_t stepDiffering = 0;
    std::size_t wholeDiffering = 0;
    std::size_t falseAlarms = 0;
    std::size_t healthyWindows = 0;
    std::size_t detected = 0;
    std::size_t faultyWindows = 0;
    long firstAlarm = -1;
    long firstSustained = -1;
    std::size_t run = 0;
    int8_t streamed[S::kModelOutputSize];
    int8_t whole[W::kModelOutputSize];
    for (std::size_t t = 0; t < W::kFrameCount; ++t)
    {
        std::memset(streamArena, 0x5A, sizeof(streamArena));
        S::step(frame(t), streamed, state, streamArena);
        const bool valid = t + 1 >= S::kModelReceptiveFrames;
        if (valid)
        {
            const std::size_t n = t + 1 - kWindowFrames;
            W::forward(frame(n), whole, wholeArena);
            const int8_t* tflm = &W::kTflmOutput[n * W::kModelOutputSize];
            for (std::size_t c = 0; c < W::kModelOutputSize; ++c)
            {
                stepDiffering += (streamed[c] != tflm[c]) ? 1u : 0u;
                wholeDiffering += (whole[c] != tflm[c]) ? 1u : 0u;
            }
            const bool alarm = faultPer256(streamed) > 128;
            if (W::kSeverityPercent[t] == 0)
            {
                ++healthyWindows;
                falseAlarms += alarm ? 1u : 0u;
            }
            else
            {
                ++faultyWindows;
                detected += alarm ? 1u : 0u;
                if (alarm && firstAlarm < 0)
                {
                    firstAlarm = static_cast<long>(t);
                }
            }
            run = alarm ? run + 1 : 0;
            if (run == kSustainFrames && firstSustained < 0)
            {
                firstSustained = static_cast<long>(t + 1 - kSustainFrames);
            }
        }
        const double seconds = static_cast<double>(t * W::kHopSamples + W::kWindowSamples) / W::kSampleRate;
        std::fprintf(results, "%zu,%.3f,%d,%d,%d", t, seconds, static_cast<int>(W::kSeverityPercent[t]),
                     faultPer256(streamed), valid ? 1 : 0);
        for (std::size_t b = 0; b < kFrame; ++b)
        {
            std::fprintf(results, ",%d", static_cast<int>(frame(t)[b]));
        }
        std::fprintf(results, "\n");
    }
    std::fclose(results);

    const auto seconds = [](long t) {
        return t < 0 ? -1.0
                     : static_cast<double>(static_cast<std::size_t>(t) * W::kHopSamples + W::kWindowSamples) /
                           W::kSampleRate;
    };
    const auto severity = [](long t) {
        return t < 0 ? -1 : static_cast<int>(W::kSeverityPercent[t]);
    };
    const double alarmSeconds = seconds(firstAlarm);
    const double sustainedSeconds = seconds(firstSustained);
    std::printf("frames:                         %zu (%zu windows of %zu frames)\n", W::kFrameCount, kWindows,
                kWindowFrames);
    std::printf("step() outputs differing from TFLM:    %zu of %zu\n", stepDiffering, kWindows * W::kModelOutputSize);
    std::printf("forward() outputs differing from TFLM: %zu of %zu\n", wholeDiffering,
                kWindows * W::kModelOutputSize);
    std::printf("false alarms:                   %zu of %zu healthy windows\n", falseAlarms, healthyWindows);
    std::printf("fault detected:                 %zu of %zu faulty windows\n", detected, faultyWindows);
    std::printf("first alarm:                    %.2f s (severity %d%%)\n", alarmSeconds, severity(firstAlarm));
    std::printf("first alarm held for 0.5 s:     %.2f s (severity %d%%)\n", sustainedSeconds,
                severity(firstSustained));
    std::printf("memory, forward():              %zu B input window + %zu B arena\n", W::kModelInputSize,
                W::kModelArenaSize);
    std::printf("memory, step():                 %zu B frame + %zu B arena + %zu B state\n", S::kModelInputSize,
                S::kModelArenaSize, S::kModelStateSize);
    std::printf("weights:                        %zu bytes, shared by both\n", W::kModelWeightBytes);

    std::FILE* summary = std::fopen(kSummaryCsv, "w");
    if (summary == nullptr)
    {
        std::fprintf(stderr, "cannot write %s\n", kSummaryCsv);
        return 1;
    }
    std::fprintf(summary, "metric,value\n");
    std::fprintf(summary, "frames,%zu\nwindows,%zu\nwindow_frames,%zu\n", W::kFrameCount, kWindows, kWindowFrames);
    std::fprintf(summary, "step_outputs_differing,%zu\nforward_outputs_differing,%zu\n", stepDiffering,
                 wholeDiffering);
    std::fprintf(summary, "false_alarms,%zu\nhealthy_windows,%zu\ndetected,%zu\nfaulty_windows,%zu\n", falseAlarms,
                 healthyWindows, detected, faultyWindows);
    std::fprintf(summary, "first_alarm_seconds,%.3f\nfirst_alarm_severity,%d\n", alarmSeconds, severity(firstAlarm));
    std::fprintf(summary, "sustained_alarm_seconds,%.3f\nsustained_alarm_severity,%d\n", sustainedSeconds,
                 severity(firstSustained));
    std::fprintf(summary, "window_bytes,%zu\nwhole_arena_bytes,%zu\n", W::kModelInputSize, W::kModelArenaSize);
    std::fprintf(summary, "frame_bytes,%zu\nstream_arena_bytes,%zu\nstate_bytes,%zu\n", S::kModelInputSize,
                 S::kModelArenaSize, S::kModelStateSize);
    std::fprintf(summary, "weight_bytes,%zu\n", W::kModelWeightBytes);
    std::fclose(summary);
    std::printf("wrote %s and %s\n", kResultsCsv, kSummaryCsv);
    return (stepDiffering == 0 && wholeDiffering == 0) ? 0 : 1;
}
