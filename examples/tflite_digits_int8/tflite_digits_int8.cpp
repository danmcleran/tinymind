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

// A residual int8 CNN on the UCI optical digits, trained in Keras, converted
// to an int8 TensorFlow Lite model by TFLiteConverter, and run from the
// model.hpp that apps/import_tflite generated from that .tflite file. The
// layer objects, the activation memory plan and forward() are all generated;
// this file only reads the test set, quantizes each digit through the input
// table, calls forward(), and checks every output against the one TensorFlow
// Lite for Microcontrollers computed for the same digit.
//
// Built with TINYMIND_ENABLE_FLOAT=0 and TINYMIND_ENABLE_STD=0: the generated
// model needs no float and no standard library.
//
//   ./tflite_digits_int8            accuracy, agreement with TFLM, CSVs
//   ./tflite_digits_int8 --golden   int8 outputs of the first 16 test digits
//   ./tflite_digits_int8 --bench    inference time, as CSV

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>

#include "model.hpp"
#include "reference.hpp"

namespace
{
    const char* const kTestSet = "../../optical_digits/optdigits.tes";
    const char* const kResultsCsv = "tflite_digits_int8.csv";
    const char* const kPlanCsv = "tflite_digits_int8_plan.csv";
    const char* const kSummaryCsv = "tflite_digits_int8_summary.csv";

    alignas(digits::kModelArenaAlignment) int8_t arena[digits::kModelArenaSize];

    int8_t images[digits::kTestCount][digits::kModelInputSize];
    uint8_t pixels[digits::kTestCount][digits::kModelInputSize];
    int labels[digits::kTestCount];

    bool loadTestSet()
    {
        std::FILE* f = std::fopen(kTestSet, "r");
        if (f == nullptr)
        {
            std::fprintf(stderr, "cannot open %s\n", kTestSet);
            return false;
        }
        for (std::size_t n = 0; n < digits::kTestCount; ++n)
        {
            for (std::size_t p = 0; p < digits::kModelInputSize; ++p)
            {
                int v = 0;
                if (std::fscanf(f, "%d,", &v) != 1 || v < 0 || v > 16)
                {
                    std::fclose(f);
                    std::fprintf(stderr, "%s: bad pixel in row %zu\n", kTestSet, n);
                    return false;
                }
                pixels[n][p] = static_cast<uint8_t>(v);
                // One input channel, so the row-major 8x8 image is already NHWC.
                images[n][p] = digits::kPixelToInput[v];
            }
            if (std::fscanf(f, "%d", &labels[n]) != 1)
            {
                std::fclose(f);
                std::fprintf(stderr, "%s: missing label in row %zu\n", kTestSet, n);
                return false;
            }
        }
        std::fclose(f);
        return true;
    }

    int argmax(const int8_t* outputs)
    {
        int best = 0;
        for (std::size_t c = 1; c < digits::kModelOutputSize; ++c)
        {
            if (outputs[c] > outputs[best])
            {
                best = static_cast<int>(c);
            }
        }
        return best;
    }

    int golden()
    {
        std::printf("# tflite_digits_int8 golden output: int8 outputs of the first 16 test digits\n");
        int8_t outputs[digits::kModelOutputSize];
        for (std::size_t n = 0; n < 16; ++n)
        {
            digits::forward(images[n], outputs, arena);
            for (std::size_t c = 0; c < digits::kModelOutputSize; ++c)
            {
                std::printf("%d%c", static_cast<int>(outputs[c]), c + 1 < digits::kModelOutputSize ? ' ' : '\n');
            }
        }
        return 0;
    }

    int bench()
    {
        int8_t outputs[digits::kModelOutputSize];
        int checksum = 0;
        const int passes = 20;
        const std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
        for (int pass = 0; pass < passes; ++pass)
        {
            for (std::size_t n = 0; n < digits::kTestCount; ++n)
            {
                digits::forward(images[n], outputs, arena);
                checksum += outputs[0];
            }
        }
        const double us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - start).count();
        std::printf("metric,value\n");
        std::printf("inferences,%d\n", passes * static_cast<int>(digits::kTestCount));
        std::printf("us_per_inference,%.3f\n", us / (passes * static_cast<double>(digits::kTestCount)));
        std::printf("arena_bytes,%zu\n", digits::kModelArenaSize);
        std::printf("checksum,%d\n", checksum);
        return 0;
    }
} // namespace

int main(int argc, char* argv[])
{
    if (!loadTestSet())
    {
        return 1;
    }
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
    std::fprintf(results, "index,label,tinymind,tflm,keras,matches_tflm,confidence");
    for (std::size_t p = 0; p < digits::kModelInputSize; ++p)
    {
        std::fprintf(results, ",p%zu", p);
    }
    std::fprintf(results, "\n");

    std::size_t correct = 0;
    std::size_t correctTflm = 0;
    std::size_t correctKeras = 0;
    std::size_t sameAsTflm = 0;
    std::size_t outputsDiffering = 0;
    std::size_t sameClassAsKeras = 0;
    int8_t outputs[digits::kModelOutputSize];
    for (std::size_t n = 0; n < digits::kTestCount; ++n)
    {
        digits::forward(images[n], outputs, arena);
        const int8_t* tflm = &digits::kTflmOutput[n * digits::kModelOutputSize];
        std::size_t differing = 0;
        for (std::size_t c = 0; c < digits::kModelOutputSize; ++c)
        {
            differing += (outputs[c] != tflm[c]) ? 1u : 0u;
        }
        outputsDiffering += differing;
        sameAsTflm += (differing == 0) ? 1u : 0u;
        const int predicted = argmax(outputs);
        const int predictedTflm = argmax(tflm);
        const int keras = digits::kKerasPrediction[n];
        correct += (predicted == labels[n]) ? 1u : 0u;
        correctTflm += (predictedTflm == labels[n]) ? 1u : 0u;
        correctKeras += (keras == labels[n]) ? 1u : 0u;
        sameClassAsKeras += (predicted == keras) ? 1u : 0u;
        // The softmax output grid is 1/256 with zero point -128.
        const int confidence = static_cast<int>(outputs[predicted]) - digits::kModelOutputZeroPoint;
        std::fprintf(results, "%zu,%d,%d,%d,%d,%d,%d", n, labels[n], predicted, predictedTflm, keras,
                     differing == 0 ? 1 : 0, confidence);
        for (std::size_t p = 0; p < digits::kModelInputSize; ++p)
        {
            std::fprintf(results, ",%d", static_cast<int>(pixels[n][p]));
        }
        std::fprintf(results, "\n");
    }
    std::fclose(results);

    std::FILE* plan = std::fopen(kPlanCsv, "w");
    if (plan == nullptr)
    {
        std::fprintf(stderr, "cannot write %s\n", kPlanCsv);
        return 1;
    }
    std::fprintf(plan, "offset,size,first_layer,last_layer,outputs\n");
    for (std::size_t b = 0; b < digits::kModelBufferCount; ++b)
    {
        const digits::ModelBuffer& buf = digits::kModelBuffers[b];
        std::fprintf(plan, "%zu,%zu,%zu,%zu,%s\n", buf.offset, buf.size, buf.first_layer, buf.last_layer,
                     buf.outputs);
    }
    std::fclose(plan);

    std::FILE* summary = std::fopen(kSummaryCsv, "w");
    if (summary == nullptr)
    {
        std::fprintf(stderr, "cannot write %s\n", kSummaryCsv);
        return 1;
    }
    const double total = static_cast<double>(digits::kTestCount);
    std::fprintf(summary, "metric,value\n");
    std::fprintf(summary, "layers,%zu\n", digits::kModelLayerCount);
    std::fprintf(summary, "arena_bytes,%zu\n", digits::kModelArenaSize);
    std::fprintf(summary, "unshared_bytes,%zu\n", digits::kModelUnsharedSize);
    std::fprintf(summary, "busiest_layer_bytes,%zu\n", digits::kModelPeakLiveSize);
    std::fprintf(summary, "weight_bytes,%zu\n", digits::kModelWeightBytes);
    std::fprintf(summary, "accuracy,%.4f\n", correct / total);
    std::fprintf(summary, "accuracy_tflm,%.4f\n", correctTflm / total);
    std::fprintf(summary, "accuracy_keras,%.4f\n", correctKeras / total);
    std::fprintf(summary, "digits_matching_tflm,%zu\n", sameAsTflm);
    std::fprintf(summary, "outputs_differing_from_tflm,%zu\n", outputsDiffering);
    std::fprintf(summary, "same_class_as_keras,%zu\n", sameClassAsKeras);
    std::fclose(summary);

    std::printf("test digits:                    %zu\n", digits::kTestCount);
    std::printf("tinymind int8 accuracy:         %.4f\n", correct / total);
    std::printf("TFLM int8 accuracy:             %.4f\n", correctTflm / total);
    std::printf("float (Keras) accuracy:         %.4f\n", correctKeras / total);
    std::printf("same 10 outputs as TFLM:        %zu of %zu digits (%zu of %zu outputs differ)\n", sameAsTflm,
                digits::kTestCount, outputsDiffering, digits::kTestCount * digits::kModelOutputSize);
    std::printf("same class as Keras:            %zu of %zu\n", sameClassAsKeras, digits::kTestCount);
    std::printf("activation arena:               %zu bytes (unshared %zu, busiest layer %zu)\n",
                digits::kModelArenaSize, digits::kModelUnsharedSize, digits::kModelPeakLiveSize);
    std::printf("weights:                        %zu bytes\n", digits::kModelWeightBytes);
    std::printf("wrote %s, %s and %s\n", kResultsCsv, kPlanCsv, kSummaryCsv);
    // The generated model must compute exactly what TFLM computes.
    return outputsDiffering == 0 ? 0 : 1;
}
