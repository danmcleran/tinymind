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

// A residual int8 CNN on the UCI optical digits, run from the model.hpp that
// apps/import_onnx/tinymind_model_codegen.py generated from an ONNX Runtime
// quantized model (see train.py). The model, its weights and its memory plan
// are all generated; this file only reads the test set, quantizes each digit
// through the input table, and calls forward().
//
// Built with TINYMIND_ENABLE_FLOAT=0 and TINYMIND_ENABLE_STD=0: the generated
// model needs no float and no standard library.
//
//   ./onnx_digits_int8            accuracy, agreement with ONNX Runtime, CSVs
//   ./onnx_digits_int8 --golden   int8 logits of the first 16 test digits
//   ./onnx_digits_int8 --bench    inference time, as CSV

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>

#include "model.hpp"
#include "model_int4.hpp"
#include "model_int16x8.hpp"
#include "reference.hpp"

namespace
{
    const char* const kTestSet = "../../optical_digits/optdigits.tes";
    const char* const kResultsCsv = "onnx_digits_int8.csv";
    const char* const kPlanCsv = "onnx_digits_int8_plan.csv";
    const char* const kSummaryCsv = "onnx_digits_int8_summary.csv";
    const char* const kLogitsCsv = "onnx_digits_int8_logits.csv";

    alignas(digits::kModelArenaAlignment) int8_t arena[digits::kModelArenaSize];
    alignas(digits::int16x8::kModelArenaAlignment) int16_t arena16[digits::int16x8::kModelArenaSize / sizeof(int16_t)];

    int8_t images[digits::kTestCount][digits::kModelInputSize];
    int16_t images16[digits::kTestCount][digits::kModelInputSize];
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
                images16[n][p] = digits::kPixelToInput16[v];
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

    template<typename T>
    int argmax(const T* logits)
    {
        int best = 0;
        for (std::size_t c = 1; c < digits::kModelOutputSize; ++c)
        {
            if (logits[c] > logits[best])
            {
                best = static_cast<int>(c);
            }
        }
        return best;
    }

    int golden()
    {
        std::printf("# onnx_digits_int8 golden output: int8 logits of the first 16 test digits\n");
        int8_t logits[digits::kModelOutputSize];
        for (std::size_t n = 0; n < 16; ++n)
        {
            digits::forward(images[n], logits, arena);
            for (std::size_t c = 0; c < digits::kModelOutputSize; ++c)
            {
                std::printf("%d%c", static_cast<int>(logits[c]), c + 1 < digits::kModelOutputSize ? ' ' : '\n');
            }
        }
        return 0;
    }

    int bench()
    {
        int8_t logits[digits::kModelOutputSize];
        int checksum = 0;
        const int passes = 20;
        const std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
        for (int pass = 0; pass < passes; ++pass)
        {
            for (std::size_t n = 0; n < digits::kTestCount; ++n)
            {
                digits::forward(images[n], logits, arena);
                checksum += logits[0];
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
    std::fprintf(results, "index,label,tinymind,tinymind_int4,tinymind_16x8,onnxruntime,float");
    for (std::size_t p = 0; p < digits::kModelInputSize; ++p)
    {
        std::fprintf(results, ",p%zu", p);
    }
    std::fprintf(results, "\n");

    std::size_t correct = 0;
    std::size_t correctInt4 = 0;
    std::size_t agreeInt4 = 0;
    std::size_t agreeOrt = 0;
    std::size_t correctOrt = 0;
    std::size_t correctFloat = 0;
    std::size_t correct16 = 0;
    std::size_t agreeFloat = 0;
    std::size_t agreeFloat16 = 0;
    double error8 = 0.0;
    double error16 = 0.0;
    double worst8 = 0.0;
    double worst16 = 0.0;
    std::FILE* logitsCsv = std::fopen(kLogitsCsv, "w");
    if (logitsCsv == nullptr)
    {
        std::fprintf(stderr, "cannot write %s\n", kLogitsCsv);
        return 1;
    }
    std::fprintf(logitsCsv, "index,class,reference,int8,int16x8\n");
    int8_t logits[digits::kModelOutputSize];
    int16_t logits16[digits::kModelOutputSize];
    for (std::size_t n = 0; n < digits::kTestCount; ++n)
    {
        digits::forward(images[n], logits, arena);
        const int predicted = argmax(logits);
        digits::int16x8::forward(images16[n], logits16, arena16);
        const int predicted16 = argmax(logits16);
        correct16 += (predicted16 == labels[n]) ? 1u : 0u;
        agreeFloat += (predicted == digits::kFloatPrediction[n]) ? 1u : 0u;
        agreeFloat16 += (predicted16 == digits::kFloatPrediction[n]) ? 1u : 0u;
        if (n < digits::kReferenceLogitCount)
        {
            // Logits in real units, against the int8-weight model run in float.
            for (std::size_t c = 0; c < digits::kModelOutputSize; ++c)
            {
                const double ref = digits::kReferenceLogits[n * digits::kModelOutputSize + c];
                const double v8 = (logits[c] - digits::kModelOutputZeroPoint) *
                                  static_cast<double>(digits::kModelOutputScale);
                const double v16 = logits16[c] * static_cast<double>(digits::int16x8::kModelOutputScale);
                const double e8 = (v8 > ref) ? v8 - ref : ref - v8;
                const double e16 = (v16 > ref) ? v16 - ref : ref - v16;
                error8 += e8;
                error16 += e16;
                worst8 = (e8 > worst8) ? e8 : worst8;
                worst16 = (e16 > worst16) ? e16 : worst16;
                std::fprintf(logitsCsv, "%zu,%zu,%.6f,%.6f,%.6f\n", n, c, ref, v8, v16);
            }
        }
        digits::int4::forward(images[n], logits, arena);
        const int predictedInt4 = argmax(logits);
        correctInt4 += (predictedInt4 == labels[n]) ? 1u : 0u;
        agreeInt4 += (predictedInt4 == predicted) ? 1u : 0u;
        const int ort = digits::kOrtPrediction[n];
        const int flt = digits::kFloatPrediction[n];
        correct += (predicted == labels[n]) ? 1u : 0u;
        agreeOrt += (predicted == ort) ? 1u : 0u;
        correctOrt += (ort == labels[n]) ? 1u : 0u;
        correctFloat += (flt == labels[n]) ? 1u : 0u;
        std::fprintf(results, "%zu,%d,%d,%d,%d,%d,%d", n, labels[n], predicted, predictedInt4, predicted16, ort, flt);
        for (std::size_t p = 0; p < digits::kModelInputSize; ++p)
        {
            std::fprintf(results, ",%d", static_cast<int>(pixels[n][p]));
        }
        std::fprintf(results, "\n");
    }
    std::fclose(results);
    std::fclose(logitsCsv);
    const double referenceCount = static_cast<double>(digits::kReferenceLogitCount * digits::kModelOutputSize);

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
    std::fprintf(summary, "metric,value\n");
    std::fprintf(summary, "layers,%zu\n", digits::kModelLayerCount);
    std::fprintf(summary, "arena_bytes,%zu\n", digits::kModelArenaSize);
    std::fprintf(summary, "unshared_bytes,%zu\n", digits::kModelUnsharedSize);
    std::fprintf(summary, "busiest_layer_bytes,%zu\n", digits::kModelPeakLiveSize);
    std::fprintf(summary, "weight_bytes_int8,%zu\n", digits::kModelWeightBytes);
    std::fprintf(summary, "weight_bytes_int4,%zu\n", digits::int4::kModelWeightBytes);
    std::fprintf(summary, "accuracy_int8,%.4f\n", correct / static_cast<double>(digits::kTestCount));
    std::fprintf(summary, "accuracy_int4,%.4f\n", correctInt4 / static_cast<double>(digits::kTestCount));
    std::fprintf(summary, "accuracy_16x8,%.4f\n", correct16 / static_cast<double>(digits::kTestCount));
    std::fprintf(summary, "arena_bytes_16x8,%zu\n", digits::int16x8::kModelArenaSize);
    std::fprintf(summary, "same_class_as_float_int8,%zu\n", agreeFloat);
    std::fprintf(summary, "same_class_as_float_16x8,%zu\n", agreeFloat16);
    std::fprintf(summary, "mean_logit_error_int8,%.6f\n", error8 / referenceCount);
    std::fprintf(summary, "mean_logit_error_16x8,%.6f\n", error16 / referenceCount);
    std::fprintf(summary, "max_logit_error_int8,%.6f\n", worst8);
    std::fprintf(summary, "max_logit_error_16x8,%.6f\n", worst16);
    std::fclose(summary);

    const double total = static_cast<double>(digits::kTestCount);
    std::printf("test digits:                    %zu\n", digits::kTestCount);
    std::printf("tinymind int8 accuracy:         %.4f\n", correct / total);
    std::printf("ONNX Runtime int8 accuracy:     %.4f\n", correctOrt / total);
    std::printf("float (PyTorch) accuracy:       %.4f\n", correctFloat / total);
    std::printf("same class as ONNX Runtime:     %zu of %zu\n", agreeOrt, digits::kTestCount);
    std::printf("activation arena:               %zu bytes (unshared %zu, busiest layer %zu)\n",
                digits::kModelArenaSize, digits::kModelUnsharedSize, digits::kModelPeakLiveSize);
    std::printf("int4 weights accuracy:          %.4f (same class as int8 on %zu of %zu)\n",
                correctInt4 / total, agreeInt4, digits::kTestCount);
    std::printf("weights:                        %zu bytes int8, %zu bytes in int4\n",
                digits::kModelWeightBytes, digits::int4::kModelWeightBytes);
    std::printf("16x8 accuracy:                  %.4f (arena %zu bytes, same weights)\n", correct16 / total,
                digits::int16x8::kModelArenaSize);
    std::printf("same class as the float model:  int8 %zu, 16x8 %zu of %zu\n", agreeFloat, agreeFloat16,
                digits::kTestCount);
    std::printf("logit error vs float activations (first %zu digits): mean %.4f int8, %.5f 16x8; "
                "max %.4f int8, %.5f 16x8\n", digits::kReferenceLogitCount, error8 / referenceCount,
                error16 / referenceCount, worst8, worst16);
    std::printf("wrote %s, %s, %s and %s\n", kResultsCsv, kPlanCsv, kSummaryCsv, kLogitsCsv);
    return 0;
}
