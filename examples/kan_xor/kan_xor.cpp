/**
* Copyright (c) 2025 Dan McLeran
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

#include <iostream>
#include <fstream>
#include <string>
#include <cstring>
#include <cstdlib>

#include "constants.hpp"
#include "kan_xornet.h"

#define TRAINING_ITERATIONS 20000U
#define NUM_SAMPLES_AVG_ERROR 100U
#define RANDOM_SEED 7U

static void generateXorTrainingValue(ValueType& x, ValueType& y, ValueType& z)
{
    const int randX = rand() & 0x1;
    const int randY = rand() & 0x1;
    const int result = (randX ^ randY);

    x = ValueType(randX, 0);
    y = ValueType(randY, 0);
    z = ValueType(result, 0);
}

// Dense-training generator: sample the whole [0,1]^2 input region (not just the
// four corners), label by the XOR of the two halves. Constraining the splines
// everywhere -- rather than at four points -- yields a smooth decision surface,
// the way a KAN classifier is trained in practice.
static void generateDenseTrainingValue(ValueType& x, ValueType& y, ValueType& z)
{
    const double fx = static_cast<double>(rand()) / static_cast<double>(RAND_MAX);
    const double fy = static_cast<double>(rand()) / static_cast<double>(RAND_MAX);
    const int result = ((fx > 0.5) != (fy > 0.5)) ? 1 : 0;
    const double scale = static_cast<double>(1 << ValueType::NumberOfFractionalBits);
    x.setValue(static_cast<FullWidthValueType>(fx * scale));
    y.setValue(static_cast<FullWidthValueType>(fy * scale));
    z = ValueType(result, 0);
}

extern KanNetworkType testKanNet;

int main(const int argc, char *argv[])
{
    using namespace std;

    // --dense: train over the whole input region for a smooth surface; default
    // trains on the four crisp XOR corners (the apples-to-apples comparison).
    const bool dense = (argc > 1 && std::strcmp(argv[1], "--dense") == 0);

    srand(RANDOM_SEED);

    char const* const path = "kan_fixed_xor.txt";
    ofstream results(path);

    // Learning curve CSV (iteration, avg_error) for plot.py.
    ofstream curve(dense ? "kan_xor_dense_training.csv" : "kan_xor_training.csv");
    curve << "iteration,avg_error" << std::endl;
    ValueType values[KanNetworkType::NumberOfInputLayerNeurons];
    ValueType output[KanNetworkType::NumberOfOutputLayerNeurons];
    ValueType learnedValues[KanNetworkType::NumberOfOutputLayerNeurons];
    ValueType error;
    ValueType avgError(0U);
    ValueType lastAvgError(0U);
    // Clean floating-point error accumulator for the CSV: the QValue avgError
    // above sums raw fixed-point integers and can wrap, producing a noisy curve.
    double errSumD = 0.0;
    const double fracScale = static_cast<double>(1 << ValueType::NumberOfFractionalBits);

    for (unsigned i = 0; i < TRAINING_ITERATIONS; ++i)
    {
        if (dense)
            generateDenseTrainingValue(values[0], values[1], output[0]);
        else
            generateXorTrainingValue(values[0], values[1], output[0]);

        testKanNet.feedForward(&values[0]);
        error = testKanNet.calculateError(&output[0]);
        if (!KanNetworkType::KanTransferFunctionsPolicy::isWithinZeroTolerance(error))
        {
            testKanNet.trainNetwork(&output[0]);
        }
        testKanNet.getLearnedValues(&learnedValues[0]);

        results << error << std::endl;

        avgError += abs(error.getValue());
        {
            double e = static_cast<double>(error.getValue()) / fracScale;
            errSumD += (e < 0.0) ? -e : e;
        }
        if ((i + 1) % NUM_SAMPLES_AVG_ERROR == 0)
        {
            avgError = avgError / ValueType(NUM_SAMPLES_AVG_ERROR, 0);
            cout << "Iteration: " << (i + 1) << " Average Error: " << avgError << endl;
            curve << (i + 1) << "," << (errSumD / NUM_SAMPLES_AVG_ERROR) << std::endl;
            errSumD = 0.0;
            lastAvgError = avgError;
            avgError = ValueType(0U);
        }
    }

    if (!KanNetworkType::KanTransferFunctionsPolicy::isWithinZeroTolerance(lastAvgError))
    {
        cout << "Training did not complete successfully after " << TRAINING_ITERATIONS << " iterations." << endl;
    }
    else
    {
        cout << "Training completed successfully." << endl;
    }

    results.close();

    // Decision-surface CSV: sweep the trained KAN over the [0,1]^2 input grid
    // so plot.py renders predicted vs actual the same way as the other XOR
    // examples.
    {
        ofstream surf(dense ? "kan_xor_dense_decision_surface.csv" : "kan_xor_decision_surface.csv");
        surf << "x0,x1,prob" << std::endl;
        const int G = 41;
        for (int a = 0; a < G; ++a)
        {
            for (int b = 0; b < G; ++b)
            {
                const double x0 = static_cast<double>(a) / (G - 1);
                const double x1 = static_cast<double>(b) / (G - 1);
                ValueType in[2];
                in[0].setValue(static_cast<ValueType::FullWidthValueType>(x0 * fracScale));
                in[1].setValue(static_cast<ValueType::FullWidthValueType>(x1 * fracScale));
                ValueType outv[KanNetworkType::NumberOfOutputLayerNeurons];
                testKanNet.feedForward(&in[0]);
                testKanNet.getLearnedValues(&outv[0]);
                const double prob = static_cast<double>(outv[0].getValue()) / fracScale;
                surf << x0 << "," << x1 << "," << prob << std::endl;
            }
        }
    }

    return 0;
}
