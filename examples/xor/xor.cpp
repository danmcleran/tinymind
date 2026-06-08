/**
* Copyright (c) 2020 Intel Corporation
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

#include "constants.hpp"
#include "nnproperties.hpp"

#include "xornet.h"

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

extern NeuralNetworkType testNeuralNet;

// Deterministic validation error over the four canonical XOR cases (real units).
// Far cleaner than averaging the noisy random training samples -- it measures
// "how well does the network solve XOR right now". Pure forward passes; does not
// alter training.
static double evaluateXorError()
{
    static const int cases[4][3] = { {0,0,0}, {0,1,1}, {1,0,1}, {1,1,0} };
    const double fracScale = static_cast<double>(1 << ValueType::NumberOfFractionalBits);
    double sum = 0.0;
    ValueType in[2];
    ValueType out[NeuralNetworkType::NumberOfOutputLayerNeurons];
    for (int c = 0; c < 4; ++c)
    {
        in[0] = ValueType(cases[c][0], 0);
        in[1] = ValueType(cases[c][1], 0);
        testNeuralNet.feedForward(&in[0]);
        testNeuralNet.getLearnedValues(&out[0]);
        const double pred = static_cast<double>(out[0].getValue()) / fracScale;
        const double e = pred - static_cast<double>(cases[c][2]);
        sum += (e < 0.0) ? -e : e;
    }
    return sum / 4.0;
}

extern NeuralNetworkType testNeuralNet;

int main(const int argc, char *argv[])
{
    using namespace std;

    srand(RANDOM_SEED); // seed random number generator

    // Optional CLI overrides for experimentation: ./xor [learningRate] [momentum]
    if (argc > 1)
    {
        ValueType lr; lr.setValue(static_cast<FullWidthValueType>(atof(argv[1]) * (1 << ValueType::NumberOfFractionalBits)));
        testNeuralNet.setLearningRate(lr);
    }
    if (argc > 2)
    {
        ValueType mom; mom.setValue(static_cast<FullWidthValueType>(atof(argv[2]) * (1 << ValueType::NumberOfFractionalBits)));
        testNeuralNet.setMomentumRate(mom);
    }

    char const* const path = "nn_fixed_xor.txt";
    ofstream results(path);

    // Learning curve CSV (header + one row per averaging window) for plot.py.
    ofstream curve("xor_training.csv");
    curve << "iteration,avg_error" << std::endl;
    ValueType values[NeuralNetworkType::NumberOfInputLayerNeurons];
    ValueType output[NeuralNetworkType::NumberOfOutputLayerNeurons];
    ValueType learnedValues[NeuralNetworkType::NumberOfOutputLayerNeurons];
    ValueType error;
    ValueType avgError(0U);
    ValueType lastAvgError(0U);

    tinymind::NetworkPropertiesFileManager<NeuralNetworkType>::writeHeader(results);

    for (unsigned i = 0; i < TRAINING_ITERATIONS; ++i)
    {
        generateXorTrainingValue(values[0], values[1], output[0]);

        testNeuralNet.feedForward(&values[0]);
        error = testNeuralNet.calculateError(&output[0]);
        if (!NeuralNetworkType::NeuralNetworkTransferFunctionsPolicy::isWithinZeroTolerance(error))
        {
            testNeuralNet.trainNetwork(&output[0]);
        }
        testNeuralNet.getLearnedValues(&learnedValues[0]);

        tinymind::NetworkPropertiesFileManager<NeuralNetworkType>::storeNetworkProperties(testNeuralNet, results, &output[0], &learnedValues[0]);
        results << error << std::endl;

        avgError += abs(error.getValue());
        if ((i + 1) % NUM_SAMPLES_AVG_ERROR == 0)
        {
            avgError = avgError / ValueType(NUM_SAMPLES_AVG_ERROR, 0);
            cout << "Iteration: " << (i + 1) << " Average Error: " << avgError << endl;
            // Log the deterministic 4-case XOR error (clean convergence signal).
            curve << (i + 1) << "," << evaluateXorError() << std::endl;
            lastAvgError = avgError;
            avgError = ValueType(0U);
        }
    }

    (void)lastAvgError;

    // Success = every XOR pattern lands on the correct side of 0.5. The sigmoid
    // output saturates softly (predictions ~0.09 / 0.91), so the residual
    // 4-case error floors around ~0.09 even when classification is perfect --
    // judge by classification, not by error == 0.
    static const int cases[4][3] = { {0,0,0}, {0,1,1}, {1,0,1}, {1,1,0} };
    const double fracScale = static_cast<double>(1 << ValueType::NumberOfFractionalBits);
    unsigned correct = 0;
    for (int c = 0; c < 4; ++c)
    {
        ValueType in[2] = { ValueType(cases[c][0], 0), ValueType(cases[c][1], 0) };
        ValueType out[NeuralNetworkType::NumberOfOutputLayerNeurons];
        testNeuralNet.feedForward(&in[0]);
        testNeuralNet.getLearnedValues(&out[0]);
        const double pred = static_cast<double>(out[0].getValue()) / fracScale;
        const bool ok = (pred >= 0.5) == (cases[c][2] == 1);
        correct += ok ? 1 : 0;
        cout << "  " << cases[c][0] << " ^ " << cases[c][1] << " = " << cases[c][2]
             << "  pred=" << pred << (ok ? "  ok" : "  WRONG") << endl;
    }
    cout << correct << "/4 patterns classified correctly. Final 4-case error: "
         << evaluateXorError() << endl;

    results.close();

    // Decision-surface CSV: sweep the trained network over the [0,1]^2 input
    // grid so plot.py renders the learned XOR boundary as a heatmap -- the same
    // predicted-vs-actual presentation as the PyTorch xor examples.
    {
        std::ofstream surf("xor_decision_surface.csv");
        surf << "x0,x1,prob" << std::endl;
        const int G = 41;
        for (int a = 0; a < G; ++a)
        {
            for (int b = 0; b < G; ++b)
            {
                const double x0 = static_cast<double>(a) / (G - 1);
                const double x1 = static_cast<double>(b) / (G - 1);
                ValueType in[2];
                in[0].setValue(static_cast<FullWidthValueType>(x0 * fracScale));
                in[1].setValue(static_cast<FullWidthValueType>(x1 * fracScale));
                ValueType out[NeuralNetworkType::NumberOfOutputLayerNeurons];
                testNeuralNet.feedForward(&in[0]);
                testNeuralNet.getLearnedValues(&out[0]);
                const double prob = static_cast<double>(out[0].getValue()) / fracScale;
                surf << x0 << "," << x1 << "," << prob << std::endl;
            }
        }
    }

    return 0;
}
