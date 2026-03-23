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

extern KanNetworkType testKanNet;

int main(const int argc, char *argv[])
{
    (void)argc;
    (void)argv;

    using namespace std;

    srand(RANDOM_SEED);

    char const* const path = "kan_fixed_xor.txt";
    ofstream results(path);
    ValueType values[KanNetworkType::NumberOfInputLayerNeurons];
    ValueType output[KanNetworkType::NumberOfOutputLayerNeurons];
    ValueType learnedValues[KanNetworkType::NumberOfOutputLayerNeurons];
    ValueType error;
    ValueType avgError(0U);
    ValueType lastAvgError(0U);

    for (unsigned i = 0; i < TRAINING_ITERATIONS; ++i)
    {
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
        if ((i + 1) % NUM_SAMPLES_AVG_ERROR == 0)
        {
            avgError = avgError / ValueType(NUM_SAMPLES_AVG_ERROR, 0);
            cout << "Iteration: " << (i + 1) << " Average Error: " << avgError << endl;
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

    return 0;
}
