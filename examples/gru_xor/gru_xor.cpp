/**
* GRU XOR example: trains a GRU network to learn the XOR function.
*
* Demonstrates the GRU recurrent network type with Q8.8 fixed-point
* arithmetic and the scheduled sampling training technique.
*/

#include <iostream>
#include <fstream>
#include <cstdlib>
#include <cstdint>

#include "qformat.hpp"
#include "neuralnet.hpp"
#include "activationFunctions.hpp"
#include "fixedPointTransferFunctions.hpp"
#include "earlyStopping.hpp"

#define TRAINING_ITERATIONS 20000U
#define RANDOM_SEED 7U

typedef tinymind::QValue<8, 8, true, tinymind::RoundUpPolicy> ValueType;

template<typename VT>
struct RandomGen
{
    static VT generateRandomWeight()
    {
        const int r = (rand() % 512) - 256;
        return VT(static_cast<typename VT::FullWidthValueType>(r));
    }
};

typedef tinymind::FixedPointTransferFunctions<
    ValueType,
    RandomGen<ValueType>,
    tinymind::TanhActivationPolicy<ValueType>,
    tinymind::TanhActivationPolicy<ValueType>,
    1,
    tinymind::DefaultNetworkInitializer<ValueType>,
    tinymind::MeanSquaredErrorCalculator<ValueType, 1>,
    tinymind::ZeroToleranceCalculator<ValueType>,
    tinymind::GradientClipByValue<ValueType>> TransferFunctionsType;

typedef tinymind::GruNeuralNetwork<
    ValueType, 2,
    tinymind::HiddenLayers<5>,
    1,
    TransferFunctionsType> GruNetworkType;

GruNetworkType gruNet;

static void generateXorValues(ValueType& x, ValueType& y, ValueType& z)
{
    const int randX = rand() & 0x1;
    const int randY = rand() & 0x1;
    const int result = (randX ^ randY);

    x = ValueType(randX, 0);
    y = ValueType(randY, 0);
    z = ValueType(result, 0);
}

int main()
{
    srand(RANDOM_SEED);

    ValueType values[2];
    ValueType output[1];
    ValueType learnedValues[1];
    ValueType error;
    ValueType lastError(0);

    tinymind::EarlyStopping<ValueType, 5000> stopper;

    std::ofstream results("gru_xor_results.txt");
    results << "Iteration,Target,Learned,Error" << std::endl;

    for (unsigned i = 0; i < TRAINING_ITERATIONS; ++i)
    {
        generateXorValues(values[0], values[1], output[0]);

        gruNet.feedForward(&values[0]);
        error = gruNet.calculateError(&output[0]);

        if (!TransferFunctionsType::isWithinZeroTolerance(error))
        {
            gruNet.trainNetwork(&output[0]);
        }

        gruNet.getLearnedValues(&learnedValues[0]);

        lastError = error;

        results << i << "," << output[0] << "," << learnedValues[0] << "," << error << std::endl;

        if (stopper.shouldStop(error))
        {
            std::cout << "Early stopping at iteration " << i << std::endl;
            break;
        }
    }

    std::cout << "GRU XOR training complete. Final error: " << lastError << std::endl;

    return 0;
}
