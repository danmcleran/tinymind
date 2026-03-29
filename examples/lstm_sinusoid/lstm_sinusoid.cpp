/**
* LSTM sinusoid prediction example: trains an LSTM network to predict
* the next value in a sinusoidal sequence.
*
* Demonstrates sequential input processing, state management between
* epochs, and auto-regressive prediction with an LSTM network.
*/

#include <iostream>
#include <fstream>
#include <cstdlib>
#include <cstdint>
#include <cmath>

#include "qformat.hpp"
#include "neuralnet.hpp"
#include "activationFunctions.hpp"
#include "fixedPointTransferFunctions.hpp"
#include "earlyStopping.hpp"
#include "teacherForcing.hpp"
#include "truncatedBPTT.hpp"

#define TRAINING_EPOCHS 50000U
#define SAMPLES_PER_PERIOD 10U
#define PREDICTION_STEPS 20U
#define RANDOM_SEED 7U

typedef tinymind::QValue<8, 8, true, tinymind::RoundUpPolicy> ValueType;
typedef ValueType::FullWidthValueType FullWidthValueType;

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
    tinymind::SigmoidActivationPolicy<ValueType>,
    1,
    tinymind::DefaultNetworkInitializer<ValueType>,
    tinymind::MeanSquaredErrorCalculator<ValueType, 1>,
    tinymind::ZeroToleranceCalculator<ValueType>,
    tinymind::GradientClipByValue<ValueType>> TransferFunctionsType;

typedef tinymind::LstmNeuralNetwork<
    ValueType, 1,
    tinymind::HiddenLayers<16>,
    1,
    TransferFunctionsType> LstmNetworkType;

LstmNetworkType lstmNet;

int main()
{
    srand(RANDOM_SEED);

    // Generate sinusoid samples scaled to [0, 1] for sigmoid output
    static const size_t NUM_SAMPLES = SAMPLES_PER_PERIOD;
    double sinSamplesDouble[NUM_SAMPLES];
    ValueType sinSamples[NUM_SAMPLES];

    for (size_t i = 0; i < NUM_SAMPLES; ++i)
    {
        sinSamplesDouble[i] = (sin(2.0 * M_PI * static_cast<double>(i) / static_cast<double>(SAMPLES_PER_PERIOD)) + 1.0) / 2.0;
        const FullWidthValueType raw = static_cast<FullWidthValueType>(sinSamplesDouble[i] * (1 << ValueType::NumberOfFractionalBits));
        sinSamples[i] = ValueType(raw);
    }

    ValueType input[1];
    ValueType target[1];
    ValueType learnedValues[1];
    ValueType error;

    std::cout << "Training LSTM on sinusoid (" << NUM_SAMPLES << " samples/period)..." << std::endl;

    // Train: feed sequential pairs (sin[t] -> sin[t+1])
    for (unsigned epoch = 0; epoch < TRAINING_EPOCHS; ++epoch)
    {
        for (size_t t = 0; t < NUM_SAMPLES - 1; ++t)
        {
            input[0] = sinSamples[t];
            target[0] = sinSamples[t + 1];

            lstmNet.feedForward(&input[0]);
            error = lstmNet.calculateError(&target[0]);

            if (!TransferFunctionsType::isWithinZeroTolerance(error))
            {
                lstmNet.trainNetwork(&target[0]);
            }
        }
    }

    std::cout << "Training complete. Final error: " << error << std::endl;

    // Auto-regressive prediction
    std::ofstream results("lstm_sinusoid_prediction.txt");
    results << "Step,True,Predicted" << std::endl;

    // Seed with first sample
    ValueType curr = sinSamples[0];

    for (size_t p = 0; p < PREDICTION_STEPS; ++p)
    {
        input[0] = curr;
        lstmNet.feedForward(&input[0]);
        lstmNet.getLearnedValues(&learnedValues[0]);

        // True value (wrapping around the period)
        const size_t trueIdx = (p + 1) % NUM_SAMPLES;
        const double trueValue = sinSamplesDouble[trueIdx];

        results << p << "," << trueValue << "," << learnedValues[0] << std::endl;

        // Feed prediction as next input (auto-regressive)
        curr = learnedValues[0];
    }

    std::cout << "Prediction written to lstm_sinusoid_prediction.txt" << std::endl;

    return 0;
}
