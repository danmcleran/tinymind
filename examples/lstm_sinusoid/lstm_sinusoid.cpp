/**
* LSTM sinusoid prediction example: trains an LSTM network to predict
* the next value in a sinusoidal sequence, then free-runs auto-regressively.
*
* Demonstrates sequential input processing, recurrent state management, and
* auto-regressive generation with an LSTM network. The single scalar input
* sin[t] is an ambiguous predictor of sin[t+1] (the rising and falling
* branches of the sine share y-values), so the network must use its recurrent
* state to track phase -- exactly what the LSTM cell is for. Q16.16 fixed-point
* gives the state enough precision to do this; the coarser Q8.8 grid collapses
* the free-run onto the 0/1 rails.
*/

#include <iostream>
#include <fstream>
#include <cstdlib>
#include <cstdint>
#include <cmath>

#include "qformat.hpp"
#include "constants.hpp"
#include "neuralnet.hpp"
#include "activationFunctions.hpp"
#include "fixedPointTransferFunctions.hpp"
#include "earlyStopping.hpp"
#include "teacherForcing.hpp"
#include "truncatedBPTT.hpp"

#define TRAINING_EPOCHS 20000U
#define SAMPLES_PER_PERIOD 20U
#define WARMUP_STEPS SAMPLES_PER_PERIOD          // prime state with one true period
#define PREDICTION_STEPS (2U * SAMPLES_PER_PERIOD) // free-run two periods
#define RANDOM_SEED 7U

typedef tinymind::QValue<16, 16, true, tinymind::RoundUpPolicy> ValueType;
typedef ValueType::FullWidthValueType FullWidthValueType;

template<typename VT>
struct RandomGen
{
    static VT generateRandomWeight()
    {
        // Symmetric small init in [-0.5, 0.5] (real units). A raw-value init
        // tuned for Q8.8 is ~256x too small in Q16.16 and never trains.
        const typename VT::FullWidthValueType half =
            tinymind::Constants<VT>::one().getValue() / 2;
        const typename VT::FullWidthValueType raw =
            (static_cast<typename VT::FullWidthValueType>(rand()) % (2 * half + 1)) - half;
        return VT(raw);
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

static inline ValueType toQ(double v)
{
    static const double scale = static_cast<double>(1ULL << ValueType::NumberOfFractionalBits);
    ValueType q;
    q.setValue(static_cast<FullWidthValueType>(std::lround(v * scale)));
    return q;
}

static inline double fromQ(const ValueType& q)
{
    static const double scale = static_cast<double>(1ULL << ValueType::NumberOfFractionalBits);
    return static_cast<double>(q.getValue()) / scale;
}

// Sinusoid sample i, scaled to [0, 1] for the sigmoid output.
static inline double trueSample(size_t i)
{
    return (std::sin(2.0 * M_PI * static_cast<double>(i) / static_cast<double>(SAMPLES_PER_PERIOD)) + 1.0) / 2.0;
}

int main()
{
    srand(RANDOM_SEED);

    static const size_t NUM_SAMPLES = SAMPLES_PER_PERIOD;
    ValueType sinSamples[NUM_SAMPLES];
    for (size_t i = 0; i < NUM_SAMPLES; ++i)
    {
        sinSamples[i] = toQ(trueSample(i));
    }

    ValueType input[1];
    ValueType target[1];
    ValueType learnedValues[1];
    ValueType error;

    std::cout << "Training LSTM on sinusoid (" << NUM_SAMPLES << " samples/period, Q16.16)..." << std::endl;

    // Train on a continuous periodic stream: sin[t] -> sin[t+1]. The recurrent
    // state carries across epoch boundaries (sample N-1 wraps to sample 0), so
    // the network sees one unbroken periodic sequence.
    for (unsigned epoch = 0; epoch < TRAINING_EPOCHS; ++epoch)
    {
        for (size_t t = 0; t < NUM_SAMPLES; ++t)
        {
            input[0] = sinSamples[t];
            target[0] = sinSamples[(t + 1) % NUM_SAMPLES];

            lstmNet.feedForward(&input[0]);
            error = lstmNet.calculateError(&target[0]);

            if (!TransferFunctionsType::isWithinZeroTolerance(error))
            {
                lstmNet.trainNetwork(&target[0]);
            }
        }
    }

    std::cout << "Training complete. Final error: " << error << std::endl;

    // Evaluate two prediction modes over the same horizon, each after a
    // teacher-forced warm-up that locks the recurrent state onto the phase:
    //
    //   one_step  -- teacher forced: always fed the TRUE sin[t], predicts
    //                sin[t+1]. Measures how well the dynamics were learned.
    //   free_run  -- auto-regressive: fed its OWN previous prediction back as
    //                the next input. The honest generation test; small
    //                one-step errors compound into amplitude/phase drift.
    double oneStep[PREDICTION_STEPS];
    double freeRun[PREDICTION_STEPS];

    // --- one-step-ahead (teacher forced) ---
    lstmNet.resetState();
    for (size_t i = 0; i + 1 < WARMUP_STEPS; ++i)
    {
        input[0] = toQ(trueSample(i));
        lstmNet.feedForward(&input[0]);
    }
    for (size_t p = 0; p < PREDICTION_STEPS; ++p)
    {
        input[0] = toQ(trueSample(WARMUP_STEPS - 1 + p));
        lstmNet.feedForward(&input[0]);
        lstmNet.getLearnedValues(&learnedValues[0]);
        oneStep[p] = fromQ(learnedValues[0]);
    }

    // --- auto-regressive free-run ---
    lstmNet.resetState();
    for (size_t i = 0; i + 1 < WARMUP_STEPS; ++i)
    {
        input[0] = toQ(trueSample(i));
        lstmNet.feedForward(&input[0]);
    }
    ValueType curr = toQ(trueSample(WARMUP_STEPS - 1));
    for (size_t p = 0; p < PREDICTION_STEPS; ++p)
    {
        input[0] = curr;
        lstmNet.feedForward(&input[0]);
        lstmNet.getLearnedValues(&learnedValues[0]);
        freeRun[p] = fromQ(learnedValues[0]);
        curr = learnedValues[0]; // feed prediction back (auto-regressive)
    }

    std::ofstream results("lstm_sinusoid.csv");
    results << "step,true,one_step,free_run" << std::endl;
    for (size_t p = 0; p < PREDICTION_STEPS; ++p)
    {
        results << p << "," << trueSample(WARMUP_STEPS + p) << ","
                << oneStep[p] << "," << freeRun[p] << std::endl;
    }

    std::cout << "Prediction written to lstm_sinusoid.csv" << std::endl;

    return 0;
}
