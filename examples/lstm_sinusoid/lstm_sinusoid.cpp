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
            static_cast<typename VT::FullWidthValueType>((rand() % (2 * half + 1)) - half);
        return VT(raw);
    }
};

// Learning rate 0.1 with no momentum. With the library default (0.25 with
// momentum 0.5) and training on every step, Q16.16 training saturated on 3 of
// 5 weight seeds (measured before the fixed-point momentum terms were made to
// round toward zero, which removed a per-step weight drift); skipping steps
// inside the zero-tolerance band avoided that but left the gates barely
// trained (one-step error about 5x worse).
template<typename VT>
struct SinusoidNetworkInitializer : public tinymind::DefaultNetworkInitializer<VT>
{
    static VT initialLearningRate()
    {
        static const VT rate(0, 6554); // 0.1 in Q16.16
        return rate;
    }

    static VT initialMomentumRate()
    {
        return tinymind::Constants<VT>::zero();
    }

    static VT initialAccelerationRate()
    {
        return tinymind::Constants<VT>::zero();
    }
};

typedef tinymind::FixedPointTransferFunctions<
    ValueType,
    RandomGen<ValueType>,
    tinymind::TanhActivationPolicy<ValueType>,
    tinymind::SigmoidActivationPolicy<ValueType>,
    1,
    SinusoidNetworkInitializer<ValueType>,
    tinymind::MeanSquaredErrorCalculator<ValueType, 1>,
    tinymind::ZeroToleranceCalculator<ValueType>,
    tinymind::GradientClipByValue<ValueType>> TransferFunctionsType;

typedef tinymind::LstmNeuralNetwork<
    ValueType, 1,
    tinymind::HiddenLayers<16>,
    1,
    TransferFunctionsType> LstmNetworkType;

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

// Sinusoid sample i, scaled to [0, 1] for the sigmoid output. `phase` is in
// samples: training uses phase 0; the held-out evaluation uses phase 0.5, so
// every input and target it sees falls between the training samples.
static inline double trueSample(size_t i, double phase = 0.0)
{
    return (std::sin(2.0 * M_PI * (static_cast<double>(i) + phase) / static_cast<double>(SAMPLES_PER_PERIOD)) + 1.0) / 2.0;
}

static const double HELD_OUT_PHASE = 0.5;

static const unsigned WEIGHT_INIT_SEED = 1U; // see main()

int main()
{
    // Two seeds: the initial weights come from WEIGHT_INIT_SEED and everything
    // else (data, shuffles, training samples) from the seed set just below.
    // WEIGHT_INIT_SEED = 1 reproduces this example's published results: the
    // network used to be a global, constructed before main() seeded rand(), so it
    // drew from glibc's default seed of 1.
    srand(WEIGHT_INIT_SEED);
    static LstmNetworkType lstmNet;

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
    double lastEpochSq = 0.0;

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
            if (epoch + 1 == TRAINING_EPOCHS)
            {
                // Last epoch: measure the training-signal error in real units
                // (the Q16.16 squared error rounds to 0 at this accuracy).
                lstmNet.getLearnedValues(&learnedValues[0]);
                const double e = fromQ(learnedValues[0]) - fromQ(target[0]);
                lastEpochSq += e * e;
            }
            lstmNet.trainNetwork(&target[0]);
        }
    }

    std::cout << "Training complete. One-step RMS error on the training signal (last epoch): "
              << std::sqrt(lastEpochSq / static_cast<double>(NUM_SAMPLES)) << std::endl;

    // Evaluate on a HELD-OUT signal: the same sinusoid shifted by half a
    // sample, so none of its values were seen in training. Two prediction
    // modes over the same horizon, each after a teacher-forced warm-up that
    // locks the recurrent state onto the phase:
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
        input[0] = toQ(trueSample(i, HELD_OUT_PHASE));
        lstmNet.feedForward(&input[0]);
    }
    for (size_t p = 0; p < PREDICTION_STEPS; ++p)
    {
        input[0] = toQ(trueSample(WARMUP_STEPS - 1 + p, HELD_OUT_PHASE));
        lstmNet.feedForward(&input[0]);
        lstmNet.getLearnedValues(&learnedValues[0]);
        oneStep[p] = fromQ(learnedValues[0]);
    }

    // --- auto-regressive free-run ---
    lstmNet.resetState();
    for (size_t i = 0; i + 1 < WARMUP_STEPS; ++i)
    {
        input[0] = toQ(trueSample(i, HELD_OUT_PHASE));
        lstmNet.feedForward(&input[0]);
    }
    ValueType curr = toQ(trueSample(WARMUP_STEPS - 1, HELD_OUT_PHASE));
    for (size_t p = 0; p < PREDICTION_STEPS; ++p)
    {
        input[0] = curr;
        lstmNet.feedForward(&input[0]);
        lstmNet.getLearnedValues(&learnedValues[0]);
        freeRun[p] = fromQ(learnedValues[0]);
        curr = learnedValues[0]; // feed prediction back (auto-regressive)
    }

    // RMS errors over the horizon, against persistence (predict sin[t+1] =
    // sin[t]), the trivial one-step forecaster.
    double oneStepSq = 0.0;
    double freeRunSq = 0.0;
    double persistenceSq = 0.0;
    for (size_t p = 0; p < PREDICTION_STEPS; ++p)
    {
        const double truth = trueSample(WARMUP_STEPS + p, HELD_OUT_PHASE);
        const double previous = trueSample(WARMUP_STEPS - 1 + p, HELD_OUT_PHASE);
        oneStepSq += (oneStep[p] - truth) * (oneStep[p] - truth);
        freeRunSq += (freeRun[p] - truth) * (freeRun[p] - truth);
        persistenceSq += (previous - truth) * (previous - truth);
    }
    const double n = static_cast<double>(PREDICTION_STEPS);
    std::cout << "Held-out signal (phase shifted by " << HELD_OUT_PHASE << " sample), "
              << PREDICTION_STEPS << " steps:" << std::endl;
    std::cout << "  one-step RMS error:    " << std::sqrt(oneStepSq / n) << std::endl;
    std::cout << "  persistence baseline:  " << std::sqrt(persistenceSq / n) << std::endl;
    std::cout << "  free-run RMS error:    " << std::sqrt(freeRunSq / n) << std::endl;

    std::ofstream results("lstm_sinusoid.csv");
    results << "step,true,one_step,free_run" << std::endl;
    for (size_t p = 0; p < PREDICTION_STEPS; ++p)
    {
        results << p << "," << trueSample(WARMUP_STEPS + p, HELD_OUT_PHASE) << ","
                << oneStep[p] << "," << freeRun[p] << std::endl;
    }

    std::cout << "Prediction written to lstm_sinusoid.csv" << std::endl;

    return 0;
}
