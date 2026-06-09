/**
* Elman network on the temporal XOR task -- the textbook demonstration that a
* recurrent (Elman) network carries memory a plain feed-forward MLP cannot.
*
* A stream of random bits x[0], x[1], ... is fed one per timestep. The target at
* step t is the XOR of the CURRENT and PREVIOUS bit:
*
*     target[t] = x[t] XOR x[t-1]        (x[-1] = 0)
*
* The network only ever sees x[t] -- never x[t-1]. So for any given input bit the
* correct answer is 0 or 1 with equal probability depending on history a memoryless
* model cannot observe. A plain MLP therefore cannot beat ~50%: its best response
* to each input is 0.5. An Elman network has a recurrent hidden context that can
* hold x[t-1], so it can recover the XOR and reach ~100%.
*
* The same task is trained on both an `ElmanNeuralNetwork` and a feed-forward
* `NeuralNetwork` of the same shape, in Q16.16 fixed-point, so the only variable
* is the recurrent connection. The CSV overlays both predictions against the
* target; stdout reports the held-out accuracy of each.
*/

#include <iostream>
#include <fstream>
#include <cstdlib>
#include <cstdint>

#include "qformat.hpp"
#include "constants.hpp"
#include "neuralnet.hpp"
#include "activationFunctions.hpp"
#include "fixedPointTransferFunctions.hpp"

#define TRAIN_BITS 2000U
#define TRAIN_EPOCHS 60U
#define EVAL_BITS 600U
#define PLOT_STEPS 48U
#define HIDDEN_NEURONS 8U
#define SEED 7U

typedef tinymind::QValue<16, 16, true, tinymind::RoundUpPolicy> ValueType;
typedef ValueType::FullWidthValueType FullWidthValueType;

template<typename VT>
struct RandomGen
{
    static VT generateRandomWeight()
    {
        // Symmetric small init in [-0.5, 0.5] (real units).
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

// Elman: single recurrent hidden layer (recurrent connection depth 1).
typedef tinymind::ElmanNeuralNetwork<
    ValueType, 1, HIDDEN_NEURONS, 1,
    TransferFunctionsType> ElmanNetworkType;

// Plain feed-forward MLP of the same shape -- no recurrent connection.
typedef tinymind::NeuralNetwork<
    ValueType, 1,
    tinymind::HiddenLayers<HIDDEN_NEURONS>,
    1,
    TransferFunctionsType> MlpNetworkType;

static inline ValueType toQ(double v)
{
    static const double scale = static_cast<double>(1ULL << ValueType::NumberOfFractionalBits);
    ValueType q;
    q.setValue(static_cast<FullWidthValueType>(v * scale));
    return q;
}

static inline double fromQ(const ValueType& q)
{
    static const double scale = static_cast<double>(1ULL << ValueType::NumberOfFractionalBits);
    return static_cast<double>(q.getValue()) / scale;
}

// Train one network on the temporal-XOR bit stream, then evaluate it on a fresh
// held-out stream. Fills preds[PLOT_STEPS] with the first predictions and
// returns held-out accuracy (threshold at 0.5) over EVAL_BITS.
template<typename NetType>
static double trainAndEvaluate(NetType& net,
                               const uint8_t* trainBits,
                               const uint8_t* evalBits,
                               double* preds)
{
    typedef typename NetType::NeuralNetworkTransferFunctionsPolicy TF;

    ValueType input[1];
    ValueType target[1];
    ValueType learned[1];

    // NOTE: a plain Elman/recurrent network exposes no public state reset (only
    // the gated LstmNeuralNetwork / GruNeuralNetwork do). The recurrent context
    // therefore flows continuously across epoch and eval boundaries. Temporal
    // XOR needs only one step of memory, so this affects at most the single bit
    // at each boundary -- immaterial over thousands of steps.
    for (unsigned epoch = 0; epoch < TRAIN_EPOCHS; ++epoch)
    {
        uint8_t prev = 0;
        for (size_t i = 0; i < TRAIN_BITS; ++i)
        {
            const uint8_t cur = trainBits[i];
            input[0] = toQ(static_cast<double>(cur));
            target[0] = toQ(static_cast<double>(cur ^ prev));

            net.feedForward(&input[0]);
            const ValueType error = net.calculateError(&target[0]);
            if (!TF::isWithinZeroTolerance(error))
            {
                net.trainNetwork(&target[0]);
            }
            prev = cur;
        }
    }

    // Warm up one step so the recurrent context encodes evalBits[0] before the
    // first scored step -- this aligns the network's memory with the prev bit
    // the target depends on, removing the train/eval boundary artifact.
    input[0] = toQ(static_cast<double>(evalBits[0]));
    net.feedForward(&input[0]);

    uint8_t prev = evalBits[0];
    size_t correct = 0;
    size_t scored = 0;
    for (size_t i = 1; i < EVAL_BITS; ++i)
    {
        const uint8_t cur = evalBits[i];
        const uint8_t tgt = (cur ^ prev);

        input[0] = toQ(static_cast<double>(cur));
        net.feedForward(&input[0]);
        net.getLearnedValues(&learned[0]);
        const double p = fromQ(learned[0]);

        if ((i - 1) < PLOT_STEPS)
        {
            preds[i - 1] = p;
        }
        if (((p >= 0.5) ? 1u : 0u) == tgt)
        {
            ++correct;
        }
        ++scored;
        prev = cur;
    }

    return static_cast<double>(correct) / static_cast<double>(scored);
}

int main()
{
    // Deterministic bit streams (training + held-out evaluation).
    srand(SEED);
    uint8_t trainBits[TRAIN_BITS];
    uint8_t evalBits[EVAL_BITS];
    for (size_t i = 0; i < TRAIN_BITS; ++i) trainBits[i] = static_cast<uint8_t>(rand() & 1);
    for (size_t i = 0; i < EVAL_BITS; ++i)  evalBits[i]  = static_cast<uint8_t>(rand() & 1);

    double elmanPreds[PLOT_STEPS];
    double mlpPreds[PLOT_STEPS];

    std::cout << "Training Elman network on temporal XOR (Q16.16)..." << std::endl;
    srand(SEED + 1);
    ElmanNetworkType elman;
    const double elmanAcc = trainAndEvaluate(elman, trainBits, evalBits, elmanPreds);

    std::cout << "Training feed-forward MLP on temporal XOR (Q16.16)..." << std::endl;
    srand(SEED + 1);
    MlpNetworkType mlp;
    const double mlpAcc = trainAndEvaluate(mlp, trainBits, evalBits, mlpPreds);

    std::cout << "Held-out accuracy (" << (EVAL_BITS - 1) << " bits):" << std::endl;
    std::cout << "  Elman (recurrent) : " << (elmanAcc * 100.0) << "%" << std::endl;
    std::cout << "  MLP   (no memory) : " << (mlpAcc * 100.0) << "%" << std::endl;

    std::ofstream results("elman_temporal_xor.csv");
    results << "step,input,target,elman,mlp" << std::endl;
    // Scored step s corresponds to eval bit s+1 (bit 0 was the warm-up).
    for (size_t s = 0; s < PLOT_STEPS; ++s)
    {
        const uint8_t cur = evalBits[s + 1];
        const uint8_t prev = evalBits[s];
        const uint8_t tgt = (cur ^ prev);
        results << s << "," << static_cast<unsigned>(cur) << "," << static_cast<unsigned>(tgt) << ","
                << elmanPreds[s] << "," << mlpPreds[s] << std::endl;
    }

    std::cout << "Prediction written to elman_temporal_xor.csv" << std::endl;
    return 0;
}
