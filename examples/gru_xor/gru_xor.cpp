/**
* GRU XOR example: trains a GRU network to learn the XOR function.
*
* Demonstrates the GRU recurrent network type with Q16.16 fixed-point
* arithmetic. XOR is not a sequential task -- each (x, y) -> z pair is
* independent -- so the recurrent state is reset before every sample; otherwise
* state left over from the previous random pattern pollutes the prediction and
* the network never converges.
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

#define TRAINING_ITERATIONS 60000U
#define REPORT_EVERY 500U
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

typedef tinymind::GruNeuralNetwork<
    ValueType, 2,
    tinymind::HiddenLayers<8>,
    1,
    TransferFunctionsType> GruNetworkType;

GruNetworkType gruNet;

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

static const int XOR_CASES[4][3] = { {0,0,0}, {0,1,1}, {1,0,1}, {1,1,0} };

// Deterministic error over the four canonical XOR cases (real units). A clean
// convergence signal. State reset before each case (cases are independent).
static double evaluateXorError()
{
    double sum = 0.0;
    ValueType in[2];
    ValueType out[1];
    for (int c = 0; c < 4; ++c)
    {
        in[0] = ValueType(XOR_CASES[c][0], 0);
        in[1] = ValueType(XOR_CASES[c][1], 0);
        gruNet.resetState();
        gruNet.feedForward(&in[0]);
        gruNet.getLearnedValues(&out[0]);
        const double e = fromQ(out[0]) - static_cast<double>(XOR_CASES[c][2]);
        sum += (e < 0.0) ? -e : e;
    }
    return sum / 4.0;
}

int main(int argc, char* argv[])
{
    srand(RANDOM_SEED);

    // Optional CLI overrides: ./gru_xor [learningRate] [momentum]
    const double lr  = (argc > 1) ? std::atof(argv[1]) : 0.5;
    const double mom = (argc > 2) ? std::atof(argv[2]) : 0.9;
    gruNet.setLearningRate(toQ(lr));
    gruNet.setMomentumRate(toQ(mom));
    gruNet.setAccelerationRate(toQ(0.0));

    ValueType values[2];
    ValueType target[1];

    std::ofstream curve("gru_xor_results.csv");
    curve << "iteration,xor_error" << std::endl;

    for (unsigned i = 0; i < TRAINING_ITERATIONS; ++i)
    {
        const int x = rand() & 0x1;
        const int y = rand() & 0x1;
        values[0] = ValueType(x, 0);
        values[1] = ValueType(y, 0);
        target[0] = ValueType(x ^ y, 0);

        // Each XOR pattern is independent -- start from a clean recurrent state.
        gruNet.resetState();
        gruNet.feedForward(&values[0]);

        const ValueType error = gruNet.calculateError(&target[0]);
        if (!TransferFunctionsType::isWithinZeroTolerance(error))
        {
            gruNet.trainNetwork(&target[0]);
        }

        if ((i + 1) % REPORT_EVERY == 0)
        {
            curve << (i + 1) << "," << evaluateXorError() << std::endl;
        }
    }

    std::cout << "GRU XOR training complete. Final 4-case XOR error: "
              << evaluateXorError() << std::endl;

    // Per-case report: confirm every pattern lands on the correct side of 0.5.
    unsigned correct = 0;
    for (int c = 0; c < 4; ++c)
    {
        ValueType in[2] = { ValueType(XOR_CASES[c][0], 0), ValueType(XOR_CASES[c][1], 0) };
        ValueType out[1];
        gruNet.resetState();
        gruNet.feedForward(&in[0]);
        gruNet.getLearnedValues(&out[0]);
        const double pred = fromQ(out[0]);
        const bool ok = (pred >= 0.5) == (XOR_CASES[c][2] == 1);
        correct += ok ? 1 : 0;
        std::cout << "  " << XOR_CASES[c][0] << " ^ " << XOR_CASES[c][1]
                  << " = " << XOR_CASES[c][2] << "  pred=" << pred
                  << (ok ? "  ok" : "  WRONG") << std::endl;
    }
    std::cout << correct << "/4 patterns classified correctly." << std::endl;

    // Decision-surface CSV: sweep the trained GRU over the [0,1]^2 input grid,
    // resetting recurrent state per point, so plot.py renders predicted vs
    // actual the same way as the other XOR examples.
    {
        std::ofstream surf("gru_xor_decision_surface.csv");
        surf << "x0,x1,prob" << std::endl;
        const int G = 41;
        for (int a = 0; a < G; ++a)
        {
            for (int b = 0; b < G; ++b)
            {
                const double x0 = static_cast<double>(a) / (G - 1);
                const double x1 = static_cast<double>(b) / (G - 1);
                ValueType in[2] = { toQ(x0), toQ(x1) };
                ValueType out[1];
                gruNet.resetState();
                gruNet.feedForward(&in[0]);
                gruNet.getLearnedValues(&out[0]);
                surf << x0 << "," << x1 << "," << fromQ(out[0]) << std::endl;
            }
        }
    }

    return 0;
}
