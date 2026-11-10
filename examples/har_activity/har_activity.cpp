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

// Human Activity Recognition (HAR) classifier built on tinymind.
//
// Inspired by the UCI "Human Activity Recognition Using Smartphones" dataset:
//   https://archive.ics.uci.edu/dataset/240
//
// This is a RECURRENT classifier: an LSTM network reads a 32-timestep window of
// tri-axial accelerometer samples (ax, ay, az) one timestep at a time and
// classifies the whole window into one of four activities at the LAST timestep:
//
//   0 WALKING             symmetric vertical (z) oscillation
//   1 WALKING_UPSTAIRS    per step, a sharp vertical rise then a slow decline
//   2 WALKING_DOWNSTAIRS  per step, a slow vertical rise then a sharp drop
//   3 SITTING             near-flat, small noise, gravity on the y axis
//
// The three gait classes share the same step rate, lateral sway, gravity
// direction, and the same per-axis mean and standard deviation; UPSTAIRS and
// DOWNSTAIRS are exact time reversals of each other, so every sample has the
// same distribution in both. What separates them is the ORDER of the samples
// (a sharp rise followed by a slow fall, or the reverse), which a classifier
// that reads one timestep at a time can only see by remembering earlier
// timesteps. The example reports a nearest-centroid baseline on per-axis
// mean + standard deviation, which cannot separate the gait classes, and
// `--readout-only` trains only the hidden-to-output weights (LSTM input,
// gate and recurrent weights frozen at their random initial values).
//
// Network: 3 inputs -> LSTM hidden 16 (tanh) -> 4 sigmoid outputs (one-hot
// activity), Q16.16 fixed-point. GradientClipByValue keeps the recurrent
// training stable in fixed point.
//
// DATA: the real HAR raw inertial signals are too large to bundle, so by
// default the example SYNTHESIZES physically-motivated accelerometer windows
// (deterministic, std::srand(7U)). If `har.csv` is present in the working
// directory it is loaded instead (long format documented in the README).

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "qformat.hpp"
#include "activationFunctions.hpp"
#include "fixedPointTransferFunctions.hpp"
#include "neuralnet.hpp"
#include "gradientClipping.hpp"

// ---------------------------------------------------------------------------
// Problem dimensions
// ---------------------------------------------------------------------------

static constexpr size_t WINDOW_LENGTH    = 32;  // timesteps per window
static constexpr size_t NUMBER_OF_INPUTS = 3;   // ax, ay, az
static constexpr size_t NUMBER_OF_OUTPUTS = 4;  // one per activity
static constexpr size_t HIDDEN_NEURONS   = 16;

static constexpr size_t TRAIN_PER_CLASS = 200;
static constexpr size_t TEST_PER_CLASS  = 50;

static const char* const CLASS_NAMES[NUMBER_OF_OUTPUTS] = {
    "walking", "upstairs", "downstairs", "sitting"
};

// ---------------------------------------------------------------------------
// Q-format recurrent network definition
// ---------------------------------------------------------------------------

typedef tinymind::QValue<16, 16, true> ValueType;
typedef typename ValueType::FullWidthValueType FullWidthValueType;

struct RandomNumberGenerator
{
    static ValueType generateRandomWeight()
    {
        // Symmetric small init in [-0.25, 0.25] (raw Q-format units) -- small
        // weights keep the recurrent forward pass inside Q16.16's stable range.
        const FullWidthValueType quarter =
            tinymind::Constants<ValueType>::one().getValue() / 4;
        const FullWidthValueType weight =
            static_cast<FullWidthValueType>((rand() % (2 * quarter + 1)) - quarter);
        return ValueType(weight);
    }
};

// Learning rate 0.05 with no momentum. The library default (0.25 with
// momentum 0.5) oscillates on this task in Q16.16.
struct HarNetworkInitializer : public tinymind::DefaultNetworkInitializer<ValueType>
{
    static ValueType initialLearningRate()
    {
        static const ValueType rate(0, 3277); // 0.05 in Q16.16
        return rate;
    }

    static ValueType initialMomentumRate()
    {
        return tinymind::Constants<ValueType>::zero();
    }

    static ValueType initialAccelerationRate()
    {
        return tinymind::Constants<ValueType>::zero();
    }
};

typedef tinymind::FixedPointTransferFunctions<
    ValueType,
    RandomNumberGenerator,
    tinymind::TanhActivationPolicy<ValueType>,
    tinymind::SigmoidActivationPolicy<ValueType>,
    NUMBER_OF_OUTPUTS,
    HarNetworkInitializer,
    tinymind::MeanSquaredErrorCalculator<ValueType, NUMBER_OF_OUTPUTS>,
    tinymind::ZeroToleranceCalculator<ValueType>,
    tinymind::GradientClipByValue<ValueType> > TransferFunctionsType;

typedef tinymind::LstmNeuralNetwork<
    ValueType,
    NUMBER_OF_INPUTS,
    tinymind::HiddenLayers<HIDDEN_NEURONS>,
    NUMBER_OF_OUTPUTS,
    TransferFunctionsType> NeuralNetworkType;

// ---------------------------------------------------------------------------
// float <-> Q helpers
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// Data: a window is WINDOW_LENGTH timesteps of 3 features, plus a label.
// ---------------------------------------------------------------------------

struct Window
{
    double ax[WINDOW_LENGTH];
    double ay[WINDOW_LENGTH];
    double az[WINDOW_LENGTH];
    int    label;
};

// Uniform noise in [-amp, amp].
static inline double noise(double amp)
{
    return amp * ((2.0 * static_cast<double>(rand()) / static_cast<double>(RAND_MAX)) - 1.0);
}

// Synthesize one physically-motivated accelerometer window for `label`.
//
// Accelerations are expressed in units of g and normalized by GSCALE so the
// static gravity component (~1 g) and the dynamic motion both land roughly in
// [-1, 1] -- the stable range for Q16.16 LSTM inputs.
//
// The gait classes use one step rate (3 steps per window), one lateral sway,
// and a vertical waveform with zero mean and unit RMS scaled by the same
// amplitude: a sinusoid for level walking and a sawtooth for stairs. The
// downstairs sawtooth is the upstairs one reversed in time (equivalently,
// negated), so the two classes differ only in sample order.
static void synthesizeWindow(int label, Window& w)
{
    static const double GSCALE = 1.5;            // normalization divisor (g)
    static const double TWO_PI = 2.0 * 3.14159265358979323846;
    static const double STEPS_PER_WINDOW = 3.0;
    static const double VERTICAL_AMPLITUDE = 0.45;
    // RMS of a sawtooth spanning [-1, 1) is 1/sqrt(3).
    static const double SAWTOOTH_RMS = 0.57735026918962576;
    static const double SINE_RMS = 0.70710678118654752;
    const double phase = noise(TWO_PI);          // random window start phase
    const double f = TWO_PI * STEPS_PER_WINDOW / static_cast<double>(WINDOW_LENGTH);

    for (size_t t = 0; t < WINDOW_LENGTH; ++t)
    {
        const double theta = f * static_cast<double>(t) + phase;
        double ax = 0.0, ay = 0.0, az = 0.0;

        if (label == 3)
        {
            // SITTING: near-flat, gravity on the y axis.
            ax = noise(0.04);
            ay = 1.0 + noise(0.04);
            az = noise(0.04);
        }
        else
        {
            double vertical = 0.0;
            if (label == 0)
            {
                vertical = std::sin(theta) / SINE_RMS;
            }
            else
            {
                // Sawtooth in [-1, 1): rises slowly through each step, then
                // drops back in one sample. UPSTAIRS uses its negation (a sharp
                // rise, then a slow decline), which is the same waveform played
                // backwards in time.
                const double cycles = theta / TWO_PI;
                const double ramp = 2.0 * (cycles - std::floor(cycles)) - 1.0;
                const double sawtooth = ramp / SAWTOOTH_RMS;
                vertical = (label == 1) ? -sawtooth : sawtooth;
            }
            ax = 1.0 + 0.10 * std::sin(theta / 2.0 + 0.7) + noise(0.05);  // gravity + slow sway
            ay = 0.25 * std::sin(theta / 2.0) + noise(0.05);              // lateral sway, half the step rate
            az = VERTICAL_AMPLITUDE * vertical + noise(0.05);
        }

        w.ax[t] = ax / GSCALE;
        w.ay[t] = ay / GSCALE;
        w.az[t] = az / GSCALE;
    }

    w.label = label;
}

// Nearest-centroid baseline on per-axis mean and standard deviation (six
// features per window), with centroids from the training split. It ignores
// sample order by construction, so it shows how much of the task can be done
// without memory.
static void windowStats(const Window& w, double* features)
{
    const double* axes[3] = { w.ax, w.ay, w.az };
    for (size_t a = 0; a < 3; ++a)
    {
        double mean = 0.0;
        for (size_t t = 0; t < WINDOW_LENGTH; ++t)
        {
            mean += axes[a][t];
        }
        mean /= static_cast<double>(WINDOW_LENGTH);
        double var = 0.0;
        for (size_t t = 0; t < WINDOW_LENGTH; ++t)
        {
            var += (axes[a][t] - mean) * (axes[a][t] - mean);
        }
        features[a] = mean;
        features[3 + a] = std::sqrt(var / static_cast<double>(WINDOW_LENGTH));
    }
}

static double nearestCentroidAccuracy(const std::vector<Window>& train, const std::vector<Window>& test)
{
    double centroid[NUMBER_OF_OUTPUTS][6] = { {0.0} };
    size_t count[NUMBER_OF_OUTPUTS] = { 0 };
    for (const auto& w : train)
    {
        double f[6];
        windowStats(w, f);
        for (size_t k = 0; k < 6; ++k)
        {
            centroid[w.label][k] += f[k];
        }
        ++count[w.label];
    }
    for (size_t c = 0; c < NUMBER_OF_OUTPUTS; ++c)
    {
        for (size_t k = 0; k < 6; ++k)
        {
            centroid[c][k] /= static_cast<double>(count[c] ? count[c] : 1);
        }
    }
    size_t correct = 0;
    for (const auto& w : test)
    {
        double f[6];
        windowStats(w, f);
        int best = 0;
        double bestDist = 1e300;
        for (size_t c = 0; c < NUMBER_OF_OUTPUTS; ++c)
        {
            double d = 0.0;
            for (size_t k = 0; k < 6; ++k)
            {
                d += (f[k] - centroid[c][k]) * (f[k] - centroid[c][k]);
            }
            if (d < bestDist)
            {
                bestDist = d;
                best = static_cast<int>(c);
            }
        }
        correct += (best == w.label) ? 1U : 0U;
    }
    return static_cast<double>(correct) / static_cast<double>(test.size());
}

// --readout-only: snapshot the LSTM's input, input-bias and recurrent weights
// and restore them after every training step, so only the hidden-to-output
// readout learns. This is the ablation that exposed the v16.9 bug, where the
// gates never trained and only the readout did.
struct ReadoutOnlyFreezer
{
    static const size_t I = NeuralNetworkType::NumberOfInputLayerNeurons;
    static const size_t G = NeuralNetworkType::InputToHiddenConnections;
    static const size_t H = NeuralNetworkType::NumberOfHiddenLayerNeurons;
    static const size_t R = NeuralNetworkType::RecurrentToHiddenConnections;
    ValueType input[I][G];
    ValueType inputBias[G];
    ValueType recurrent[H][R];

    void snapshot(NeuralNetworkType& net)
    {
        for (size_t i = 0; i < I; ++i)
        {
            for (size_t k = 0; k < G; ++k)
            {
                input[i][k] = net.getInputLayerWeightForNeuronAndConnection(i, k);
            }
        }
        for (size_t k = 0; k < G; ++k)
        {
            inputBias[k] = net.getInputLayerBiasNeuronWeightForConnection(k);
        }
        for (size_t j = 0; j < H; ++j)
        {
            for (size_t k = 0; k < R; ++k)
            {
                recurrent[j][k] = net.getRecurrentLayer().getWeightForNeuronAndConnection(j, k);
            }
        }
    }

    void restore(NeuralNetworkType& net) const
    {
        for (size_t i = 0; i < I; ++i)
        {
            for (size_t k = 0; k < G; ++k)
            {
                net.setInputLayerWeightForNeuronAndConnection(i, k, input[i][k]);
            }
        }
        for (size_t k = 0; k < G; ++k)
        {
            net.setInputLayerBiasWeightForConnection(k, inputBias[k]);
        }
        for (size_t j = 0; j < H; ++j)
        {
            for (size_t k = 0; k < R; ++k)
            {
                net.getRecurrentLayer().setWeightForNeuronAndConnection(j, k, recurrent[j][k]);
            }
        }
    }
};

static ReadoutOnlyFreezer gFreezer;
static bool gReadoutOnly = false;

// ---------------------------------------------------------------------------
// Optional CSV loader (long format -- see README). Returns false if absent.
// ---------------------------------------------------------------------------

static bool loadCsv(const std::string& path, std::vector<Window>& out)
{
    std::ifstream in(path);
    if (!in.is_open())
    {
        return false;
    }

    std::string line;
    bool header = true;
    // Accumulate rows keyed by (window_id). Format: window,t,ax,ay,az,label
    std::vector<Window> windows;
    int currentId = -1;
    Window cur;
    bool haveCur = false;

    while (std::getline(in, line))
    {
        if (line.empty())
        {
            continue;
        }
        if (header)
        {
            header = false;
            continue;
        }

        std::stringstream ss(line);
        std::string cell;
        std::vector<std::string> cells;
        while (std::getline(ss, cell, ','))
        {
            cells.push_back(cell);
        }
        if (cells.size() < 6)
        {
            continue;
        }

        const int id    = std::stoi(cells[0]);
        const size_t t  = static_cast<size_t>(std::stoul(cells[1]));
        const double ax = std::stod(cells[2]);
        const double ay = std::stod(cells[3]);
        const double az = std::stod(cells[4]);
        const int label = std::stoi(cells[5]);

        if (t >= WINDOW_LENGTH)
        {
            continue;
        }

        if (id != currentId)
        {
            if (haveCur)
            {
                windows.push_back(cur);
            }
            currentId = id;
            haveCur = true;
        }
        cur.ax[t] = ax;
        cur.ay[t] = ay;
        cur.az[t] = az;
        cur.label = label;
    }
    if (haveCur)
    {
        windows.push_back(cur);
    }

    if (windows.empty())
    {
        return false;
    }
    out.swap(windows);
    return true;
}

// ---------------------------------------------------------------------------
// Run one window through the LSTM. Resets recurrent state first, processes all
// timesteps, and (optionally) trains toward the one-hot target. Returns the
// argmax prediction read at the LAST timestep, and accumulates |error|.
// ---------------------------------------------------------------------------

static int processWindow(NeuralNetworkType& net, const Window& w, bool train, double& errAccum)
{
    ValueType input[NUMBER_OF_INPUTS];
    ValueType target[NUMBER_OF_OUTPUTS];
    ValueType learned[NUMBER_OF_OUTPUTS];

    for (size_t k = 0; k < NUMBER_OF_OUTPUTS; ++k)
    {
        target[k] = toQ(w.label == static_cast<int>(k) ? 1.0 : 0.0);
    }

    // CRITICAL: the LSTM is stateful across feedForward calls -- clear the
    // recurrent cell state at the start of every window.
    net.resetState();

    for (size_t t = 0; t < WINDOW_LENGTH; ++t)
    {
        input[0] = toQ(w.ax[t]);
        input[1] = toQ(w.ay[t]);
        input[2] = toQ(w.az[t]);

        net.feedForward(input);

        // Supervise toward the (constant) window label at every timestep: this
        // gives the recurrent net a strong, repeated gradient signal and teaches
        // it to settle on the classification as evidence accumulates. The
        // reported prediction is still read at the LAST timestep below.
        const ValueType err = net.calculateError(target);
        if (t == WINDOW_LENGTH - 1)
        {
            errAccum += std::fabs(fromQ(err));
        }
        // Train on every step. Skipping steps whose error is within the
        // fixed-point zero tolerance stopped this network from learning the
        // order-dependent classes at all (50% test accuracy).
        if (train)
        {
            net.trainNetwork(target);
            if (gReadoutOnly)
            {
                gFreezer.restore(net);
            }
        }
    }

    net.getLearnedValues(learned);
    int pred = 0;
    double best = fromQ(learned[0]);
    for (size_t k = 1; k < NUMBER_OF_OUTPUTS; ++k)
    {
        const double v = fromQ(learned[k]);
        if (v > best) { best = v; pred = static_cast<int>(k); }
    }
    return pred;
}

static const unsigned WEIGHT_INIT_SEED = 1U; // see main()

int main(int argc, char* argv[])
{
    gReadoutOnly = (argc > 1) && (std::string(argv[1]) == "--readout-only");

    // Two seeds: the initial weights come from WEIGHT_INIT_SEED and everything
    // else (data, shuffles, training samples) from the seed set just below.
    // WEIGHT_INIT_SEED = 1 reproduces this example's published results: the
    // network used to be a global, constructed before main() seeded rand(), so it
    // drew from glibc's default seed of 1.
    std::srand(WEIGHT_INIT_SEED);
    static NeuralNetworkType net;
    gFreezer.snapshot(net);
    if (gReadoutOnly)
    {
        std::cout << "--readout-only: LSTM input, gate and recurrent weights stay at their initial values." << std::endl;
    }

    std::srand(7U);

    std::vector<Window> train;
    std::vector<Window> test;

    std::vector<Window> loaded;
    if (loadCsv("har.csv", loaded))
    {
        std::cout << "Loaded " << loaded.size() << " windows from har.csv." << std::endl;
        // Deterministic shuffle + 80/20 split.
        for (size_t i = loaded.size(); i > 1; --i)
        {
            const size_t j = static_cast<size_t>(rand()) % i;
            std::swap(loaded[i - 1], loaded[j]);
        }
        const size_t nTrain = (loaded.size() * 4) / 5;
        train.assign(loaded.begin(), loaded.begin() + nTrain);
        test.assign(loaded.begin() + nTrain, loaded.end());
    }
    else
    {
        std::cout << "har.csv not found -- synthesizing accelerometer windows."
                  << std::endl;
        for (int label = 0; label < static_cast<int>(NUMBER_OF_OUTPUTS); ++label)
        {
            for (size_t i = 0; i < TRAIN_PER_CLASS; ++i)
            {
                Window w; synthesizeWindow(label, w); train.push_back(w);
            }
            for (size_t i = 0; i < TEST_PER_CLASS; ++i)
            {
                Window w; synthesizeWindow(label, w); test.push_back(w);
            }
        }
    }
    std::cout << "Train: " << train.size() << "  Test: " << test.size() << std::endl;

    // Representative window per class for the trace plot (synthesize fresh so we
    // always have one per class even when loading a custom CSV).
    {
        std::ofstream samples("har_samples.csv");
        samples << "class,t,ax,ay,az" << std::endl;
        for (int label = 0; label < static_cast<int>(NUMBER_OF_OUTPUTS); ++label)
        {
            Window w; synthesizeWindow(label, w);
            for (size_t t = 0; t < WINDOW_LENGTH; ++t)
            {
                samples << CLASS_NAMES[label] << "," << t << ","
                        << w.ax[t] << "," << w.ay[t] << "," << w.az[t] << std::endl;
            }
        }
    }

    // ---- Training ----------------------------------------------------------
    const unsigned epochs      = 60U;
    double errSum = 0.0;
    size_t errCount = 0;

    std::ofstream lossCsv("har_loss.csv");
    lossCsv << "epoch,avg_err" << std::endl;

    for (unsigned epoch = 0; epoch < epochs; ++epoch)
    {
        // Shuffle the training order each epoch (deterministic, seeded above).
        for (size_t i = train.size(); i > 1; --i)
        {
            const size_t j = static_cast<size_t>(rand()) % i;
            std::swap(train[i - 1], train[j]);
        }

        for (const auto& w : train)
        {
            processWindow(net, w, /*train=*/true, errSum);
            ++errCount;
        }

        const double avg = errCount ? (errSum / static_cast<double>(errCount)) : 0.0;
        std::cout << "epoch " << std::setw(3) << (epoch + 1)
                  << "   avg|err| = " << std::fixed << std::setprecision(4)
                  << avg << std::endl;
        lossCsv << (epoch + 1) << "," << avg << std::endl;
        errSum = 0.0;
        errCount = 0;
    }
    lossCsv.close();

    // ---- Evaluation --------------------------------------------------------
    int confusion[NUMBER_OF_OUTPUTS][NUMBER_OF_OUTPUTS] = { {0} };
    double evalErr = 0.0;
    size_t correct = 0;
    for (const auto& w : test)
    {
        const int pred = processWindow(net, w, /*train=*/false, evalErr);
        ++confusion[w.label][pred];
        if (pred == w.label) ++correct;
    }

    {
        std::ofstream cm("har_confusion.csv");
        cm << "actual,pred_walking,pred_upstairs,pred_downstairs,pred_sitting" << std::endl;
        for (int r = 0; r < static_cast<int>(NUMBER_OF_OUTPUTS); ++r)
        {
            cm << CLASS_NAMES[r];
            for (int c = 0; c < static_cast<int>(NUMBER_OF_OUTPUTS); ++c)
            {
                cm << "," << confusion[r][c];
            }
            cm << std::endl;
        }
    }

    const double acc = static_cast<double>(correct) / static_cast<double>(test.size());
    std::cout << "\nTest accuracy: " << std::fixed << std::setprecision(4)
              << acc << "  (" << correct << "/" << test.size() << ")" << std::endl;
    std::cout << "Baseline, nearest centroid on per-axis mean + std (ignores sample order): "
              << std::fixed << std::setprecision(4) << nearestCentroidAccuracy(train, test) << std::endl;
    std::cout << "Baseline, chance: " << std::fixed << std::setprecision(4)
              << (1.0 / static_cast<double>(NUMBER_OF_OUTPUTS)) << std::endl;

    return 0;
}
