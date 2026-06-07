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

// Human Activity Recognition (HAR) classifier built on TinyMind.
//
// Inspired by the UCI "Human Activity Recognition Using Smartphones" dataset:
//   https://archive.ics.uci.edu/dataset/240
//
// This is a RECURRENT classifier: an LSTM network reads a 32-timestep window of
// tri-axial accelerometer samples (ax, ay, az) one timestep at a time and
// classifies the whole window into one of four activities at the LAST timestep:
//
//   0 WALKING            periodic ~2 Hz oscillation, all axes, gravity baseline
//   1 WALKING_UPSTAIRS   lower freq, larger amplitude, strong vertical (z)
//   2 SITTING            near-flat, small noise, gravity on the y axis
//   3 STANDING           near-flat, small noise, gravity on the x axis
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
    "walking", "upstairs", "sitting", "standing"
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
            (static_cast<FullWidthValueType>(rand()) % (2 * quarter + 1)) - quarter;
        return ValueType(weight);
    }
};

typedef tinymind::FixedPointTransferFunctions<
    ValueType,
    RandomNumberGenerator,
    tinymind::TanhActivationPolicy<ValueType>,
    tinymind::SigmoidActivationPolicy<ValueType>,
    NUMBER_OF_OUTPUTS,
    tinymind::DefaultNetworkInitializer<ValueType>,
    tinymind::MeanSquaredErrorCalculator<ValueType, NUMBER_OF_OUTPUTS>,
    tinymind::ZeroToleranceCalculator<ValueType>,
    tinymind::GradientClipByValue<ValueType> > TransferFunctionsType;

typedef tinymind::LstmNeuralNetwork<
    ValueType,
    NUMBER_OF_INPUTS,
    tinymind::HiddenLayers<HIDDEN_NEURONS>,
    NUMBER_OF_OUTPUTS,
    TransferFunctionsType> NeuralNetworkType;

static NeuralNetworkType gNet;

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
static void synthesizeWindow(int label, Window& w)
{
    static const double GSCALE = 1.5;            // normalization divisor (g)
    static const double TWO_PI = 2.0 * 3.14159265358979323846;
    const double phase = noise(TWO_PI);          // random window start phase

    for (size_t t = 0; t < WINDOW_LENGTH; ++t)
    {
        const double tt = static_cast<double>(t);
        double ax = 0.0, ay = 0.0, az = 0.0;

        switch (label)
        {
        case 0: // WALKING: ~2 Hz periodic on all axes + 2nd harmonic, g on x
        {
            const double f = TWO_PI * 2.0 / static_cast<double>(WINDOW_LENGTH) * 2.0;
            ax = 1.0 + 0.35 * std::sin(f * tt + phase)
                     + 0.12 * std::sin(2.0 * f * tt + phase);
            ay = 0.30 * std::sin(f * tt + phase + 1.0)
                     + 0.10 * std::sin(2.0 * f * tt + phase);
            az = 0.30 * std::cos(f * tt + phase)
                     + 0.10 * std::cos(2.0 * f * tt + phase);
            ax += noise(0.05); ay += noise(0.05); az += noise(0.05);
            break;
        }
        case 1: // WALKING_UPSTAIRS: lower freq, bigger amplitude, strong z swing.
                // Climbing leans the device forward, so gravity is shared
                // between x and z rather than resting fully on x (as in flat
                // walking) -- a distinct static signature plus distinct dynamics.
        {
            const double f = TWO_PI * 1.0 / static_cast<double>(WINDOW_LENGTH) * 2.0;
            ax = 0.55 + 0.25 * std::sin(f * tt + phase);
            ay = 0.30 * std::sin(f * tt + phase + 0.5);
            az = 0.55                                 // forward-lean gravity bias
                     + 0.70 * std::sin(f * tt + phase) // dominant vertical swing
                     + 0.20 * std::sin(2.0 * f * tt + phase);
            ax += noise(0.06); ay += noise(0.06); az += noise(0.06);
            break;
        }
        case 2: // SITTING: near-flat, gravity on y axis
        {
            ax = 0.0  + noise(0.04);
            ay = 1.0  + noise(0.04);
            az = 0.0  + noise(0.04);
            break;
        }
        case 3: // STANDING: near-flat, gravity on z axis (different orientation
                // from both SITTING's y and WALKING's x baseline)
        {
            ax = 0.0  + noise(0.04);
            ay = 0.0  + noise(0.04);
            az = 1.0  + noise(0.04);
            break;
        }
        default:
            break;
        }

        w.ax[t] = ax / GSCALE;
        w.ay[t] = ay / GSCALE;
        w.az[t] = az / GSCALE;
    }

    w.label = label;
}

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

static int processWindow(const Window& w, bool train, double& errAccum)
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
    gNet.resetState();

    for (size_t t = 0; t < WINDOW_LENGTH; ++t)
    {
        input[0] = toQ(w.ax[t]);
        input[1] = toQ(w.ay[t]);
        input[2] = toQ(w.az[t]);

        gNet.feedForward(input);

        // Supervise toward the (constant) window label at every timestep: this
        // gives the recurrent net a strong, repeated gradient signal and teaches
        // it to settle on the classification as evidence accumulates. The
        // reported prediction is still read at the LAST timestep below.
        const ValueType err = gNet.calculateError(target);
        if (t == WINDOW_LENGTH - 1)
        {
            errAccum += std::fabs(fromQ(err));
        }
        if (train &&
            !NeuralNetworkType::NeuralNetworkTransferFunctionsPolicy::isWithinZeroTolerance(err))
        {
            gNet.trainNetwork(target);
        }
    }

    gNet.getLearnedValues(learned);
    int pred = 0;
    double best = fromQ(learned[0]);
    for (size_t k = 1; k < NUMBER_OF_OUTPUTS; ++k)
    {
        const double v = fromQ(learned[k]);
        if (v > best) { best = v; pred = static_cast<int>(k); }
    }
    return pred;
}

int main(int argc, char* argv[])
{
    (void)argc;
    (void)argv;

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
    const unsigned epochs      = 45U;
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
            processWindow(w, /*train=*/true, errSum);
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
        const int pred = processWindow(w, /*train=*/false, evalErr);
        ++confusion[w.label][pred];
        if (pred == w.label) ++correct;
    }

    {
        std::ofstream cm("har_confusion.csv");
        cm << "actual,pred_walk,pred_upstairs,pred_sitting,pred_standing" << std::endl;
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

    return 0;
}
