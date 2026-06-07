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

// Air-quality hourly pollutant forecaster built on TinyMind.
//
// Inspired by the UCI Air Quality dataset:
//   https://archive.ics.uci.edu/dataset/360
//
// An LSTM (1 input -> 16 hidden -> 1 output) in Q16.16 fixed-point learns to
// predict the next hour's normalized pollutant concentration (e.g. CO) from the
// current hour. Hidden layers use tanh; the single output uses sigmoid because
// the target series is normalized to [0, 1].
//
// The real AirQualityUCI.csv (';'-separated, decimal commas, -200 for missing)
// is loaded if it is present in the run directory; otherwise the example falls
// back to a SYNTHETIC hourly series so it runs fully offline (a strong 24-hour
// daily cycle + a slow multi-day trend + mild noise). See the README for the
// file format. The irregular-sampling sibling lives in examples/cfc_sequence.
//
// Outputs (header-row CSVs, written to the working directory at runtime):
//   airq_loss.csv      epoch,avg_err          training loss over epochs
//   airq_forecast.csv  hour,true,predicted    one-step-ahead over a held-out tail

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
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

#define TRAINING_EPOCHS 8000U
#define NUM_SAMPLES 2400U     // 100 days of hourly samples
#define HOURS_PER_DAY 24U
#define HOLDOUT_HOURS 200U    // last 200 hours held out for one-step eval
#define LEARNING_RATE 0.3     // lower than the Q16.16 default (0.25 too coarse here)
#define TRAIN_WINDOW 24U      // reset recurrent state every day during training
#define RANDOM_SEED 7U

typedef tinymind::QValue<16, 16, true> ValueType;
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

// Same TransferFunctions form as examples/lstm_sinusoid: tanh hidden, sigmoid
// output (target normalized to [0, 1]), gradient clipping for recurrent
// stability. Output count = 1.
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

// Convert a [0, 1] double into a Q16.16 ValueType.
static ValueType toQ(double v)
{
    const FullWidthValueType raw =
        static_cast<FullWidthValueType>(v * (1 << ValueType::NumberOfFractionalBits));
    return ValueType(raw);
}

// Convert a Q16.16 ValueType back to a double.
static double fromQ(const ValueType& v)
{
    return static_cast<double>(v.getValue()) /
           static_cast<double>(1 << ValueType::NumberOfFractionalBits);
}

// Try to load the real UCI AirQualityUCI.csv. It is ';'-separated, uses a
// decimal comma, and marks missing values with -200. We read the first numeric
// pollutant column (CO(GT)) and skip rows whose value is missing. Returns true
// if at least a few hundred valid samples were read.
static bool loadRealSeries(std::vector<double>& out)
{
    std::ifstream in("AirQualityUCI.csv");
    if (!in.is_open())
    {
        return false;
    }

    std::string line;
    // Header row.
    if (!std::getline(in, line))
    {
        return false;
    }

    while (std::getline(in, line))
    {
        // Replace decimal commas with dots, treat ';' as the field separator.
        std::vector<std::string> fields;
        std::string field;
        for (size_t i = 0; i < line.size(); ++i)
        {
            const char c = line[i];
            if (c == ';')
            {
                fields.push_back(field);
                field.clear();
            }
            else if (c == ',')
            {
                field.push_back('.');
            }
            else if (c != '\r')
            {
                field.push_back(c);
            }
        }
        fields.push_back(field);

        // Column 0 = Date, 1 = Time, 2 = CO(GT). Need at least 3 columns.
        if (fields.size() < 3 || fields[2].empty())
        {
            continue;
        }

        std::istringstream iss(fields[2]);
        double value = 0.0;
        if (!(iss >> value))
        {
            continue;
        }
        if (value <= -200.0)
        {
            continue; // missing
        }
        out.push_back(value);
    }

    return out.size() >= 300;
}

// Synthesize a realistic hourly pollutant (CO-like) series: a strong 24-hour
// daily cycle with morning/evening rush-hour peaks, a slow multi-day trend, and
// mild noise. Values are in arbitrary "mg/m^3"-ish units before normalization.
static void synthSeries(std::vector<double>& out)
{
    out.resize(NUM_SAMPLES);
    for (size_t t = 0; t < NUM_SAMPLES; ++t)
    {
        const double hourOfDay = static_cast<double>(t % HOURS_PER_DAY);
        const double phase = 2.0 * M_PI * hourOfDay / static_cast<double>(HOURS_PER_DAY);

        // Twin rush-hour peaks (~8am and ~6pm) on top of a daily baseline.
        const double daily =
            1.6
            + 0.9 * std::exp(-0.5 * std::pow((hourOfDay - 8.0) / 2.0, 2.0))   // morning
            + 1.1 * std::exp(-0.5 * std::pow((hourOfDay - 18.0) / 2.2, 2.0))  // evening
            + 0.3 * std::sin(phase);                                          // smooth diurnal

        // Gentle multi-day trend (weather fronts, weekly pattern). Kept small so
        // the series stays close to periodic and the next-hour value is well
        // determined by the current hour plus the LSTM's phase memory.
        const double days = static_cast<double>(t) / static_cast<double>(HOURS_PER_DAY);
        const double trend = 0.25 * std::sin(2.0 * M_PI * days / 7.0);

        // Mild noise.
        const double noise = 0.04 * ((static_cast<double>(rand()) / RAND_MAX) - 0.5);

        out[t] = daily + trend + noise;
    }
}

int main(int argc, char** argv)
{
    srand(RANDOM_SEED);

    // Optional CLI overrides for quick experimentation:
    //   argv[1] = training epochs, argv[2] = learning rate.
    unsigned trainingEpochs = TRAINING_EPOCHS;
    double learningRate = LEARNING_RATE;
    if (argc > 1) trainingEpochs = static_cast<unsigned>(std::atoi(argv[1]));
    if (argc > 2) learningRate = std::atof(argv[2]);
    lstmNet.setLearningRate(toQ(learningRate));

    // Acquire the raw series (real file if present, else synthetic).
    std::vector<double> raw;
    const bool usedReal = loadRealSeries(raw);
    if (!usedReal)
    {
        synthSeries(raw);
        std::cout << "Using synthetic hourly air-quality series ("
                  << raw.size() << " samples)." << std::endl;
    }
    else
    {
        std::cout << "Loaded real AirQualityUCI.csv ("
                  << raw.size() << " valid CO(GT) samples)." << std::endl;
    }

    const size_t numSamples = raw.size();

    // Normalize the series to [0, 1]; track min/max so we can de-normalize the
    // forecast back into real-ish units for reporting.
    double dataMin = raw[0];
    double dataMax = raw[0];
    for (size_t t = 0; t < numSamples; ++t)
    {
        if (raw[t] < dataMin) dataMin = raw[t];
        if (raw[t] > dataMax) dataMax = raw[t];
    }
    const double range = (dataMax > dataMin) ? (dataMax - dataMin) : 1.0;

    std::vector<double> normDouble(numSamples);
    std::vector<ValueType> series(numSamples);
    for (size_t t = 0; t < numSamples; ++t)
    {
        const double n = (raw[t] - dataMin) / range;
        normDouble[t] = n;
        series[t] = toQ(n);
    }

    // Train on everything except the held-out tail.
    const size_t holdout = (numSamples > HOLDOUT_HOURS + 1)
                               ? HOLDOUT_HOURS : (numSamples / 4);
    const size_t trainEnd = numSamples - holdout; // [0, trainEnd) used for training

    ValueType input[1];
    ValueType target[1];
    ValueType learnedValues[1];

    std::cout << "Training LSTM forecaster: " << trainingEpochs
              << " epochs over " << (trainEnd - 1)
              << " sequential pairs (holdout=" << holdout << " hours)..."
              << std::endl;

    std::ofstream lossCsv("airq_loss.csv");
    lossCsv << "epoch,avg_err" << std::endl;

    // Reset the recurrent state at window boundaries so each BPTT segment is a
    // clean one-day stretch of the series. On a 2000+ step unbroken sequence the
    // gradient washes out and the net collapses to predicting the mean; windowed
    // resets keep the daily-cycle signal trainable.
    const size_t WINDOW = TRAIN_WINDOW;

    for (unsigned epoch = 0; epoch < trainingEpochs; ++epoch)
    {
        double epochAbsErr = 0.0;
        size_t count = 0;

        lstmNet.resetState();
        for (size_t t = 0; t + 1 < trainEnd; ++t)
        {
            if (t > 0 && (t % WINDOW) == 0)
            {
                lstmNet.resetState();
            }

            input[0] = series[t];
            target[0] = series[t + 1];

            lstmNet.feedForward(&input[0]);
            const ValueType error = lstmNet.calculateError(&target[0]);

            epochAbsErr += std::fabs(fromQ(error));
            ++count;

            if (!TransferFunctionsType::isWithinZeroTolerance(error))
            {
                lstmNet.trainNetwork(&target[0]);
            }
        }

        const double avgErr = (count > 0) ? (epochAbsErr / static_cast<double>(count)) : 0.0;
        lossCsv << epoch << "," << avgErr << std::endl;
    }
    lossCsv.close();

    std::cout << "Training complete." << std::endl;

    // One-step-ahead evaluation over the held-out tail. Warm the LSTM state by
    // streaming the hours leading up to the holdout through feedForward (state
    // reset first, matching the windowed-training convention), then forecast
    // each held-out hour from the true current hour (one-step, not free-running).
    lstmNet.resetState();
    const size_t warmStart = (trainEnd > WINDOW) ? (trainEnd - WINDOW) : 0;
    for (size_t t = warmStart; t + 1 < trainEnd; ++t)
    {
        input[0] = series[t];
        lstmNet.feedForward(&input[0]);
    }

    std::ofstream fc("airq_forecast.csv");
    fc << "hour,true,predicted" << std::endl;

    double sumAbsErr = 0.0;
    size_t evalCount = 0;
    for (size_t t = trainEnd - 1; t + 1 < numSamples; ++t)
    {
        input[0] = series[t];
        lstmNet.feedForward(&input[0]);
        lstmNet.getLearnedValues(&learnedValues[0]);

        const double predNorm = fromQ(learnedValues[0]);
        const double trueNorm = normDouble[t + 1];

        // De-normalize back to real-ish units for a friendlier plot.
        const double predReal = dataMin + predNorm * range;
        const double trueReal = dataMin + trueNorm * range;

        fc << (t + 1) << "," << trueReal << "," << predReal << std::endl;

        sumAbsErr += std::fabs(predReal - trueReal);
        ++evalCount;
    }
    fc.close();

    const double mae = (evalCount > 0) ? (sumAbsErr / static_cast<double>(evalCount)) : 0.0;
    std::cout << "One-step-ahead forecast over " << evalCount
              << " held-out hours. MAE = " << mae
              << " (real units; series range " << dataMin << ".." << dataMax << ")."
              << std::endl;
    std::cout << "Wrote airq_loss.csv and airq_forecast.csv." << std::endl;

    return 0;
}
