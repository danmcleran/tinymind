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

// Vibration anomaly detection that learns "normal" on the device.
//
// A synthetic machine (3,200 samples per second, int16) runs at a speed that
// wanders between 28 and 32 Hz and a load that moves its level by +-6 dB. The
// stream is processed the way a sensor node would process it, frame by frame
// (256 samples, hop 128), with no buffering:
//
//   1. QLogMelSpectrogram<256, 32> -> 32 log-mel bands (int16, Q8), less
//      the frame's mean band (the spectrum's shape, not its level), mapped
//      onto int8 features at 1/8 nat per step
//   2. learn      120 s of normal running: QKMeansAnomalyTrainer<32, 8>
//                 learns 8 centroids, one frame at a time
//   3. calibrate   60 s: the trainer measures each cluster's radius
//   4. threshold   60 s: the largest score seen, times 1.25, is the alarm
//                 threshold
//   5. test        normal running, then three faults whose severity grows from
//                 0 to full over 60 s each, with normal running in between:
//                 an outer-race bearing defect (impacts at 3.57x speed ringing
//                 a 900 Hz resonance), imbalance (the 1x component grows to 3x
//                 its normal size), and mechanical looseness (half-order and
//                 high-order harmonics)
//
// For comparison, the same trainer with feature weighting off (a plain
// Euclidean distance) learns alongside, and a diagonal Gaussian
// (QDiagGaussianAnomaly, one component) is fitted on the host to the same
// 120 s of learning frames; both are thresholded the same way. Steps 1 to 4 and the k-means score use integer arithmetic only.
//
// Writes output/vibration_anomaly.csv (one row per test frame) and
// output/vibration_anomaly_summary.csv.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <random>
#include <string>
#include <vector>

#include "qanomaly.hpp"
#include "qcalibration.hpp"
#include "qmel.hpp"

namespace
{
    const double kPi = 3.14159265358979323846;
    const double kRate = 3200.0;
    const std::size_t kN = 256;
    const std::size_t kHop = 128;
    const std::size_t kBands = 32;
    const std::size_t kClusters = 8;

    enum Condition
    {
        Normal = 0,
        Bearing = 1,
        Imbalance = 2,
        Looseness = 3
    };

    const char* const kConditionName[4] = {"normal", "bearing", "imbalance", "looseness"};

    struct Segment
    {
        const char* phase;
        Condition condition;
        double seconds;
    };

    const Segment kSegments[] = {
        {"learn", Normal, 120.0},
        {"calibrate", Normal, 60.0},
        {"threshold", Normal, 60.0},
        {"test", Normal, 60.0},
        {"test", Bearing, 60.0},
        {"test", Normal, 30.0},
        {"test", Imbalance, 60.0},
        {"test", Normal, 30.0},
        {"test", Looseness, 60.0}};

    // The machine: state carried across segments so the stream is continuous.
    struct Machine
    {
        std::mt19937 rng;
        std::normal_distribution<double> gauss;
        double phase;
        double speed;      // Hz
        double loadDb;
        double ring;
        double ringPrev;
        double impactPhase;

        Machine() : rng(11), gauss(0.0, 1.0), phase(0.0), speed(30.0), loadDb(0.0), ring(0.0), ringPrev(0.0),
                    impactPhase(0.0)
        {
        }

        int16_t next(Condition condition, double severity)
        {
            // Slow random walks of speed and load.
            speed += 0.0005 * gauss(rng);
            speed = std::max(28.0, std::min(32.0, speed));
            loadDb += 0.002 * gauss(rng);
            loadDb = std::max(-6.0, std::min(6.0, loadDb));

            phase += 2.0 * kPi * speed / kRate;
            if (phase > 2.0 * kPi)
            {
                phase -= 2.0 * kPi;
            }
            double s = 0.0;
            const double oneX = (condition == Imbalance) ? 1.0 + 2.0 * severity : 1.0;
            s += oneX * std::sin(phase);
            s += 0.35 * std::sin(2.0 * phase + 0.4);
            s += 0.15 * std::sin(3.0 * phase + 1.1);
            s += 0.08 * std::sin(4.0 * phase + 2.0);
            if (condition == Looseness)
            {
                s += 0.25 * severity * std::sin(0.5 * phase + 0.3);
                for (int h = 5; h <= 12; ++h)
                {
                    s += 0.12 * severity * std::sin(h * phase + 0.7 * h);
                }
            }
            // Outer-race defect: impacts at 3.57x shaft speed ring a damped
            // 900 Hz resonance.
            double impact = 0.0;
            if (condition == Bearing)
            {
                impactPhase += 3.57 * speed / kRate;
                if (impactPhase >= 1.0)
                {
                    impactPhase -= 1.0;
                    impact = 4.0 * severity;
                }
            }
            const double omega = 2.0 * kPi * 900.0 / kRate;
            const double r = 0.985;
            const double nextRing = 2.0 * r * std::cos(omega) * ring - r * r * ringPrev + impact;
            ringPrev = ring;
            ring = nextRing;
            s += 0.3 * ring;
            s += 0.04 * gauss(rng);

            const double amplitude = std::pow(10.0, (loadDb - 20.0) / 20.0) * 32767.0 / 2.0;
            return static_cast<int16_t>(std::max(-32768.0, std::min(32767.0, std::round(s * amplitude))));
        }
    };

    // Log-mel (Q8 nats) to int8 features at 1/8 nat per step, relative to the
    // frame's mean band: q = (ln E_m - mean_m ln E_m) * 8. Removing the mean
    // keeps the spectrum's shape and drops its overall level, which the load
    // moves by +-6 dB in normal running.
    void toFeatures(const int16_t* logMel, int8_t* features)
    {
        int32_t mean = 0;
        for (std::size_t m = 0; m < kBands; ++m)
        {
            mean += logMel[m];
        }
        mean /= static_cast<int32_t>(kBands);
        for (std::size_t m = 0; m < kBands; ++m)
        {
            int32_t q = (static_cast<int32_t>(logMel[m]) - mean + 16) >> 5;
            q = (q < -128) ? -128 : ((q > 127) ? 127 : q);
            features[m] = static_cast<int8_t>(q);
        }
    }

    struct Frame
    {
        double time;
        Condition condition;
        double severity;
        int32_t score[3]; // weighted k-means, Euclidean k-means, Gaussian
    };

    double auc(const std::vector<int32_t>& normal, const std::vector<int32_t>& anomalous)
    {
        std::vector<int32_t> sorted(normal);
        std::sort(sorted.begin(), sorted.end());
        double wins = 0.0;
        for (std::size_t i = 0; i < anomalous.size(); ++i)
        {
            const std::size_t below = static_cast<std::size_t>(
                std::lower_bound(sorted.begin(), sorted.end(), anomalous[i]) - sorted.begin());
            const std::size_t upto = static_cast<std::size_t>(
                std::upper_bound(sorted.begin(), sorted.end(), anomalous[i]) - sorted.begin());
            wins += below + 0.5 * (upto - below);
        }
        return wins / (static_cast<double>(anomalous.size()) * sorted.size());
    }
}

int main()
{
    static int16_t window[kN];
    static int16_t twiddleCos[kN / 2];
    static int16_t twiddleSin[kN / 2];
    static uint16_t bandStart[kBands];
    static uint16_t bandLength[kBands];
    static uint16_t weights[2 * (kN / 2 + 1)];
    tinymind::buildQHannWindow(kN, window);
    tinymind::buildQFFTTwiddles(kN, twiddleCos, twiddleSin);
    tinymind::buildQMelFilterbank(kN, kRate, kBands, 5.0, 1600.0, bandStart, bandLength, weights, 2 * (kN / 2 + 1));
    const tinymind::QLogMelSpectrogram<kN, kBands> mel = {window, twiddleCos, twiddleSin, bandStart, bandLength,
                                                          weights, 0, -32768};
    tinymind::QMelWorkspace<kN> workspace;
    static tinymind::QKMeansAnomalyTrainer<kBands, kClusters> trainer;
    // The same trainer with a plain Euclidean distance, for comparison.
    static tinymind::QKMeansAnomalyTrainer<kBands, kClusters> euclidean;
    euclidean.setFeatureWeighting(false);

    // Host-side statistics of the learning frames, for the Gaussian.
    std::vector<double> sum(kBands, 0.0);
    std::vector<double> sumSquares(kBands, 0.0);
    std::size_t learnFrames = 0;
    int8_t gaussMeans[kBands] = {0};
    uint32_t gaussPrecision[kBands] = {0};
    int32_t gaussBias[1] = {0};
    bool gaussianBuilt = false;
    // Largest score in the threshold phase, and the threshold, per detector.
    int32_t maxScore[3] = {0, 0, INT32_MIN};
    int32_t threshold[3] = {0, 0, 0};
    std::vector<Frame> testFrames;

    Machine machine;
    int16_t ringBuffer[kN];
    std::size_t filled = 0;
    std::size_t sinceFrame = 0;
    int16_t previousSample = 0;
    double time = 0.0;
    std::string lastPhase;
    for (std::size_t s = 0; s < sizeof(kSegments) / sizeof(kSegments[0]); ++s)
    {
        const Segment& seg = kSegments[s];
        const std::string phase(seg.phase);
        if (phase == "threshold" && lastPhase != "threshold")
        {
            // Fit the Gaussian to the learning frames (float, on the host).
            const double scale = 0.125;
            double means[kBands];
            double variances[kBands];
            const double one[1] = {1.0};
            for (std::size_t m = 0; m < kBands; ++m)
            {
                means[m] = scale * (sum[m] / learnFrames);
                variances[m] = scale * scale * (sumSquares[m] / learnFrames - (sum[m] / learnFrames) * (sum[m] / learnFrames));
            }
            tinymind::buildQDiagGaussianAnomaly(1, kBands, means, variances, one, static_cast<float>(scale), 0, 0.25,
                                                gaussMeans, gaussPrecision, gaussBias);
            gaussianBuilt = true;
        }
        if (phase == "test" && lastPhase != "test")
        {
            threshold[0] = static_cast<int32_t>(1.25 * maxScore[0]);
            threshold[1] = static_cast<int32_t>(1.25 * maxScore[1]);
            // The Gaussian's score is a log-likelihood: give it the same 25%
            // margin on the part of the score above its minimum, the bias.
            threshold[2] = gaussBias[0] + static_cast<int32_t>(1.25 * (maxScore[2] - gaussBias[0]));
        }
        lastPhase = phase;

        const std::size_t samples = static_cast<std::size_t>(seg.seconds * kRate);
        for (std::size_t n = 0; n < samples; ++n)
        {
            const double severity = static_cast<double>(n) / samples;
            const int16_t x = machine.next(seg.condition, severity);
            time += 1.0 / kRate;
            // Slide the frame: keep the last kN samples in order.
            if (filled < kN)
            {
                ringBuffer[filled++] = x;
            }
            else
            {
                for (std::size_t i = 1; i < kN; ++i)
                {
                    ringBuffer[i - 1] = ringBuffer[i];
                }
                ringBuffer[kN - 1] = x;
            }
            if (filled < kN || ++sinceFrame < kHop)
            {
                continue;
            }
            sinceFrame = 0;

            int16_t logMel[kBands];
            mel.forward(ringBuffer, previousSample, logMel, workspace);
            previousSample = ringBuffer[kHop - 1];
            int8_t features[kBands];
            toFeatures(logMel, features);

            if (phase == "learn")
            {
                trainer.learn(features);
                euclidean.learn(features);
                for (std::size_t m = 0; m < kBands; ++m)
                {
                    sum[m] += features[m];
                    sumSquares[m] += static_cast<double>(features[m]) * features[m];
                }
                ++learnFrames;
                continue;
            }
            if (phase == "calibrate")
            {
                trainer.calibrate(features);
                euclidean.calibrate(features);
                continue;
            }
            const tinymind::QDiagGaussianAnomaly<kBands, 1> gaussian = {gaussMeans, gaussPrecision, gaussBias};
            const int32_t scores[3] = {trainer.model().score(features), euclidean.model().score(features),
                                       gaussianBuilt ? gaussian.score(features) : 0};
            if (phase == "threshold")
            {
                for (int d = 0; d < 3; ++d)
                {
                    maxScore[d] = std::max(maxScore[d], scores[d]);
                }
                continue;
            }
            const Frame f = {time, seg.condition, (seg.condition == Normal) ? 0.0 : severity,
                             {scores[0], scores[1], scores[2]}};
            testFrames.push_back(f);
        }
    }

    std::FILE* csv = std::fopen("vibration_anomaly.csv", "w");
    std::FILE* summary = std::fopen("vibration_anomaly_summary.csv", "w");
    if (csv == nullptr || summary == nullptr)
    {
        std::fprintf(stderr, "cannot write the output CSVs\n");
        return 1;
    }
    std::fprintf(csv, "time_s,condition,severity,kmeans_score,kmeans_threshold,euclidean_score,euclidean_threshold,"
                      "gaussian_score,gaussian_threshold\n");
    for (std::size_t i = 0; i < testFrames.size(); ++i)
    {
        const Frame& f = testFrames[i];
        std::fprintf(csv, "%.3f,%s,%.4f,%d,%d,%d,%d,%d,%d\n", f.time, kConditionName[f.condition], f.severity,
                     f.score[0], threshold[0], f.score[1], threshold[1], f.score[2], threshold[2]);
    }
    std::fclose(csv);

    std::printf("learned %zu clusters from %zu frames; thresholds: k-means %d and %d (Q8 radii^2), Gaussian %d (Q8 nats)\n",
                trainer.seededClusters(), learnFrames, threshold[0], threshold[1], threshold[2]);
    std::fprintf(summary, "detector,condition,frames,auc,alarm_rate,first_sustained_alarm_severity\n");
    const char* detectorName[3] = {"kmeans", "euclidean", "gaussian"};
    for (int d = 0; d < 3; ++d)
    {
        std::vector<int32_t> normalScores;
        std::size_t normalAlarms = 0;
        for (std::size_t i = 0; i < testFrames.size(); ++i)
        {
            if (testFrames[i].condition == Normal)
            {
                const int32_t v = testFrames[i].score[d];
                normalScores.push_back(v);
                normalAlarms += (v > threshold[d]) ? 1u : 0u;
            }
        }
        const double falseAlarmRate = static_cast<double>(normalAlarms) / normalScores.size();
        std::fprintf(summary, "%s,normal,%zu,,%.4f,\n", detectorName[d], normalScores.size(), falseAlarmRate);
        std::printf("%-9s  normal running: %zu frames, false alarms %.2f%%\n", detectorName[d], normalScores.size(),
                    100.0 * falseAlarmRate);
        for (int c = Bearing; c <= Looseness; ++c)
        {
            std::vector<int32_t> scores;
            std::size_t alarms = 0;
            double firstSustained = -1.0;
            int run = 0;
            for (std::size_t i = 0; i < testFrames.size(); ++i)
            {
                if (testFrames[i].condition != c)
                {
                    continue;
                }
                const int32_t v = testFrames[i].score[d];
                const bool alarm = v > threshold[d];
                scores.push_back(v);
                alarms += alarm ? 1u : 0u;
                // Sustained: 8 alarms in a row (about a third of a second).
                run = alarm ? run + 1 : 0;
                if (run == 8 && firstSustained < 0.0)
                {
                    firstSustained = testFrames[i].severity;
                }
            }
            const double a = auc(normalScores, scores);
            std::fprintf(summary, "%s,%s,%zu,%.4f,%.4f,%.4f\n", detectorName[d], kConditionName[c], scores.size(), a,
                         static_cast<double>(alarms) / scores.size(), firstSustained);
            std::printf("%-9s  %-9s: AUC %.3f, alarms on %.1f%% of frames, first sustained alarm at severity %.2f\n",
                        detectorName[d], kConditionName[c], a, 100.0 * alarms / scores.size(), firstSustained);
        }
    }
    std::fclose(summary);
    std::printf("on-device state: k-means trainer %zu bytes, log-mel workspace %zu bytes\n", sizeof(trainer),
                sizeof(workspace));
    return 0;
}
