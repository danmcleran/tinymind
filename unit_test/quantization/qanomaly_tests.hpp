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

// Tests for cpp/qanomaly.hpp. Included by quantization_unit_test.cpp inside its
// test suite, after <algorithm>, <cmath>, <random> and <vector>.

namespace qanomaly_test
{
    const std::size_t F = 6;
    const std::size_t K = 3;

    // Three well-separated clusters with different spreads.
    const double kCentre[K][F] = {
        {-60.0, -40.0, 10.0, 0.0, 30.0, -20.0},
        {40.0, 50.0, -30.0, 20.0, -10.0, 60.0},
        {0.0, 0.0, 70.0, -70.0, 0.0, 0.0}};
    const double kSpread[K] = {3.0, 6.0, 10.0};

    void sample(std::mt19937& rng, std::size_t cluster, double shift, int8_t* x)
    {
        std::normal_distribution<double> gauss(0.0, 1.0);
        for (std::size_t f = 0; f < F; ++f)
        {
            const double v = kCentre[cluster][f] + shift + kSpread[cluster] * gauss(rng);
            x[f] = static_cast<int8_t>(std::max(-128.0, std::min(127.0, std::round(v))));
        }
    }

    // Probability that a random anomalous sample outscores a random normal one.
    double auc(const std::vector<int32_t>& normal, const std::vector<int32_t>& anomalous)
    {
        double wins = 0.0;
        for (std::size_t i = 0; i < anomalous.size(); ++i)
        {
            for (std::size_t j = 0; j < normal.size(); ++j)
            {
                wins += (anomalous[i] > normal[j]) ? 1.0 : ((anomalous[i] == normal[j]) ? 0.5 : 0.0);
            }
        }
        return wins / (static_cast<double>(anomalous.size()) * normal.size());
    }
}

BOOST_AUTO_TEST_CASE(qanomaly_kmeans_score_is_normalized_nearest_distance)
{
    std::mt19937 rng(1);
    std::uniform_int_distribution<int> value(-128, 127);
    std::uniform_int_distribution<uint32_t> radius(1, 5000);
    int8_t centroids[4 * 5];
    uint32_t inv[4];
    for (std::size_t i = 0; i < 20; ++i)
    {
        centroids[i] = static_cast<int8_t>(value(rng));
    }
    for (int trial = 0; trial < 500; ++trial)
    {
        uint32_t r2[4];
        for (std::size_t k = 0; k < 4; ++k)
        {
            r2[k] = radius(rng);
            inv[k] = static_cast<uint32_t>((static_cast<uint64_t>(1) << 32) / r2[k]);
        }
        inv[trial % 4] = (trial % 7 == 0) ? 0u : inv[trial % 4]; // sometimes a cluster is unused
        const tinymind::QKMeansAnomaly<5, 4> model = {centroids, inv, nullptr};
        int8_t x[5];
        for (std::size_t f = 0; f < 5; ++f)
        {
            x[f] = static_cast<int8_t>(value(rng));
        }
        uint64_t want = UINT64_MAX;
        for (std::size_t k = 0; k < 4; ++k)
        {
            if (inv[k] == 0u)
            {
                continue;
            }
            uint64_t d2 = 0;
            for (std::size_t f = 0; f < 5; ++f)
            {
                const int64_t d = static_cast<int64_t>(x[f]) - centroids[k * 5 + f];
                d2 += static_cast<uint64_t>(d * d);
            }
            want = std::min(want, (d2 * inv[k]) >> 24);
            // Within an LSB of the real ratio in Q8.
            BOOST_TEST(std::fabs(static_cast<double>((d2 * inv[k]) >> 24) - 256.0 * d2 / r2[k]) <= 1.0 + 256.0 * d2 / r2[k] * 1e-6);
        }
        BOOST_TEST(model.score(x) == static_cast<int32_t>(std::min<uint64_t>(want, INT32_MAX)));
    }

    const uint32_t none[4] = {0, 0, 0, 0};
    const tinymind::QKMeansAnomaly<5, 4> empty = {centroids, none, nullptr};
    const int8_t zero[5] = {0, 0, 0, 0, 0};
    BOOST_TEST(empty.score(zero) == INT32_MAX);
}

BOOST_AUTO_TEST_CASE(qanomaly_kmeans_trainer_learns_clusters_and_radii)
{
    using namespace qanomaly_test;
    tinymind::QKMeansAnomalyTrainer<F, K> trainer;
    std::mt19937 rng(2);
    int8_t x[F];
    // Seed one centroid from each cluster, then stream mixed samples.
    for (std::size_t k = 0; k < K; ++k)
    {
        sample(rng, k, 0.0, x);
        trainer.learn(x);
    }
    BOOST_TEST(trainer.seededClusters() == K);
    for (int i = 0; i < 3000; ++i)
    {
        sample(rng, static_cast<std::size_t>(i % 3), 0.0, x);
        trainer.learn(x);
    }
    for (int i = 0; i < 1500; ++i)
    {
        sample(rng, static_cast<std::size_t>(i % 3), 0.0, x);
        trainer.calibrate(x);
    }

    for (std::size_t k = 0; k < K; ++k)
    {
        // Centroid k was seeded from cluster k; within 1.5 LSB of its centre.
        for (std::size_t f = 0; f < F; ++f)
        {
            BOOST_TEST(std::fabs(trainer.centroid(k)[f] - kCentre[k][f]) <= 1.5, "cluster " << k << " feature " << f);
        }
        BOOST_TEST(trainer.calibrationCount(k) == 500u);
    }

    // Normal samples score about 256 (one mean squared radius) whichever
    // cluster they belong to; the normalization is what makes that true for
    // clusters whose spreads differ by more than 3x.
    const tinymind::QKMeansAnomaly<F, K> model = trainer.model();
    for (std::size_t k = 0; k < K; ++k)
    {
        double mean = 0.0;
        for (int i = 0; i < 400; ++i)
        {
            sample(rng, k, 0.0, x);
            mean += model.score(x);
        }
        mean /= 400.0;
        BOOST_TEST(mean > 200.0, "cluster " << k << " mean score " << mean);
        BOOST_TEST(mean < 320.0, "cluster " << k << " mean score " << mean);
    }

    // Samples shifted off every cluster are separated from normal ones.
    std::vector<int32_t> normal;
    std::vector<int32_t> shifted;
    for (int i = 0; i < 300; ++i)
    {
        sample(rng, static_cast<std::size_t>(i % 3), 0.0, x);
        normal.push_back(model.score(x));
        sample(rng, static_cast<std::size_t>(i % 3), 2.5 * kSpread[i % 3], x);
        shifted.push_back(model.score(x));
    }
    BOOST_TEST(auc(normal, shifted) > 0.97);
}

BOOST_AUTO_TEST_CASE(qanomaly_kmeans_trainer_edge_cases)
{
    using namespace qanomaly_test;
    // Fewer samples than clusters: only the seeded clusters score.
    tinymind::QKMeansAnomalyTrainer<F, K> sparse;
    const int8_t a[F] = {10, 10, 10, 10, 10, 10};
    const int8_t b[F] = {12, 10, 10, 10, 10, 10};
    sparse.learn(a);
    BOOST_TEST(sparse.seededClusters() == 1u);
    sparse.calibrate(a);
    sparse.calibrate(b);
    const tinymind::QKMeansAnomaly<F, K> model = sparse.model();
    // r^2 = ceil((0 + 4) / 2) = 2, so b scores 4 / 2 in Q8.
    BOOST_TEST(model.score(b) == 512);
    BOOST_TEST(model.score(a) == 0);

    // learn() after calibration has started is ignored: the model is frozen.
    sparse.learn(b);
    BOOST_TEST(sparse.seededClusters() == 1u);

    // Nothing learned: score is INT32_MAX, and calibrate() does nothing.
    tinymind::QKMeansAnomalyTrainer<F, K> none;
    none.calibrate(a);
    BOOST_TEST(none.model().score(a) == INT32_MAX);

    // A capped count turns the centroid into a moving average that follows a
    // step change; an uncapped one moves only by 1/n.
    tinymind::QKMeansAnomalyTrainer<1, 1> capped;
    tinymind::QKMeansAnomalyTrainer<1, 1> uncapped;
    capped.setMaximumCount(16);
    const int8_t low[1] = {0};
    const int8_t high[1] = {100};
    for (int i = 0; i < 1000; ++i)
    {
        capped.learn(low);
        uncapped.learn(low);
    }
    for (int i = 0; i < 200; ++i)
    {
        capped.learn(high);
        uncapped.learn(high);
    }
    capped.calibrate(high);
    uncapped.calibrate(high);
    BOOST_TEST(capped.centroid(0)[0] >= 99);
    BOOST_TEST(uncapped.centroid(0)[0] <= 20); // 100 * 200 / 1200 = 16.7
}

BOOST_AUTO_TEST_CASE(qanomaly_kmeans_feature_weighting)
{
    // One cluster; feature 0 swings widely in normal data, feature 1 barely
    // moves. A sample displaced by 6 LSB in the steady feature is an
    // anomaly; one displaced 6 LSB in the swinging feature is not. The plain
    // Euclidean distance cannot tell them apart; the weighted one can.
    std::mt19937 rng(8);
    std::normal_distribution<double> gauss(0.0, 1.0);
    tinymind::QKMeansAnomalyTrainer<2, 1> weighted;
    tinymind::QKMeansAnomalyTrainer<2, 1> plain;
    plain.setFeatureWeighting(false);
    for (int phase = 0; phase < 2; ++phase)
    {
        for (int i = 0; i < 3000; ++i)
        {
            const int8_t x[2] = {static_cast<int8_t>(std::lround(20.0 * gauss(rng))),
                                 static_cast<int8_t>(std::lround(1.0 * gauss(rng)))};
            if (phase == 0)
            {
                weighted.learn(x);
                plain.learn(x);
            }
            else
            {
                weighted.calibrate(x);
                plain.calibrate(x);
            }
        }
    }
    // Variances about 400 and 1.08 (a rounded unit normal), mean about 200:
    // weights near 256 * 200 / 400 = 128 and 256 * 200 / 1.08 = 47,000.
    BOOST_TEST(std::abs(weighted.featureWeight(0) - 128) <= 10);
    BOOST_TEST(std::abs(weighted.featureWeight(1) - 47500) <= 4000);
    const int8_t steadyMoved[2] = {0, 6};
    const int8_t swingingMoved[2] = {6, 0};
    const tinymind::QKMeansAnomaly<2, 1> w = weighted.model();
    const tinymind::QKMeansAnomaly<2, 1> p = plain.model();
    BOOST_TEST(w.score(steadyMoved) > 10 * w.score(swingingMoved));
    BOOST_TEST(w.score(steadyMoved) > 4 * 256);   // far outside the normal radius
    BOOST_TEST(w.score(swingingMoved) < 256);     // well inside it
    BOOST_TEST(p.score(steadyMoved) == p.score(swingingMoved));
    BOOST_TEST(p.score(steadyMoved) < 256);       // Euclidean: looks normal
}

BOOST_AUTO_TEST_CASE(qanomaly_diag_gaussian_matches_float_nll)
{
    using namespace qanomaly_test;
    // Fit a 3-component mixture exactly (known parameters) on a grid with
    // scale 0.05 and zero point 3, then compare scores with the float
    // negative log-likelihood of the best component.
    const float scale = 0.05f;
    const int32_t zp = 3;
    double means[K * F];
    double variances[K * F];
    const double weights[K] = {0.5, 0.3, 0.2};
    for (std::size_t k = 0; k < K; ++k)
    {
        for (std::size_t f = 0; f < F; ++f)
        {
            means[k * F + f] = (kCentre[k][f] - zp) * scale + 0.013 * f; // off-grid on purpose
            variances[k * F + f] = std::pow(kSpread[k] * scale * (1.0 + 0.1 * f), 2.0);
        }
    }
    int8_t qMeans[K * F];
    uint32_t precision[K * F];
    int32_t bias[K];
    tinymind::buildQDiagGaussianAnomaly(K, F, means, variances, weights, scale, zp, 0.25,
                                        qMeans, precision, bias);
    const tinymind::QDiagGaussianAnomaly<F, K> model = {qMeans, precision, bias};

    std::mt19937 rng(4);
    std::uniform_int_distribution<int> value(-128, 127);
    const double pi = 3.14159265358979323846;
    double worst = 0.0;
    for (int trial = 0; trial < 2000; ++trial)
    {
        int8_t x[F];
        if (trial % 2)
        {
            sample(rng, static_cast<std::size_t>(trial % 3), 0.0, x);
        }
        else
        {
            for (std::size_t f = 0; f < F; ++f)
            {
                x[f] = static_cast<int8_t>(value(rng));
            }
        }
        double best = 1e300;
        for (std::size_t k = 0; k < K; ++k)
        {
            double nll = -std::log(weights[k]);
            for (std::size_t f = 0; f < F; ++f)
            {
                // The means as the builder put them on the grid: that half-LSB
                // move is the documented approximation, not under test here.
                const double v = variances[k * F + f];
                const double d = scale * (x[f] - qMeans[k * F + f]);
                nll += d * d / (2.0 * v) + 0.5 * std::log(2.0 * pi * v);
            }
            best = std::min(best, nll);
        }
        const double got = model.score(x) / 256.0;
        // Left: the Q8 rounding of the score and bias (1/256 nat) and the
        // relative rounding of each precision, 0.5 / (2^24 / (2 sigma^2)),
        // below 2e-5 for these spreads.
        const double tolerance = 2.0 / 256.0 + 2e-5 * std::fabs(best);
        worst = std::max(worst, std::fabs(got - best) / tolerance);
        BOOST_TEST(std::fabs(got - best) <= tolerance, "trial " << trial << " got " << got << " want " << best);
    }
    BOOST_TEST(worst <= 1.0);

    // The variance floor caps the precision of a feature that never moved.
    const double flatMean[1] = {0.0};
    const double flatVariance[1] = {0.0};
    const double one[1] = {1.0};
    int8_t m[1];
    uint32_t p[1];
    int32_t b[1];
    tinymind::buildQDiagGaussianAnomaly(1, 1, flatMean, flatVariance, one, 0.1f, 0, 0.25, m, p, b);
    BOOST_TEST(p[0] == 33554432u); // 2^24 / (2 * 0.25)
}

BOOST_AUTO_TEST_CASE(qanomaly_diag_gaussian_integer_score_is_exact)
{
    // Against an exact integer reference of the documented formula, including
    // saturation at INT32_MAX.
    std::mt19937 rng(6);
    std::uniform_int_distribution<int> value(-128, 127);
    std::uniform_int_distribution<uint32_t> prec(1, 4000000000u);
    std::uniform_int_distribution<int32_t> biasValue(-100000, 100000);
    int8_t means[2 * 4];
    uint32_t precision[2 * 4];
    int32_t bias[2];
    for (int trial = 0; trial < 500; ++trial)
    {
        for (std::size_t i = 0; i < 8; ++i)
        {
            means[i] = static_cast<int8_t>(value(rng));
            precision[i] = (trial % 3 == 0) ? prec(rng) : prec(rng) % 200000u + 1u;
        }
        bias[0] = biasValue(rng);
        bias[1] = biasValue(rng);
        const tinymind::QDiagGaussianAnomaly<4, 2> model = {means, precision, bias};
        int8_t x[4];
        for (std::size_t f = 0; f < 4; ++f)
        {
            x[f] = static_cast<int8_t>(value(rng));
        }
        int64_t want = INT64_MAX;
        for (std::size_t k = 0; k < 2; ++k)
        {
            uint64_t q = 0;
            for (std::size_t f = 0; f < 4; ++f)
            {
                const int64_t d = static_cast<int64_t>(x[f]) - means[k * 4 + f];
                q += static_cast<uint64_t>(d * d) * precision[k * 4 + f];
            }
            want = std::min(want, static_cast<int64_t>((q + 32768) >> 16) + bias[k]);
        }
        BOOST_TEST(model.score(x) == static_cast<int32_t>(std::min<int64_t>(want, INT32_MAX)));
    }
}
