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

#pragma once

#include "include/tinymind_platform.hpp"

#include <cstddef>
#include <cstdint>

/*
 * Anomaly scores for int8 feature vectors: how far a sample sits from the
 * "normal" data a model was fitted to. The features are any int8 vector on a
 * fixed affine grid -- log-mel bands from qmel.hpp requantized to int8, an
 * int8 network's embedding, or quantized sensor readings.
 *
 *   QKMeansAnomaly        the distance to the nearest of K centroids, in
 *                         units of that cluster's mean squared radius
 *   QKMeansAnomalyTrainer learns those centroids and radii on the device, one
 *                         sample at a time, with no float and no buffered data
 *   QDiagGaussianAnomaly  the negative log-likelihood under a mixture of K
 *                         diagonal Gaussians (hard assignment: the best
 *                         component), fitted on the host (for example by
 *                         scikit-learn's GaussianMixture with covariance_type
 *                         "diag") and built by buildQDiagGaussianAnomaly
 *
 * A higher score is more anomalous. Each model scores a sample with integer
 * arithmetic only; a deployment compares the score with a threshold chosen on
 * normal data (for example the largest score seen on held-out normal samples
 * times a margin).
 *
 * Pure integer at runtime; freestanding-safe. Host builders need
 * TINYMIND_ENABLE_FLOAT and TINYMIND_ENABLE_STD.
 */

namespace tinymind {

    namespace qanomaly_detail {

        inline int32_t saturateToInt32(uint64_t v)
        {
            return (v > static_cast<uint64_t>(INT32_MAX)) ? INT32_MAX : static_cast<int32_t>(v);
        }

        // sum_f w_f (a_f - b_f)^2, or the unweighted sum when w is null. One
        // function for both, so every instantiation a model uses is called.
        template<std::size_t NumFeatures>
        inline uint64_t weightedSquaredDistance(const int8_t* a, const int8_t* b, const uint16_t* w)
        {
            if (w == nullptr)
            {
                uint32_t plain = 0;
                for (std::size_t f = 0; f < NumFeatures; ++f)
                {
                    const int32_t d = static_cast<int32_t>(a[f]) - static_cast<int32_t>(b[f]);
                    plain += static_cast<uint32_t>(d * d);
                }
                return plain;
            }
            uint64_t sum = 0;
            for (std::size_t f = 0; f < NumFeatures; ++f)
            {
                const int32_t d = static_cast<int32_t>(a[f]) - static_cast<int32_t>(b[f]);
                sum += static_cast<uint64_t>(static_cast<uint32_t>(d * d)) * w[f];
            }
            return sum;
        }

        // (d2 * inv) >> 24 without overflowing 64 bits: exact while d2 fits 32
        // bits, and to 16 bits of d2 beyond that, where the score is huge.
        inline uint64_t normalizeDistance(uint64_t d2, uint32_t inv)
        {
            if (d2 <= static_cast<uint64_t>(UINT32_MAX))
            {
                return (d2 * inv) >> 24;
            }
            return ((d2 >> 16) * inv) >> 8;
        }

        // round(a / b) for b > 0, ties away from zero.
        inline int32_t roundedDivide(int32_t a, int32_t b)
        {
            return (a >= 0) ? (a + b / 2) / b : -((-a + b / 2) / b);
        }

    } // namespace qanomaly_detail

    /**
     * K-means anomaly score over NumFeatures int8 features.
     *
     *   score(x) = min over clusters k of  |x - c_k|_w^2 / r_k^2,  in Q8
     *
     * |v|_w^2 = sum_f w_f v_f^2 is a squared distance with a weight per feature
     * (all weights 1 when feature_weight is null). The trainer sets w_f inversely
     * to how much feature f varies within the clusters of normal data, which makes
     * the distance a diagonal Mahalanobis distance: a feature that is normally
     * steady counts for more when it moves than one that normally swings.
     * where r_k^2 is cluster k's mean squared radius (the mean of |x - c_k|^2
     * over the normal samples assigned to it), so a typical member of a
     * cluster scores about 256 and a sample twice as far from every centroid as
     * that cluster's typical member scores about 1024. Normalizing by the
     * radius lets a tight cluster and a loose one share one threshold.
     *
     * centroids      : NumClusters x NumFeatures int8, on the features' grid
     * inv_radius_sq  : NumClusters values floor(2^32 / r_k^2), with r_k^2 in
     *                  the units of the weighted distance (LSB^2 times the
     *                  weights); 0 marks a cluster that is not in use
     * feature_weight : NumFeatures weights (any fixed point; the trainer uses
     *                  Q8), or null for an unweighted distance
     *
     * score() returns INT32_MAX if no cluster is in use. RAM: none; the tables
     * can live in flash, or in a QKMeansAnomalyTrainer that learned them.
     */
    template<std::size_t NumFeatures, std::size_t NumClusters>
    struct QKMeansAnomaly
    {
        static_assert(NumFeatures > 0 && NumClusters > 0, "At least one feature and one cluster.");
        // A weighted distance is below 2^32 * NumFeatures (65025 * 65535 per
        // feature); normalizeDistance keeps it to 2^16 * NumFeatures, whose
        // product with inv_radius_sq (< 2^32) must fit uint64.
        static_assert(NumFeatures <= 65536u, "Too many features for a 64-bit weighted distance.");

        const int8_t* centroids;
        const uint32_t* inv_radius_sq;
        const uint16_t* feature_weight;

        int32_t score(const int8_t* x) const
        {
            uint64_t best = UINT64_MAX;
            for (std::size_t k = 0; k < NumClusters; ++k)
            {
                const uint32_t inv = inv_radius_sq[k];
                if (inv == 0u)
                {
                    continue;
                }
                const uint64_t d2 = qanomaly_detail::weightedSquaredDistance<NumFeatures>(
                    x, centroids + k * NumFeatures, feature_weight);
                const uint64_t normalized = qanomaly_detail::normalizeDistance(d2, inv);
                best = (normalized < best) ? normalized : best;
            }
            return qanomaly_detail::saturateToInt32(best);
        }
    };

    /**
     * Learns a QKMeansAnomaly from normal samples on the device, in two
     * phases, one sample at a time:
     *
     *   learn(x)      online (MacQueen) k-means. The first NumClusters samples
     *                 seed the centroids; after that each sample moves its
     *                 nearest centroid by (x - c) / n, n the number of samples
     *                 that centroid has absorbed. Once n reaches the cap set by
     *                 setMaximumCount(), the step stays 1 / cap: the centroid
     *                 becomes a moving average that follows slow drift. The
     *                 centroids are kept in Q8 (int16), so steps far below one
     *                 LSB still accumulate.
     *                 learn() also measures how much each feature varies around
     *                 its centroid (from samples whose centroid has absorbed at
     *                 least 8), which sets the feature weights.
     *   calibrate(x)  the first call rounds the centroids to int8, sets the
     *                 feature weights and freezes both; every call adds the
     *                 weighted |x - c|^2 to the nearest cluster's radius estimate.
     *                 Calibrate on normal samples not used in learn() for an
     *                 honest radius, or on the same ones.
     *
     * Feature weights: w_f = 256 * mean_g(v_g) / v_f in Q8, clamped to 1..65535,
     * where v_f is feature f's mean squared deviation from its centroid during
     * learning. setFeatureWeighting(false) before calibrating gives the plain
     * Euclidean distance instead.
     *
     * model() then gives the QKMeansAnomaly (it points into this object). A
     * cluster that no calibration sample reached is left out of the score.
     *
     * Passing the learning data several times through learn() (on a host, or
     * from a buffer) brings online k-means closer to batch k-means; one pass is
     * enough for a stationary signal.
     *
     * RAM: NumClusters x NumFeatures x 3 bytes, plus 18 bytes per cluster and
 * 10 bytes per feature.
     */
    template<std::size_t NumFeatures, std::size_t NumClusters>
    class QKMeansAnomalyTrainer
    {
    public:
        typedef QKMeansAnomaly<NumFeatures, NumClusters> ModelType;

        QKMeansAnomalyTrainer() : mMaximumCount(UINT16_MAX), mWeighting(true)
        {
            reset();
        }

        void reset()
        {
            mSeeded = 0;
            mFrozen = false;
            mResidualCount = 0;
            for (std::size_t f = 0; f < NumFeatures; ++f)
            {
                mResidualSum[f] = 0;
                mWeight[f] = 256;
            }
            for (std::size_t k = 0; k < NumClusters; ++k)
            {
                mCount[k] = 0;
                mCalibrationCount[k] = 0;
                mSumSquaredDistance[k] = 0;
                mInverseRadiusSquared[k] = 0;
                for (std::size_t f = 0; f < NumFeatures; ++f)
                {
                    mCentroidQ8[k][f] = 0;
                    mCentroid[k][f] = 0;
                }
            }
        }

        void setFeatureWeighting(bool enabled)
        {
            mWeighting = enabled;
        }

        void setMaximumCount(uint16_t count)
        {
            mMaximumCount = (count == 0) ? 1 : count;
        }

        void learn(const int8_t* x)
        {
            if (mFrozen)
            {
                return;
            }
            if (mSeeded < NumClusters)
            {
                for (std::size_t f = 0; f < NumFeatures; ++f)
                {
                    mCentroidQ8[mSeeded][f] = static_cast<int16_t>(static_cast<int32_t>(x[f]) * 256);
                }
                mCount[mSeeded] = 1;
                ++mSeeded;
                return;
            }

            std::size_t nearest = 0;
            uint64_t best = UINT64_MAX;
            for (std::size_t k = 0; k < mSeeded; ++k)
            {
                uint64_t d2 = 0;
                for (std::size_t f = 0; f < NumFeatures; ++f)
                {
                    const int64_t d = static_cast<int64_t>(x[f]) * 256 - mCentroidQ8[k][f];
                    d2 += static_cast<uint64_t>(d * d);
                }
                if (d2 < best)
                {
                    best = d2;
                    nearest = k;
                }
            }

            const bool settled = mCount[nearest] >= 8;
            if (mCount[nearest] < mMaximumCount)
            {
                ++mCount[nearest];
            }
            const int32_t n = static_cast<int32_t>(mCount[nearest]);
            for (std::size_t f = 0; f < NumFeatures; ++f)
            {
                const int32_t c = mCentroidQ8[nearest][f];
                const int32_t residual = static_cast<int32_t>(x[f]) * 256 - c;
                if (settled)
                {
                    // Q16 LSB^2.
                    mResidualSum[f] += static_cast<uint64_t>(static_cast<int64_t>(residual) * residual);
                }
                const int32_t step = qanomaly_detail::roundedDivide(residual, n);
                mCentroidQ8[nearest][f] = static_cast<int16_t>(c + step);
            }
            mResidualCount += settled ? 1u : 0u;
        }

        void calibrate(const int8_t* x)
        {
            if (!mFrozen)
            {
                freeze();
            }
            const uint16_t* const weights = mWeighting ? mWeight : nullptr;
            std::size_t nearest = NumClusters;
            uint64_t best = UINT64_MAX;
            for (std::size_t k = 0; k < mSeeded; ++k)
            {
                const uint64_t d2 = qanomaly_detail::weightedSquaredDistance<NumFeatures>(x, mCentroid[k], weights);
                if (d2 < best || nearest == NumClusters)
                {
                    best = d2;
                    nearest = k;
                }
            }
            if (nearest == NumClusters)
            {
                return;
            }
            mSumSquaredDistance[nearest] += best;
            ++mCalibrationCount[nearest];
            // r^2 = ceil(mean |x - c|^2), at least 1 LSB^2.
            const uint64_t n = mCalibrationCount[nearest];
            uint64_t radiusSquared = (mSumSquaredDistance[nearest] + n - 1) / n;
            radiusSquared = (radiusSquared == 0u) ? 1u : radiusSquared;
            const uint64_t inverse = (static_cast<uint64_t>(1) << 32) / radiusSquared;
            mInverseRadiusSquared[nearest] = (inverse > UINT32_MAX) ? UINT32_MAX : static_cast<uint32_t>(inverse);
        }

        ModelType model() const
        {
            const ModelType m = {&mCentroid[0][0], mInverseRadiusSquared, mWeighting ? mWeight : nullptr};
            return m;
        }

        std::size_t seededClusters() const
        {
            return mSeeded;
        }

        uint16_t learnCount(std::size_t cluster) const
        {
            return mCount[cluster];
        }

        uint32_t calibrationCount(std::size_t cluster) const
        {
            return mCalibrationCount[cluster];
        }

        const int8_t* centroid(std::size_t cluster) const
        {
            return mCentroid[cluster];
        }

        uint16_t featureWeight(std::size_t feature) const
        {
            return mWeight[feature];
        }

    private:
        void freeze()
        {
            for (std::size_t k = 0; k < NumClusters; ++k)
            {
                for (std::size_t f = 0; f < NumFeatures; ++f)
                {
                    int32_t v = qanomaly_detail::roundedDivide(mCentroidQ8[k][f], 256);
                    v = (v < -128) ? -128 : ((v > 127) ? 127 : v);
                    mCentroid[k][f] = static_cast<int8_t>(v);
                }
            }
            if (mResidualCount > 0u)
            {
                uint64_t total = 0;
                for (std::size_t f = 0; f < NumFeatures; ++f)
                {
                    total += mResidualSum[f];
                }
                // w_f = 256 * (total / F) / sum_f, the residual count
                // cancelling.
                const uint64_t meanVariance = total / NumFeatures;
                for (std::size_t f = 0; f < NumFeatures; ++f)
                {
                    // Scale numerator and denominator down together until
                    // the numerator times 256 fits 64 bits.
                    uint64_t numerator = meanVariance;
                    uint64_t denominator = mResidualSum[f];
                    while (numerator > (UINT64_MAX >> 9))
                    {
                        numerator >>= 1;
                        denominator >>= 1;
                    }
                    uint64_t w = (denominator == 0u)
                        ? 65535u
                        : (numerator * 256u + denominator / 2u) / denominator;
                    w = (w < 1u) ? 1u : ((w > 65535u) ? 65535u : w);
                    mWeight[f] = static_cast<uint16_t>(w);
                }
            }
            mFrozen = true;
        }

        uint64_t mResidualSum[NumFeatures];
        uint32_t mResidualCount;
        uint16_t mWeight[NumFeatures];
        int16_t mCentroidQ8[NumClusters][NumFeatures];
        int8_t mCentroid[NumClusters][NumFeatures];
        uint64_t mSumSquaredDistance[NumClusters];
        uint32_t mCalibrationCount[NumClusters];
        uint32_t mInverseRadiusSquared[NumClusters];
        uint16_t mCount[NumClusters];
        uint16_t mMaximumCount;
        std::size_t mSeeded;
        bool mFrozen;
        bool mWeighting;
    };

    /**
     * Mixture of NumComponents diagonal Gaussians over NumFeatures int8
     * features, scored by hard assignment:
     *
     *   score(x) = min over k of  sum_f (x_f - mu_kf)^2 / (2 sigma_kf^2) + b_k
     *   b_k      = -ln(pi_k) + sum_f ln(sqrt(2 pi) sigma_kf)
     *
     * which is the negative log-likelihood of x under its most likely
     * component (an upper bound on the mixture's negative log-likelihood),
     * in nats, Q8 (1/256 nat) as int32, saturating at INT32_MAX. The
     * variances are of the real-valued features, so b_k is the real density's
     * normalizer; the quadratic term is the same in any units.
     *
     * means     : NumComponents x NumFeatures int8, on the features' grid
     * precision : NumComponents x NumFeatures, 2^24 / (2 sigma^2) with sigma in
     *             LSBs (Q24 of 1 / (2 sigma^2); a variance of 1/512 LSB^2 is
     *             the smallest it can hold)
     * bias      : NumComponents values b_k in Q8
     *
     * buildQDiagGaussianAnomaly makes all three from a float fit.
     */
    template<std::size_t NumFeatures, std::size_t NumComponents>
    struct QDiagGaussianAnomaly
    {
        static_assert(NumFeatures > 0 && NumComponents > 0, "At least one feature and one component.");
        // Each term is below 2^16 * 2^32; their sum must fit uint64.
        static_assert(NumFeatures <= 65536u, "Too many features for a 64-bit sum.");

        const int8_t* means;
        const uint32_t* precision;
        const int32_t* bias;

        int32_t score(const int8_t* x) const
        {
            int64_t best = INT64_MAX;
            for (std::size_t k = 0; k < NumComponents; ++k)
            {
                const int8_t* mu = means + k * NumFeatures;
                const uint32_t* p = precision + k * NumFeatures;
                uint64_t quadratic = 0; // Q24
                for (std::size_t f = 0; f < NumFeatures; ++f)
                {
                    const int32_t d = static_cast<int32_t>(x[f]) - static_cast<int32_t>(mu[f]);
                    quadratic += static_cast<uint64_t>(static_cast<uint32_t>(d * d)) * p[f];
                }
                // To Q8, rounded; at most 2^48 after the shift, so the sum
                // with the bias cannot overflow int64.
                const int64_t value = static_cast<int64_t>((quadratic + 32768u) >> 16) + bias[k];
                best = (value < best) ? value : best;
            }
            if (best > static_cast<int64_t>(INT32_MAX))
            {
                return INT32_MAX;
            }
            return (best < static_cast<int64_t>(INT32_MIN)) ? INT32_MIN : static_cast<int32_t>(best);
        }
    };

} // namespace tinymind

#if TINYMIND_ENABLE_FLOAT && TINYMIND_ENABLE_STD
#include <cmath>

namespace tinymind {

    /**
     * Build QDiagGaussianAnomaly tables from a float fit.
     *
     * means, variances : num_components x num_features, in the features' real
     *                    units (scikit-learn: means_ and covariances_ with
     *                    covariance_type "diag")
     * weights          : num_components mixture weights pi_k (weights_)
     * input_scale / input_zero_point : the features' int8 grid,
     *                    real = scale * (q - zero_point)
     * min_variance_lsb2: floor on every variance in LSB^2, so a feature that
     *                    barely moved in the fit does not dominate the score
     *                    (0.25, half an LSB, is a reasonable floor)
     *
     * Means are rounded onto the grid (so a mean is off by up to half an LSB);
     * the bias uses the floored variances.
     */
    inline void buildQDiagGaussianAnomaly(std::size_t num_components, std::size_t num_features,
                                          const double* means, const double* variances, const double* weights,
                                          float input_scale, int32_t input_zero_point, double min_variance_lsb2,
                                          int8_t* means_out, uint32_t* precision_out, int32_t* bias_out)
    {
        const double pi = 3.14159265358979323846;
        const double scale = static_cast<double>(input_scale);
        for (std::size_t k = 0; k < num_components; ++k)
        {
            double b = -std::log(weights[k]);
            for (std::size_t f = 0; f < num_features; ++f)
            {
                const std::size_t i = k * num_features + f;
                long q = std::lround(means[i] / scale) + input_zero_point;
                q = (q < -128) ? -128 : ((q > 127) ? 127 : q);
                means_out[i] = static_cast<int8_t>(q);

                double varianceLsb = variances[i] / (scale * scale);
                varianceLsb = (varianceLsb < min_variance_lsb2) ? min_variance_lsb2 : varianceLsb;
                double p = std::round(16777216.0 / (2.0 * varianceLsb));
                p = (p > 4294967295.0) ? 4294967295.0 : ((p < 1.0) ? 1.0 : p);
                precision_out[i] = static_cast<uint32_t>(p);

                b += 0.5 * std::log(2.0 * pi * varianceLsb * scale * scale);
            }
            double bq = std::round(b * 256.0);
            bq = (bq > 2147483647.0) ? 2147483647.0 : ((bq < -2147483648.0) ? -2147483648.0 : bq);
            bias_out[k] = static_cast<int32_t>(bq);
        }
    }

} // namespace tinymind
#endif // TINYMIND_ENABLE_FLOAT && TINYMIND_ENABLE_STD
