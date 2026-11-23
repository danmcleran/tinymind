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

// Tests for cpp/qmel.hpp: the fixed-point log-mel spectrogram and MFCCs,
// against a double-precision reference of the same definition. Included by
// quantization_unit_test.cpp inside its test suite, after <algorithm>,
// <cmath>, <random> and <vector>.

namespace qmel_test
{
    const double kPi = 3.14159265358979323846;

    // A log-mel front end with host-built tables, for frame length N.
    template<std::size_t N, std::size_t B>
    struct Frontend
    {
        static const std::size_t Bins = N / 2 + 1;
        int16_t window[N];
        int16_t twiddleCos[N / 2];
        int16_t twiddleSin[N / 2];
        uint16_t bandStart[B];
        uint16_t bandLength[B];
        uint16_t weights[2 * (N / 2 + 1)];
        std::size_t weightCount;
        double sampleRate;
        double fMin;
        double fMax;
        tinymind::QLogMelSpectrogram<N, B> mel;

        Frontend(double sr, double fmin, double fmax, int16_t preemphasis)
            : weightCount(0), sampleRate(sr), fMin(fmin), fMax(fmax), mel()
        {
            tinymind::buildQHannWindow(N, window);
            buildQFFTTwiddles(N, twiddleCos, twiddleSin);
            weightCount = tinymind::buildQMelFilterbank(N, sr, B, fmin, fmax, bandStart, bandLength,
                                                        weights, 2 * (N / 2 + 1));
            mel.window = window;
            mel.twiddle_cos = twiddleCos;
            mel.twiddle_sin = twiddleSin;
            mel.band_start = bandStart;
            mel.band_length = bandLength;
            mel.weights = weights;
            mel.preemphasis = preemphasis;
            mel.log_floor = -32768;
        }

        // ln of the mel-weighted power spectrum of the pre-emphasized,
        // Hann-windowed frame, samples read as int16 / 32768.
        void reference(const int16_t* x, int16_t previous, double* out) const
        {
            const double a = mel.preemphasis / 32768.0;
            std::vector<double> y(N);
            for (std::size_t n = 0; n < N; ++n)
            {
                const double prev = (n == 0) ? previous / 32768.0 : x[n - 1] / 32768.0;
                y[n] = (x[n] / 32768.0 - a * prev) * (0.5 - 0.5 * std::cos(2.0 * kPi * n / N));
            }
            std::vector<double> power(Bins);
            for (std::size_t k = 0; k < Bins; ++k)
            {
                double re = 0.0;
                double im = 0.0;
                for (std::size_t n = 0; n < N; ++n)
                {
                    re += y[n] * std::cos(2.0 * kPi * k * n / N);
                    im -= y[n] * std::sin(2.0 * kPi * k * n / N);
                }
                power[k] = re * re + im * im;
            }
            for (std::size_t m = 0; m < B; ++m)
            {
                double e = 0.0;
                for (std::size_t k = 0; k < Bins; ++k)
                {
                    e += tinymind::qmelFilterWeight(N, sampleRate, B, fMin, fMax, m, k) * power[k];
                }
                out[m] = std::log(e);
            }
        }
    };

    // A frame of kind 0 (tone), 1 (white noise), 2 (low-passed noise) or 3
    // (tone plus noise), peak-normalized and then scaled to levelDb dBFS.
    template<std::size_t N>
    void makeFrame(std::mt19937& rng, int kind, double levelDb, double sampleRate, int16_t* x)
    {
        std::uniform_real_distribution<double> uniform(0.0, 1.0);
        std::normal_distribution<double> gauss(0.0, 1.0);
        const double f0 = 50.0 + uniform(rng) * (sampleRate / 2.0 - 100.0);
        std::vector<double> v(N);
        double lowpass = 0.0;
        double peak = 0.0;
        for (std::size_t n = 0; n < N; ++n)
        {
            const double t = n / sampleRate;
            double s;
            if (kind == 0)
            {
                s = std::sin(2.0 * kPi * f0 * t);
            }
            else if (kind == 1)
            {
                s = gauss(rng);
            }
            else if (kind == 2)
            {
                lowpass = 0.95 * lowpass + 0.05 * gauss(rng);
                s = lowpass;
            }
            else
            {
                s = std::sin(2.0 * kPi * f0 * t) + 0.1 * gauss(rng);
            }
            v[n] = s;
            peak = std::max(peak, std::fabs(s));
        }
        const double amplitude = std::pow(10.0, levelDb / 20.0) * 32767.0 / peak;
        for (std::size_t n = 0; n < N; ++n)
        {
            x[n] = static_cast<int16_t>(std::lround(v[n] * amplitude));
        }
    }

    struct ErrorStats
    {
        std::vector<double> buckets[5]; // 0-10, 10-20, 20-30, 30-40, 40+ dB below the loudest band
    };

    double percentile(std::vector<double> v, double p)
    {
        std::sort(v.begin(), v.end());
        return v.empty() ? 0.0 : v[static_cast<std::size_t>(p * (v.size() - 1))];
    }

    template<std::size_t N, std::size_t B>
    ErrorStats measure(const Frontend<N, B>& f, int trials, unsigned seed)
    {
        std::mt19937 rng(seed);
        std::uniform_real_distribution<double> uniform(0.0, 1.0);
        tinymind::QMelWorkspace<N> ws;
        ErrorStats stats;
        int16_t x[N];
        int16_t out[B];
        double ref[B];
        for (int trial = 0; trial < trials; ++trial)
        {
            makeFrame<N>(rng, trial % 4, -70.0 * uniform(rng), f.sampleRate, x);
            const int16_t previous = static_cast<int16_t>(std::lround(2000.0 * (uniform(rng) - 0.5)));
            f.mel.forward(x, previous, out, ws);
            f.reference(x, previous, ref);
            const double peak = *std::max_element(ref, ref + B);
            for (std::size_t m = 0; m < B; ++m)
            {
                const double below = (peak - ref[m]) * 10.0 / std::log(10.0);
                const std::size_t bucket = std::min<std::size_t>(4, static_cast<std::size_t>(below / 10.0));
                stats.buckets[bucket].push_back(std::fabs(out[m] / 256.0 - ref[m]));
            }
        }
        return stats;
    }
}

BOOST_AUTO_TEST_CASE(qmel_log2_matches_std_log2)
{
    std::mt19937_64 rng(5);
    double worst = 0.0;
    for (int i = 0; i < 200000; ++i)
    {
        uint64_t x = rng() >> (rng() % 64);
        x = (x == 0u) ? 1u : x;
        const double got = static_cast<double>(tinymind::qmel_detail::log2Q30(x)) / 1073741824.0;
        worst = std::max(worst, std::fabs(got - std::log2(static_cast<double>(x))));
    }
    for (int b = 0; b < 64; ++b)
    {
        const uint64_t x = static_cast<uint64_t>(1) << b;
        BOOST_TEST(tinymind::qmel_detail::log2Q30(x) == static_cast<int64_t>(b) << 30);
    }
    BOOST_TEST(worst < 2e-4);
}

BOOST_AUTO_TEST_CASE(qmel_window_filterbank_and_dct_builders)
{
    int16_t w[16];
    tinymind::buildQHannWindow(16, w);
    BOOST_TEST(w[0] == 0);
    BOOST_TEST(w[8] == 32767); // 1.0 is stored as 32767
    BOOST_TEST(w[4] == 16384);
    tinymind::buildQHannWindow(16, w, false);
    BOOST_TEST(w[15] == 0);

    const std::size_t n = 512;
    const std::size_t bands = 40;
    const std::size_t bins = n / 2 + 1;
    uint16_t start[bands];
    uint16_t length[bands];
    uint16_t weights[2 * bins];
    const std::size_t used = tinymind::buildQMelFilterbank(n, 16000.0, bands, 20.0, 8000.0,
                                                           start, length, weights, 2 * bins);
    BOOST_TEST(used > 0u);
    BOOST_TEST(used <= 2 * bins);
    std::size_t offset = 0;
    for (std::size_t m = 0; m < bands; ++m)
    {
        BOOST_TEST(length[m] > 0u);
        for (std::size_t j = 0; j < length[m]; ++j)
        {
            const double f = tinymind::qmelFilterWeight(n, 16000.0, bands, 20.0, 8000.0, m, start[m] + j);
            BOOST_TEST(std::abs(static_cast<long>(weights[offset + j]) - std::lround(f * 65536.0)) <= 1);
        }
        if (m > 0)
        {
            BOOST_TEST(start[m] >= start[m - 1]); // bands climb in frequency
        }
        offset += length[m];
    }
    BOOST_TEST(offset == used);
    BOOST_TEST(tinymind::buildQMelFilterbank(n, 16000.0, bands, 20.0, 8000.0, start, length, weights, 10) == 0u);

    // Too many bands for a short FFT: the lowest bands fall between bins.
    uint16_t s2[40];
    uint16_t l2[40];
    uint16_t w2[2 * 33];
    tinymind::buildQMelFilterbank(64, 16000.0, 40, 0.0, 8000.0, s2, l2, w2, 2 * 33);
    BOOST_TEST(std::count(l2, l2 + 40, static_cast<uint16_t>(0)) > 0);

    int16_t dct[13 * 40];
    tinymind::buildQDctMatrix(40, 13, dct);
    for (std::size_t c = 0; c < 13; ++c)
    {
        for (std::size_t b = 0; b < 40; ++b)
        {
            const double v = std::sqrt((c == 0 ? 1.0 : 2.0) / 40.0) *
                             std::cos(qmel_test::kPi / 40.0 * (b + 0.5) * c);
            BOOST_TEST(std::abs(dct[c * 40 + b] - std::lround(v * 32768.0)) <= 0);
        }
    }
}

BOOST_AUTO_TEST_CASE(qmel_log_mel_matches_float_reference)
{
    // Error by how far a band sits below the frame's loudest band. Within
    // 30 dB the fixed-point bands track the float definition to a few
    // hundredths of a nat (a tenth of a dB); further down, the int16 FFT's
    // rounding floor takes over, which the 30-40 dB bound allows for.
    qmel_test::Frontend<512, 40> speech(16000.0, 20.0, 8000.0, 0);
    qmel_test::Frontend<256, 32> vibration(3200.0, 5.0, 1600.0, 31785);
    BOOST_TEST(speech.weightCount > 0u);
    BOOST_TEST(vibration.weightCount > 0u);
    const qmel_test::ErrorStats a = qmel_test::measure(speech, 160, 11);
    const qmel_test::ErrorStats b = qmel_test::measure(vibration, 160, 12);
    const qmel_test::ErrorStats* all[2] = {&a, &b};
    for (std::size_t i = 0; i < 2; ++i)
    {
        const qmel_test::ErrorStats& s = *all[i];
        BOOST_TEST(s.buckets[0].size() > 100u);
        BOOST_TEST(*std::max_element(s.buckets[0].begin(), s.buckets[0].end()) < 0.05);
        BOOST_TEST(qmel_test::percentile(s.buckets[1], 0.99) < 0.06);
        BOOST_TEST(qmel_test::percentile(s.buckets[2], 0.99) < 0.12);
        BOOST_TEST(qmel_test::percentile(s.buckets[3], 0.99) < 0.4);
    }
}

BOOST_AUTO_TEST_CASE(qmel_block_scaling_is_exact)
{
    // Halving a frame whose samples are all even halves y exactly, so the
    // block scale moves by one step and the FFT sees the same samples: every
    // band drops by exactly ln(4), which rounds to 354 or 355 in Q8. The same
    // holds down to a frame of +-2 LSB, where an unscaled int16 FFT would
    // have nothing left.
    qmel_test::Frontend<256, 24> f(8000.0, 0.0, 4000.0, 31785);
    tinymind::QMelWorkspace<256> ws;
    std::mt19937 rng(3);
    std::uniform_int_distribution<int> sample(-16383, 16383);
    int16_t x[256];
    int16_t loud[24];
    int16_t quiet[24];
    for (std::size_t n = 0; n < 256; ++n)
    {
        x[n] = static_cast<int16_t>(2 * sample(rng));
    }
    for (int halving = 0; halving < 13; ++halving)
    {
        f.mel.forward(x, 0, loud, ws);
        for (std::size_t n = 0; n < 256; ++n)
        {
            x[n] = static_cast<int16_t>(x[n] / 2);
        }
        if (halving == 12)
        {
            break;
        }
        // The halved frame must stay all-even for the next step to be exact.
        bool even = true;
        for (std::size_t n = 0; n < 256; ++n)
        {
            even = even && (x[n] % 2 == 0);
        }
        f.mel.forward(x, 0, quiet, ws);
        for (std::size_t m = 0; m < 24; ++m)
        {
            const int drop = loud[m] - quiet[m];
            BOOST_TEST((drop == 354 || drop == 355), "halving " << halving << " band " << m << " drop " << drop);
        }
        if (!even)
        {
            for (std::size_t n = 0; n < 256; ++n)
            {
                x[n] = static_cast<int16_t>(x[n] & ~1);
            }
        }
    }
}

BOOST_AUTO_TEST_CASE(qmel_silence_floor_and_previous_sample)
{
    qmel_test::Frontend<128, 16> f(8000.0, 0.0, 4000.0, 31785);
    f.mel.log_floor = -2560; // ln(E) = -10
    tinymind::QMelWorkspace<128> ws;
    int16_t x[128] = {0};
    int16_t out[16];

    // A silent frame reads the floor in every band.
    f.mel.forward(x, 0, out, ws);
    for (std::size_t m = 0; m < 16; ++m)
    {
        BOOST_TEST(out[m] == -2560);
    }

    // A zero frame after a nonzero sample is not silent once pre-emphasized:
    // y[0] = -a * previous.
    f.mel.forward(x, 1000, out, ws);
    double ref[16];
    f.reference(x, 1000, ref);
    for (std::size_t m = 0; m < 16; ++m)
    {
        const double expected = std::max(ref[m], -10.0);
        BOOST_TEST(std::fabs(out[m] / 256.0 - expected) < 0.05, "band " << m);
    }

    // A quiet band is clamped to the floor rather than reading below it.
    for (std::size_t n = 0; n < 128; ++n)
    {
        x[n] = static_cast<int16_t>(std::lround(3.0 * std::sin(2.0 * qmel_test::kPi * 500.0 * n / 8000.0)));
    }
    f.mel.forward(x, 0, out, ws);
    for (std::size_t m = 0; m < 16; ++m)
    {
        BOOST_TEST(out[m] >= -2560);
    }
    BOOST_TEST(std::count(out, out + 16, static_cast<int16_t>(-2560)) > 0);
}

BOOST_AUTO_TEST_CASE(qmel_mfcc_matches_float_dct)
{
    int16_t dct[13 * 40];
    tinymind::buildQDctMatrix(40, 13, dct);
    const tinymind::QMfcc<40, 13> mfcc = {dct};
    std::mt19937 rng(9);
    std::uniform_int_distribution<int> value(-32768, 32767);
    for (int trial = 0; trial < 200; ++trial)
    {
        int16_t logMel[40];
        for (std::size_t m = 0; m < 40; ++m)
        {
            logMel[m] = static_cast<int16_t>(value(rng));
        }
        int32_t out[13];
        mfcc.forward(logMel, out);
        for (std::size_t c = 0; c < 13; ++c)
        {
            double ref = 0.0;
            for (std::size_t m = 0; m < 40; ++m)
            {
                ref += std::sqrt((c == 0 ? 1.0 : 2.0) / 40.0) *
                       std::cos(qmel_test::kPi / 40.0 * (m + 0.5) * c) * logMel[m];
            }
            // Q1.15 table rounding: at most half an LSB per entry, 40 entries
            // of up to 32768 / 32768.
            BOOST_TEST(std::fabs(out[c] - ref) <= 21.0, "c " << c);
        }
    }

    // Flat log-mel: all energy in coefficient 0, sqrt(40) times the level.
    int16_t flat[40];
    std::fill(flat, flat + 40, static_cast<int16_t>(-1280));
    int32_t out[13];
    mfcc.forward(flat, out);
    BOOST_TEST(std::abs(out[0] - std::lround(-1280.0 * std::sqrt(40.0))) <= 2);
    for (std::size_t c = 1; c < 13; ++c)
    {
        BOOST_TEST(std::abs(out[c]) <= 2);
    }
}
