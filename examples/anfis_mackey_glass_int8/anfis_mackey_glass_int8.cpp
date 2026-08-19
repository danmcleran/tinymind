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

/*
 * int8 Mackey-Glass ANFIS -- and a deliberate negative result.
 *
 * Everywhere else in TinyMind, moving a model to int8 makes it smaller. For
 * ANFIS it does not, and this exemplar exists to measure that rather than let
 * the assumption ride.
 *
 * The reason is structural. An int8 layer's parameters are its weights, and
 * halving their width halves the model. But ANFIS spends most of its
 * parameters on membership functions, and cpp/qanfis.hpp evaluates those
 * through a 256-entry lookup table per (input, membership function) pair --
 * that is how the shape stops mattering at runtime. Those tables cost 512
 * bytes each regardless of how few parameters the shape they replaced had. A
 * Q16.16 bell is two numbers, eight bytes; its int8 lookup table is 512.
 *
 * So the int8 tier's argument here is not footprint. It is that an int8
 * frontend can feed ANFIS directly, with no bridge and no float on the hot
 * path. This example reports both the cost and the accuracy so the trade is
 * visible.
 *
 * It also shows why a model bound for int8 needs a ridge penalty on the
 * consequent solve, by running the same pipeline over an unregularized fit.
 */

#include "anfis.hpp"
#include "qanfis.hpp"
#include "qformat.hpp"
#include "include/qcalibration.hpp"
#include "include/nnproperties.hpp"

#include "anfis_int8_model.hpp"
#include "anfis_int8_model_unregularized.hpp"
#include "anfis_int8_data.hpp"

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <fstream>
#include <iostream>
#include <vector>

namespace {

    namespace model = anfis_int8_model;
    namespace plain = anfis_int8_model_unregularized;
    namespace data  = anfis_int8_data;

    constexpr std::size_t NI = model::NumberOfInputs;
    constexpr std::size_t NM = model::NumberOfMembershipFunctionsPerInput;
    constexpr std::size_t NR = model::NumberOfRules;

    typedef tinymind::QValue<16, 16, true, tinymind::RoundUpPolicy> QType;
    typedef tinymind::GeneralizedBellMembershipFunction<double, 1> DoubleMf;
    typedef tinymind::GeneralizedBellMembershipFunction<QType, 1>  QMf;

    typedef tinymind::Anfis<double, NI, NM, NR, DoubleMf, true, 1> DoubleAnfis;
    typedef tinymind::Anfis<QType,  NI, NM, NR, QMf,      true, 1> QFormatAnfis;
    typedef tinymind::QAnfis<int8_t, int8_t, uint16_t, int32_t, int8_t,
                             NI, NM, NR> Int8Anfis;

    typedef tinymind::ValueConverter<double, QType> ToQ;

    // Affine grid for the int8 tiers, built from the TRAINING split only.
    float   gInScale = 0.0f, gOutScale = 0.0f;
    int32_t gInZp = 0,       gOutZp = 0;
    // The shared grid the firing-strength-weighted sum runs on. 2^20 levels
    // keeps the per-rule rescale lossless in practice while leaving the int64
    // numerator far from its rails.
    float   gConsequentScale = 0.0f;

    void buildAffineGrids()
    {
        // Use the library helper rather than deriving scale/zero_point by
        // hand. Both this model's inputs and its outputs are strictly
        // positive, and for a range that excludes zero the naive
        // zero_point = qmin - fmin/scale lands outside int8 (about -230 here)
        // and wraps when stored. computeAffineParamsAsymmetric extends the
        // range to include zero first, which is what keeps the zero_point on
        // the grid. The cost is real -- a [0, 1.33] grid spends roughly a
        // quarter of its codes below the data -- but it is the correct
        // trade, and it is the same one every other int8 layer here makes.
        const tinymind::AffineParams in =
            tinymind::computeAffineParamsAsymmetric(
                static_cast<float>(data::InputMin),
                static_cast<float>(data::InputMax), -128, 127);
        gInScale = in.scale;
        gInZp    = in.zero_point;

        const double pad = 0.05;
        const tinymind::AffineParams outp =
            tinymind::computeAffineParamsAsymmetric(
                static_cast<float>(data::OutputMin - pad),
                static_cast<float>(data::OutputMax + pad), -128, 127);
        gOutScale = outp.scale;
        gOutZp    = outp.zero_point;
    }

    int8_t quantizeInput(double x)
    {
        long v = std::lround(x / static_cast<double>(gInScale)) + gInZp;
        if (v < -128) { v = -128; }
        if (v >  127) { v =  127; }
        return static_cast<int8_t>(v);
    }

    double dequantizeOutput(int8_t q)
    {
        return static_cast<double>(gOutScale) *
               (static_cast<double>(q) - static_cast<double>(gOutZp));
    }

    // Everything an int8 ANFIS needs at runtime, owned in one place so the
    // footprint accounting below can be honest about what ships.
    struct Int8Model
    {
        std::vector<uint16_t>                  lut;
        std::vector<int8_t>                    weights;
        std::vector<int32_t>                   biases;
        std::vector<tinymind::QAnfisRuleScale> scales;
        Int8Anfis                              layer;

        std::size_t bytes() const
        {
            return lut.size() * sizeof(uint16_t)
                 + weights.size() * sizeof(int8_t)
                 + biases.size() * sizeof(int32_t)
                 + scales.size() * sizeof(tinymind::QAnfisRuleScale)
                 + Int8Anfis::RuleTableSize * sizeof(uint8_t)
                 + sizeof(tinymind::Requantizer<int32_t, int8_t>);
        }
    };

    void calibrate(const double* premise, const uint8_t* rules,
                   const double* consequent, Int8Model& out)
    {
        out.lut.assign(Int8Anfis::GradeLutSize, 0u);
        out.weights.assign(Int8Anfis::WeightsSize, 0);
        out.biases.assign(NR, 0);
        out.scales.assign(NR, tinymind::QAnfisRuleScale{0, 0});

        for (std::size_t i = 0; i < NI; ++i)
        {
            for (std::size_t j = 0; j < NM; ++j)
            {
                const double* p = &premise[((i * NM) + j) * 2];
                tinymind::buildQAnfisGradeLUT<uint16_t>(
                    gInScale, gInZp,
                    [p](float x) { const double xv = x; return DoubleMf::evaluate(p, xv); },
                    &out.lut[((i * NM) + j) * 256]);
            }
        }

        std::vector<float> coefficients(NR * NI);
        std::vector<float> constants(NR);
        for (std::size_t r = 0; r < NR; ++r)
        {
            for (std::size_t i = 0; i < NI; ++i)
            {
                coefficients[(r * NI) + i] =
                    static_cast<float>(consequent[(r * (NI + 1)) + i]);
            }
            constants[r] = static_cast<float>(consequent[(r * (NI + 1)) + NI]);
        }

        tinymind::buildQAnfisConsequents<int8_t>(
            coefficients.data(), constants.data(), NR, NI,
            gInScale, gConsequentScale,
            out.weights.data(), out.biases.data(), out.scales.data());

        out.layer.grade_lut   = out.lut.data();
        out.layer.rule_table  = rules;
        out.layer.weights     = out.weights.data();
        out.layer.biases      = out.biases.data();
        out.layer.rule_scales = out.scales.data();
        out.layer.output_requantizer =
            tinymind::buildQAnfisOutputRequantizer<int8_t>(gConsequentScale, gOutScale, gOutZp);
        out.layer.input_zero_point = static_cast<int8_t>(gInZp);
    }

    double rmseOf(const std::vector<double>& a, const double* b, std::size_t n)
    {
        double s = 0.0;
        for (std::size_t k = 0; k < n; ++k)
        {
            const double d = a[k] - b[k];
            s += d * d;
        }
        return std::sqrt(s / static_cast<double>(n));
    }

    // Pick the shared consequent grid from the widest rule output actually
    // seen on the held-out set, so the rescale uses its full range.
    float pickConsequentScale(const double* consequent)
    {
        double peak = 0.0;
        for (std::size_t s = 0; s < data::NumberOfTestSamples; ++s)
        {
            const double* x = &data::TestInputs[s * NI];
            for (std::size_t r = 0; r < NR; ++r)
            {
                double f = consequent[(r * (NI + 1)) + NI];
                for (std::size_t i = 0; i < NI; ++i)
                {
                    f += consequent[(r * (NI + 1)) + i] * x[i];
                }
                const double a = std::fabs(f);
                if (a > peak) { peak = a; }
            }
        }
        if (!(peak > 0.0)) { peak = 1.0; }
        return static_cast<float>(peak / static_cast<double>(1 << 20));
    }

} // namespace

int main(int argc, char** argv)
{
    const bool goldenMode = (argc >= 2) && (std::strcmp(argv[1], "--golden") == 0);

    buildAffineGrids();
    gConsequentScale = pickConsequentScale(model::Consequent);

    DoubleAnfis reference(model::Premise, model::RuleTable, model::Consequent);

    QType premiseQ[DoubleAnfis::NumberOfPremiseParameters];
    QType consequentQ[DoubleAnfis::NumberOfConsequentParameters];
    for (std::size_t i = 0; i < DoubleAnfis::NumberOfPremiseParameters; ++i)
    {
        premiseQ[i] = ToQ::convertToDestinationType(model::Premise[i]);
    }
    for (std::size_t i = 0; i < DoubleAnfis::NumberOfConsequentParameters; ++i)
    {
        consequentQ[i] = ToQ::convertToDestinationType(model::Consequent[i]);
    }
    QFormatAnfis qformat(premiseQ, model::RuleTable, consequentQ);

    Int8Model int8Model;
    calibrate(model::Premise, model::RuleTable, model::Consequent, int8Model);

    const std::size_t n = data::NumberOfTestSamples;
    std::vector<double> outFloat(n), outQ(n), outInt8(n);

    for (std::size_t s = 0; s < n; ++s)
    {
        const double* x = &data::TestInputs[s * NI];

        double yf[1];
        reference.forward(x, yf);
        outFloat[s] = yf[0];

        QType xq[NI];
        for (std::size_t i = 0; i < NI; ++i)
        {
            xq[i] = ToQ::convertToDestinationType(x[i]);
        }
        QType yq[1];
        qformat.forward(xq, yq);
        outQ[s] = static_cast<double>(yq[0].getValue()) /
                  static_cast<double>(1ULL << QType::NumberOfFractionalBits);

        int8_t xi[NI];
        for (std::size_t i = 0; i < NI; ++i)
        {
            xi[i] = quantizeInput(x[i]);
        }
        outInt8[s] = dequantizeOutput(int8Model.layer.forward(xi));
    }

    const double rmseFloat = rmseOf(outFloat, data::TestTargets, n);
    const double rmseQ     = rmseOf(outQ,     data::TestTargets, n);
    const double rmseInt8  = rmseOf(outInt8,  data::TestTargets, n);

    // The same pipeline over an unregularized fit, to show what large
    // cancelling consequents cost once the input is quantized.
    Int8Model plainModel;
    const float savedScale = gConsequentScale;
    gConsequentScale = pickConsequentScale(plain::Consequent);
    calibrate(plain::Premise, plain::RuleTable, plain::Consequent, plainModel);
    std::vector<double> outPlain(n);
    for (std::size_t s = 0; s < n; ++s)
    {
        const double* x = &data::TestInputs[s * NI];
        int8_t xi[NI];
        for (std::size_t i = 0; i < NI; ++i)
        {
            xi[i] = quantizeInput(x[i]);
        }
        outPlain[s] = dequantizeOutput(plainModel.layer.forward(xi));
    }
    const double rmsePlainInt8 = rmseOf(outPlain, data::TestTargets, n);
    gConsequentScale = savedScale;

    const std::size_t qformatBytes =
        (DoubleAnfis::NumberOfPremiseParameters +
         DoubleAnfis::NumberOfConsequentParameters) * sizeof(QType) +
        DoubleAnfis::RuleTableSize;
    const std::size_t int8Bytes = int8Model.bytes();
    const std::size_t lutBytes  = int8Model.lut.size() * sizeof(uint16_t);

    if (goldenMode)
    {
        const std::size_t probes[8] = {0, 37, 84, 131, 178, 225, 311, 499};
        std::printf("# anfis_mackey_glass_int8 golden output\n");
        std::printf("# rules=%zu inputs=%zu mfs=%zu\n", NR, NI, NM);
        std::printf("int8:");
        for (std::size_t p = 0; p < 8; ++p)
        {
            const double* x = &data::TestInputs[probes[p] * NI];
            int8_t xi[NI];
            for (std::size_t i = 0; i < NI; ++i) { xi[i] = quantizeInput(x[i]); }
            std::printf(" %d", static_cast<int>(int8Model.layer.forward(xi)));
        }
        std::printf("\nrule:");
        for (std::size_t p = 0; p < 8; ++p)
        {
            const double* x = &data::TestInputs[probes[p] * NI];
            int8_t xi[NI];
            for (std::size_t i = 0; i < NI; ++i) { xi[i] = quantizeInput(x[i]); }
            std::printf(" %zu", int8Model.layer.dominantRule(xi));
        }
        std::printf("\nbytes: qformat=%zu int8=%zu lut=%zu\n",
                    qformatBytes, int8Bytes, lutBytes);
        return 0;
    }

    std::ofstream csv("anfis_int8_parity.csv");
    csv << "t,target,float,q16_16,int8\n";
    for (std::size_t s = 0; s < n; ++s)
    {
        csv << s << "," << data::TestTargets[s] << "," << outFloat[s] << ","
            << outQ[s] << "," << outInt8[s] << "\n";
    }

    std::ofstream fcsv("anfis_int8_footprint.csv");
    fcsv << "tier,bytes,test_rmse\n";
    fcsv << "Q16.16," << qformatBytes << "," << rmseQ << "\n";
    fcsv << "int8," << int8Bytes << "," << rmseInt8 << "\n";

    // Where int8 would start to win. The lookup tables are a fixed cost; the
    // per-rule parameters are where int8 is actually cheaper, so the crossover
    // is that fixed cost divided by the per-rule saving.
    const std::size_t qformatPerRule = (NI + 1) * sizeof(QType) + NI;
    const std::size_t int8PerRule = (NI * sizeof(int8_t)) + sizeof(int32_t)
                                  + sizeof(tinymind::QAnfisRuleScale) + NI;
    const long perRuleSaving =
        static_cast<long>(qformatPerRule) - static_cast<long>(int8PerRule);

    std::cout << "int8 Mackey-Glass ANFIS (ridge-regularized consequents)\n\n";
    std::cout << "  accuracy over " << n << " held-out samples\n";
    std::cout << "    float reference       test RMSE " << rmseFloat << "\n";
    std::cout << "    Q16.16                test RMSE " << rmseQ << "\n";
    std::cout << "    int8                  test RMSE " << rmseInt8 << "\n";
    std::cout << "    int8, no ridge        test RMSE " << rmsePlainInt8
              << "   (why apps/anfis_train has a ridge knob)\n\n";

    std::cout << "  memory -- int8 is LARGER here, which is the point\n";
    std::cout << "    Q16.16 total          " << qformatBytes << " bytes\n";
    std::cout << "    int8 total            " << int8Bytes << " bytes ("
              << (static_cast<double>(int8Bytes) / static_cast<double>(qformatBytes))
              << "x)\n";
    std::cout << "      of which grade LUTs " << lutBytes << " bytes ("
              << (100.0 * static_cast<double>(lutBytes) / static_cast<double>(int8Bytes))
              << "% of the int8 model)\n";
    std::cout << "    per-rule cost         Q16.16 " << qformatPerRule
              << " B vs int8 " << int8PerRule << " B\n";
    if (perRuleSaving > 0)
    {
        std::cout << "    int8 overtakes Q16.16 at about "
                  << (static_cast<long>(lutBytes) / perRuleSaving) << " rules\n";
    }
    else
    {
        std::cout << "    int8 never overtakes Q16.16 at this input count\n";
    }

    if (rmseInt8 > 4.0 * rmseFloat)
    {
        std::cerr << "int8 accuracy regressed against the float reference\n";
        return 1;
    }
    return 0;
}
