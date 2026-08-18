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
 * Mackey-Glass chaotic time-series prediction with a Takagi-Sugeno ANFIS --
 * Jang's original 1993 benchmark for the architecture.
 *
 * Predict x(t+6) from four delayed samples x(t-18), x(t-12), x(t-6), x(t).
 * Two generalized-bell membership functions per input over the full grid is
 * 2^4 = 16 rules with first-order consequents: 16 premise parameters and 80
 * consequent parameters, under 1 KB in Q16.16.
 *
 * The premise and consequent parameters were fitted on the host by
 * apps/anfis_train (hybrid least-squares + gradient descent) and are frozen
 * here -- cpp/anfis.hpp is inference only. Regenerate with
 * `make regenerate-model`.
 *
 * The example runs the same frozen model twice, in double and in Q16.16, so
 * the fixed-point cost is measured rather than assumed. It also reads the
 * rule base back: per-rule mean firing strength over the held-out set says
 * which rules actually carry the prediction, which is the reason to reach
 * for ANFIS over a dense net of the same size.
 *
 * Membership functions are generalized bells with the exponent pinned at 1,
 * so the inference path is compare/add/multiply/divide only -- no
 * transcendental function, and the model would deploy unchanged at
 * TINYMIND_ENABLE_FLOAT=0 / TINYMIND_ENABLE_STD=0.
 */

#include "anfis.hpp"
#include "qformat.hpp"
#include "include/nnproperties.hpp"

#include "anfis_model.hpp"
#include "anfis_data.hpp"

#include <cstddef>
#include <cstring>
#include <fstream>
#include <iostream>
#include <cmath>

namespace {

    namespace model = anfis_mackey_glass_model;
    namespace data = anfis_mackey_glass_data;

    typedef tinymind::QValue<16, 16, true, tinymind::RoundUpPolicy> QType;

    typedef tinymind::GeneralizedBellMembershipFunction<double, 1> DoubleMf;
    typedef tinymind::GeneralizedBellMembershipFunction<QType, 1> QMf;

    typedef tinymind::Anfis<double, model::NumberOfInputs,
                            model::NumberOfMembershipFunctionsPerInput,
                            model::NumberOfRules, DoubleMf, true,
                            model::NumberOfOutputs> DoubleAnfis;

    typedef tinymind::Anfis<QType, model::NumberOfInputs,
                            model::NumberOfMembershipFunctionsPerInput,
                            model::NumberOfRules, QMf, true,
                            model::NumberOfOutputs> QAnfis;

    typedef tinymind::ValueConverter<double, QType> ToQ;

    QType gPremiseQ[DoubleAnfis::NumberOfPremiseParameters];
    QType gConsequentQ[DoubleAnfis::NumberOfConsequentParameters];

    double gRuleImportance[model::NumberOfRules];

    void convertModelToFixedPoint()
    {
        for (std::size_t i = 0; i < DoubleAnfis::NumberOfPremiseParameters; ++i)
        {
            gPremiseQ[i] = ToQ::convertToDestinationType(model::Premise[i]);
        }

        for (std::size_t i = 0; i < DoubleAnfis::NumberOfConsequentParameters; ++i)
        {
            gConsequentQ[i] = ToQ::convertToDestinationType(model::Consequent[i]);
        }
    }

    double qToDouble(const QType& value)
    {
        return static_cast<double>(value.getValue()) /
               static_cast<double>(1ULL << QType::NumberOfFractionalBits);
    }

    void writeTrainingCsv()
    {
        std::ofstream csv("anfis_training.csv");
        csv << "epoch,train_rmse,test_rmse\n";
        for (std::size_t e = 0; e < data::NumberOfEpochs; ++e)
        {
            csv << e << "," << data::TrainRmse[e] << "," << data::TestRmse[e] << "\n";
        }
    }

    void writePruningCsv()
    {
        std::ofstream csv("anfis_pruning.csv");
        csv << "threshold,rules_kept,rules_total,train_rmse,test_rmse\n";
        for (std::size_t r = 0; r < data::NumberOfPruningRows; ++r)
        {
            csv << data::PruningThreshold[r] << ","
                << data::PruningRulesKept[r] << ","
                << data::PruningRulesTotal[r] << ","
                << data::PruningTrainRmse[r] << ","
                << data::PruningTestRmse[r] << "\n";
        }
    }

    void writeRulesCsv()
    {
        std::ofstream csv("anfis_rules.csv");
        csv << "rule,mean_firing_strength";
        for (std::size_t i = 0; i < model::NumberOfInputs; ++i)
        {
            csv << ",mf_input" << i;
        }
        csv << "\n";

        for (std::size_t r = 0; r < model::NumberOfRules; ++r)
        {
            csv << r << "," << gRuleImportance[r];
            for (std::size_t i = 0; i < model::NumberOfInputs; ++i)
            {
                csv << "," << static_cast<int>(
                    model::RuleTable[(r * model::NumberOfInputs) + i]);
            }
            csv << "\n";
        }
    }

} // namespace

// Emit a compact, deterministic byte stream for the golden-regression test in
// unit_test/integration.
//
// Only Q16.16 raw integers are printed, never doubles. The fixed-point path is
// pure integer arithmetic, so its bit patterns are reproducible across
// compilers, optimization levels, and hosts; formatted floating point is not,
// and a golden that drifts with the platform is worse than no golden at all.
// The rule indices and firing-strength raw values likewise pin the premise and
// t-norm stages, so a regression anywhere in the five-stage forward pass trips
// this string.
namespace {

int emitGolden(QAnfis& anfisQ)
{
    // Fixed probes spanning the held-out set rather than the first few
    // consecutive samples, so the stream covers the series' whole range.
    static const std::size_t probes[8] = {0, 37, 84, 131, 178, 225, 311, 499};

    std::printf("# anfis_mackey_glass golden output\n");
    std::printf("# rules=%zu inputs=%zu mfs=%zu q=16.16\n",
                static_cast<std::size_t>(model::NumberOfRules),
                static_cast<std::size_t>(model::NumberOfInputs),
                static_cast<std::size_t>(model::NumberOfMembershipFunctionsPerInput));

    std::printf("q16_16:");
    for (std::size_t p = 0; p < 8; ++p)
    {
        const double* sample =
            &data::TestInputs[probes[p] * model::NumberOfInputs];

        QType inputQ[model::NumberOfInputs];
        for (std::size_t i = 0; i < model::NumberOfInputs; ++i)
        {
            inputQ[i] = ToQ::convertToDestinationType(sample[i]);
        }

        QType outputQ[model::NumberOfOutputs];
        anfisQ.forward(inputQ, outputQ);
        std::printf(" %ld", static_cast<long>(outputQ[0].getValue()));
    }
    std::printf("\n");

    // Dominant rule per probe: pins the premise + product-t-norm stages, which
    // the defuzzified output alone could mask.
    std::printf("rule:");
    for (std::size_t p = 0; p < 8; ++p)
    {
        const double* sample =
            &data::TestInputs[probes[p] * model::NumberOfInputs];

        QType inputQ[model::NumberOfInputs];
        for (std::size_t i = 0; i < model::NumberOfInputs; ++i)
        {
            inputQ[i] = ToQ::convertToDestinationType(sample[i]);
        }

        QType outputQ[model::NumberOfOutputs];
        anfisQ.forward(inputQ, outputQ);
        std::printf(" %zu", anfisQ.getDominantRule());
    }
    std::printf("\n");

    return 0;
}

} // namespace

int main(int argc, char** argv)
{
    const bool goldenMode = (argc >= 2) && (std::strcmp(argv[1], "--golden") == 0);

    convertModelToFixedPoint();

    DoubleAnfis anfisDouble(model::Premise, model::RuleTable, model::Consequent);
    QAnfis anfisQ(gPremiseQ, model::RuleTable, gConsequentQ);

    if (goldenMode)
    {
        return emitGolden(anfisQ);
    }

    std::ofstream predictions("anfis_prediction.csv");
    predictions << "t,target,predicted_double,predicted_q16_16\n";

    double sumSquaredDouble = 0.0;
    double sumSquaredQ = 0.0;
    double maxAbsGap = 0.0;

    for (std::size_t r = 0; r < model::NumberOfRules; ++r)
    {
        gRuleImportance[r] = 0.0;
    }

    for (std::size_t s = 0; s < data::NumberOfTestSamples; ++s)
    {
        const double* sample = &data::TestInputs[s * model::NumberOfInputs];
        const double target = data::TestTargets[s];

        double outputDouble[model::NumberOfOutputs];
        anfisDouble.forward(sample, outputDouble);

        QType inputQ[model::NumberOfInputs];
        for (std::size_t i = 0; i < model::NumberOfInputs; ++i)
        {
            inputQ[i] = ToQ::convertToDestinationType(sample[i]);
        }

        QType outputQ[model::NumberOfOutputs];
        anfisQ.forward(inputQ, outputQ);
        const double predictedQ = qToDouble(outputQ[0]);

        const double errorDouble = outputDouble[0] - target;
        const double errorQ = predictedQ - target;
        sumSquaredDouble += errorDouble * errorDouble;
        sumSquaredQ += errorQ * errorQ;

        const double gap = std::fabs(predictedQ - outputDouble[0]);
        if (gap > maxAbsGap)
        {
            maxAbsGap = gap;
        }

        // Read the rule base back: how much of the output did each rule own?
        for (std::size_t r = 0; r < model::NumberOfRules; ++r)
        {
            gRuleImportance[r] += anfisDouble.getNormalizedFiringStrength(r);
        }

        predictions << s << "," << target << "," << outputDouble[0] << ","
                    << predictedQ << "\n";
    }

    const double n = static_cast<double>(data::NumberOfTestSamples);
    for (std::size_t r = 0; r < model::NumberOfRules; ++r)
    {
        gRuleImportance[r] /= n;
    }

    const double rmseDouble = std::sqrt(sumSquaredDouble / n);
    const double rmseQ = std::sqrt(sumSquaredQ / n);

    writeTrainingCsv();
    writePruningCsv();
    writeRulesCsv();

    std::size_t dominantRule = 0;
    for (std::size_t r = 1; r < model::NumberOfRules; ++r)
    {
        if (gRuleImportance[r] > gRuleImportance[dominantRule])
        {
            dominantRule = r;
        }
    }

    std::cout << "Mackey-Glass ANFIS -- x(t+6) from x(t-18), x(t-12), x(t-6), x(t)\n";
    std::cout << "  rules                 " << model::NumberOfRules << "\n";
    std::cout << "  premise parameters    "
              << DoubleAnfis::NumberOfPremiseParameters << "\n";
    std::cout << "  consequent parameters "
              << DoubleAnfis::NumberOfConsequentParameters << "\n";
    std::cout << "  Q16.16 model bytes    "
              << ((DoubleAnfis::NumberOfPremiseParameters +
                   DoubleAnfis::NumberOfConsequentParameters) * sizeof(QType) +
                  DoubleAnfis::RuleTableSize) << "\n";
    std::cout << "  held-out samples      " << data::NumberOfTestSamples << "\n";
    // What the host trainer measured for this same frozen model.
    const double trainerRmse = data::TestRmse[data::NumberOfEpochs - 1];

    std::cout << "  test RMSE (double)    " << rmseDouble << "\n";
    std::cout << "  test RMSE (Q16.16)    " << rmseQ << "\n";
    std::cout << "  test RMSE (trainer)   " << trainerRmse << "\n";
    std::cout << "  max |Q16.16 - double| " << maxAbsGap << "\n";
    std::cout << "  strongest rule        " << dominantRule
              << " (mean firing strength " << gRuleImportance[dominantRule] << ")\n";

    // The fixed-point path must track the double reference closely; a large
    // divergence means the Q format ran out of headroom for the weighted sum.
    if (maxAbsGap > 0.01)
    {
        std::cerr << "Q16.16 diverged from the double reference\n";
        return 1;
    }

    // Cross-check the two independent implementations of the same forward
    // pass: this C++ one and the numpy one in apps/anfis_train that fitted the
    // parameters. They evaluate identical math on identical inputs, so they
    // must agree to well within the text round-trip of the emitted constants.
    // A drift here means one of the two changed without the other.
    if (std::fabs(rmseDouble - trainerRmse) > 1.0e-6)
    {
        std::cerr << "C++ forward pass disagrees with the host trainer: "
                  << rmseDouble << " vs " << trainerRmse << "\n";
        return 1;
    }

    return 0;
}
