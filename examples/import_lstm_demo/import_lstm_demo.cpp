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

// LSTM importer round-trip.
//
// Consumes the Python-emitted weights_cell16.hpp / weights_cell8.hpp
// (apps/import_pytorch importer, one nn.LSTM followed by a linear head) and
// rebuilds cpp/qlstm.hpp::QLSTMCell and cpp/qdense.hpp::QDense from the
// generated constants:
//
//   * QLSTMScales is filled from the emitted float scales and handed to
//     buildQLSTMParams for every multiplier and shift the cell consumes;
//   * the sigmoid and tanh LUTs are built on the emitted LUT input scale;
//   * the head's requantizer comes from buildRequantizer, its input grid
//     being the LSTM hidden state's.
//
// For each held-out test sequence the binary quantizes every step with the
// emitted input scale / zero point, runs the cell from zero state, runs the
// head on the last hidden state and compares the dequantized output with
// the float model's. It does this for the int16 and the int8 cell-state
// builds and writes both to output/import_lstm_demo.csv, plus the hidden
// state of test sequence 0 after every step to
// output/import_lstm_demo_trace.csv.

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>

#include "qactivations.hpp"
#include "qaffine.hpp"
#include "qcalibration.hpp"
#include "qdense.hpp"
#include "qlstm.hpp"

#include "lstm_reference.hpp"
#include "weights_cell16.hpp"
#include "weights_cell8.hpp"

namespace {

using namespace import_lstm_demo;

namespace W16 = import_lstm_demo_cell16;
namespace W8  = import_lstm_demo_cell8;

static_assert(W16::klstm_NumInputs == NumInputs, "input width");
static_assert(W16::klstm_NumHidden == NumHidden, "hidden width");
static_assert(W16::klstm_SequenceLength == SequenceLength, "sequence length");
static_assert(W8::klstm_NumInputs == NumInputs, "input width");
static_assert(W8::klstm_NumHidden == NumHidden, "hidden width");
static_assert(W8::klstm_SequenceLength == SequenceLength, "sequence length");

constexpr int32_t kQmin = -128;
constexpr int32_t kQmax =  127;

// Everything one emitted header provides, gathered so a single function
// template can run either cell-state width.
struct ImportedModel
{
    float          input_scale;
    int32_t        input_zero_point;
    float          hidden_scale;
    int32_t        hidden_zero_point;
    float          cell_scale;
    int32_t        cell_zero_point;
    int32_t        cell_qmin;
    int32_t        cell_qmax;
    float          lut_input_scale;
    const float*   input_weight_scales;
    const float*   recurrent_weight_scales;
    const int8_t*  input_weights;
    const int8_t*  recurrent_weights;
    const int32_t* lstm_biases;
    float          head_weight_scale;
    const int8_t*  head_weights;
    const int32_t* head_biases;
    float          output_scale;
    int32_t        output_zero_point;
};

const ImportedModel kModelCell16 = {
    W16::kinput_Scale, W16::kinput_ZeroPoint,
    W16::klstm_Scale, W16::klstm_ZeroPoint,
    W16::klstm_CellScale, W16::klstm_CellZeroPoint,
    W16::klstm_CellQMin, W16::klstm_CellQMax,
    W16::klstm_LutInputScale,
    W16::klstm_InputWeightScales, W16::klstm_RecurrentWeightScales,
    W16::klstm_InputWeights, W16::klstm_RecurrentWeights, W16::klstm_Biases,
    W16::khead_WeightScale, W16::khead_Weights, W16::khead_Biases,
    W16::khead_Scale, W16::khead_ZeroPoint,
};

const ImportedModel kModelCell8 = {
    W8::kinput_Scale, W8::kinput_ZeroPoint,
    W8::klstm_Scale, W8::klstm_ZeroPoint,
    W8::klstm_CellScale, W8::klstm_CellZeroPoint,
    W8::klstm_CellQMin, W8::klstm_CellQMax,
    W8::klstm_LutInputScale,
    W8::klstm_InputWeightScales, W8::klstm_RecurrentWeightScales,
    W8::klstm_InputWeights, W8::klstm_RecurrentWeights, W8::klstm_Biases,
    W8::khead_WeightScale, W8::khead_Weights, W8::khead_Biases,
    W8::khead_Scale, W8::khead_ZeroPoint,
};

struct ParityResult
{
    float max_abs_err;
    float max_hidden_err;
};

template<typename CellType>
ParityResult runModel(const ImportedModel& m, const int cell_bits,
                      std::ofstream& csv, std::ofstream& trace_csv)
{
    typedef tinymind::QLSTMCell<int8_t, int8_t, int32_t, int8_t, int8_t, CellType,
                                NumInputs, NumHidden> Cell;
    typedef tinymind::QDense<int8_t, int8_t, int32_t, int8_t,
                             NumHidden, NumOutputs> Head;

    tinymind::QLSTMScales sc;
    sc.input_scale     = m.input_scale;
    sc.hidden_scale    = m.hidden_scale;
    sc.cell_scale      = m.cell_scale;
    sc.lut_input_scale = m.lut_input_scale;
    for (std::size_t g = 0; g < tinymind::kQLstmNumGates; ++g)
    {
        sc.w_input_scale[g]     = m.input_weight_scales[g];
        sc.w_recurrent_scale[g] = m.recurrent_weight_scales[g];
    }
    tinymind::QLSTMParams p;
    tinymind::buildQLSTMParams(sc, p);

    // One tanh LUT serves both the g gate and tanh(c): both read the same
    // LUT input grid and write the fixed 1/128 tanh grid.
    static int8_t sigmoid_lut[tinymind::kQActivationLUTSize];
    static int8_t tanh_lut[tinymind::kQActivationLUTSize];
    tinymind::buildQSigmoidLUT(m.lut_input_scale, 0, tinymind::kQLstmSigmoidOutputScale,
                               tinymind::kQLstmSigmoidZeroPoint, sigmoid_lut);
    tinymind::buildQTanhLUT(m.lut_input_scale, 0, tinymind::kQLstmTanhOutputScale,
                            tinymind::kQLstmTanhZeroPoint, tanh_lut);

    Cell cell;
    cell.w_input           = m.input_weights;
    cell.w_recurrent       = m.recurrent_weights;
    cell.biases            = m.lstm_biases;
    cell.input_zero_point  = static_cast<int8_t>(m.input_zero_point);
    cell.hidden_zero_point = static_cast<int8_t>(m.hidden_zero_point);
    cell.cell_zero_point   = static_cast<CellType>(m.cell_zero_point);
    for (std::size_t g = 0; g < tinymind::kQLstmNumGates; ++g)
    {
        cell.input_to_lut_multiplier[g]     = p.input_to_lut_multiplier[g];
        cell.input_to_lut_shift[g]          = p.input_to_lut_shift[g];
        cell.recurrent_to_lut_multiplier[g] = p.recurrent_to_lut_multiplier[g];
        cell.recurrent_to_lut_shift[g]      = p.recurrent_to_lut_shift[g];
    }
    cell.sigmoid_lut             = sigmoid_lut;
    cell.tanh_lut                = tanh_lut;
    cell.tanh_cell_lut           = tanh_lut;
    cell.f_times_c_multiplier    = p.f_times_c_multiplier;
    cell.f_times_c_shift         = p.f_times_c_shift;
    cell.i_times_g_multiplier    = p.i_times_g_multiplier;
    cell.i_times_g_shift         = p.i_times_g_shift;
    cell.cell_qmin               = static_cast<CellType>(m.cell_qmin);
    cell.cell_qmax               = static_cast<CellType>(m.cell_qmax);
    cell.cell_to_tanh_multiplier = p.cell_to_tanh_multiplier;
    cell.cell_to_tanh_shift      = p.cell_to_tanh_shift;
    cell.output_requantizer.multiplier = p.output_multiplier;
    cell.output_requantizer.shift      = p.output_shift;
    cell.output_requantizer.zero_point = m.hidden_zero_point;
    cell.output_requantizer.qmin       = kQmin;
    cell.output_requantizer.qmax       = kQmax;

    Head head;
    head.weights          = m.head_weights;
    head.biases           = m.head_biases;
    head.input_zero_point = static_cast<int8_t>(m.hidden_zero_point);
    head.requantizer      = tinymind::buildRequantizer<int8_t>(
        m.hidden_scale, m.head_weight_scale, m.output_scale, m.output_zero_point,
        kQmin, kQmax);

    ParityResult r = { 0.0f, 0.0f };
    for (std::size_t s = 0; s < NumTest; ++s)
    {
        int8_t   h[NumHidden];
        CellType c[NumHidden];
        for (std::size_t u = 0; u < NumHidden; ++u)
        {
            h[u] = static_cast<int8_t>(m.hidden_zero_point);
            c[u] = static_cast<CellType>(m.cell_zero_point);
        }

        for (std::size_t t = 0; t < SequenceLength; ++t)
        {
            int8_t qx[NumInputs];
            for (std::size_t i = 0; i < NumInputs; ++i)
            {
                qx[i] = tinymind::quantize<int8_t>(kTestSequences[s][t][i],
                                                   m.input_scale, m.input_zero_point,
                                                   kQmin, kQmax);
            }
            cell.forward(qx, h, c);

            if (s == 0)
            {
                for (std::size_t u = 0; u < NumHidden; ++u)
                {
                    const float hq = tinymind::dequantize<int8_t>(
                        h[u], m.hidden_scale, m.hidden_zero_point);
                    const float err = std::fabs(hq - kFloatHiddenTrace[t][u]);
                    if (err > r.max_hidden_err)
                    {
                        r.max_hidden_err = err;
                    }
                    trace_csv << cell_bits << "," << t << "," << u << ","
                              << kFloatHiddenTrace[t][u] << "," << hq << "\n";
                }
            }
        }

        int8_t qy[NumOutputs];
        head.forward(h, qy);
        for (std::size_t o = 0; o < NumOutputs; ++o)
        {
            const float yq = tinymind::dequantize<int8_t>(
                qy[o], m.output_scale, m.output_zero_point);
            const float err = std::fabs(yq - kFloatOutputs[s][o]);
            if (err > r.max_abs_err)
            {
                r.max_abs_err = err;
            }
            csv << cell_bits << "," << s << "," << o << ","
                << kFloatOutputs[s][o] << "," << yq << "\n";
        }
    }
    return r;
}

} // namespace

int main()
{
    std::ofstream csv("output/import_lstm_demo.csv");
    std::ofstream trace_csv("output/import_lstm_demo_trace.csv");
    if (!csv || !trace_csv)
    {
        std::cerr << "cannot open output/*.csv (run from the example directory)\n";
        return 1;
    }
    csv << std::setprecision(9);
    trace_csv << std::setprecision(9);
    csv << "cell_bits,sample,output,float,int8\n";
    trace_csv << "cell_bits,step,unit,float,int8\n";

    const ParityResult r16 = runModel<int16_t>(kModelCell16, 16, csv, trace_csv);
    const ParityResult r8  = runModel<int8_t>(kModelCell8, 8, csv, trace_csv);

    // Measured on the bundled model: max 0.063 (int16 cell) and 0.074 (int8
    // cell), mean 0.022 and 0.035; the head's output grid is ~0.005 per LSB.
    // The error is already 0.018 after the first step and does not grow over
    // the 24 steps: it is the floor set by the fixed 8/127 gate grid and the
    // 1/128 tanh LUT, as in the single-step QLSTMCell unit test, not drift.
    constexpr float kTolerance16 = 0.08f;
    constexpr float kTolerance8  = 0.08f;
    const bool ok16 = r16.max_abs_err <= kTolerance16;
    const bool ok8  = r8.max_abs_err <= kTolerance8;

    std::cout << std::fixed << std::setprecision(6);
    std::cout << "import_lstm_demo parity test (" << NumTest << " sequences x "
              << SequenceLength << " steps, " << NumInputs << "-" << NumHidden
              << " LSTM + " << NumHidden << "-" << NumOutputs << " head)\n";
    std::cout << "  int16 cell: max |y_int8 - y_float| = " << r16.max_abs_err
              << "  (hidden trace " << r16.max_hidden_err << ")  tolerance "
              << kTolerance16 << " : " << (ok16 ? "PASS" : "FAIL") << "\n";
    std::cout << "  int8  cell: max |y_int8 - y_float| = " << r8.max_abs_err
              << "  (hidden trace " << r8.max_hidden_err << ")  tolerance "
              << kTolerance8 << " : " << (ok8 ? "PASS" : "FAIL") << "\n";
    return (ok16 && ok8) ? 0 : 1;
}
