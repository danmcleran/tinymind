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
#include "qaffine.hpp"
#include "qactivations.hpp"

#include <cstddef>
#include <cstdint>

/*
 * Quantized LSTM cell (single time step).
 *
 * Standalone layer mirroring the affine quantization model used by
 * TFLite / CMSIS-NN reference kernels. Does NOT integrate with the
 * NeuralNet<> template; weights, biases, requantizer constants, and the
 * sigmoid/tanh LUTs are caller-owned. The cell processes one time step
 * and updates the hidden and cell state buffers in place; the caller
 * drives the time loop.
 *
 * Type contract:
 *
 *   InputStorage  = int8_t      input activations (asymmetric)
 *   WeightStorage = int8_t      symmetric per-tensor weights
 *   AccumStorage  = int32_t     MAC accumulator and bias storage
 *   GateActStorage = int8_t     LUT input grid (post-MAC pre-activation)
 *   HiddenStorage = int8_t      hidden state (asymmetric)
 *   CellStorage   = int8_t or int16_t (Phase 12 wide-cell option)
 *
 * Math (one time step):
 *
 *   acc_in_g [h] = sum_j W_in[g, h, j]  * (x[j]       - x_zp)        (int32)
 *   acc_rec_g[h] = sum_k W_rec[g, h, k] * (h_prev[k]  - h_zp)        (int32)
 *
 *   pre_act_g[h] = bias[g, h]
 *                + multiplyByQuantizedMultiplier(acc_in_g [h], m_in [g], s_in [g])
 *                + multiplyByQuantizedMultiplier(acc_rec_g[h], m_rec[g], s_rec[g])
 *                (int32, in LUT input scale)
 *
 *   pre_act_g[h] is saturated to int8 then fed into the LUT:
 *     i_t[h] = sigmoid_lut[idx(pre_act_i[h])]   (1/256, zp=-128)
 *     f_t[h] = sigmoid_lut[idx(pre_act_f[h])]   (1/256, zp=-128)
 *     g_t[h] = tanh_lut   [idx(pre_act_g[h])]   (1/128, zp=0)
 *     o_t[h] = sigmoid_lut[idx(pre_act_o[h])]   (1/256, zp=-128)
 *
 *   c_t[h] = clamp(
 *              multiplyByQuantizedMultiplier((f_t[h] + 128) * (c_prev[h] - c_zp),
 *                                            m_fc, s_fc)
 *            + multiplyByQuantizedMultiplier((i_t[h] + 128) * g_t[h],
 *                                            m_ig, s_ig)
 *            + c_zp,
 *            cell_qmin, cell_qmax)
 *
 *   tanh_c[h] = tanh_cell_lut[idx(saturate_int8(
 *                  multiplyByQuantizedMultiplier(c_t[h] - c_zp,
 *                                                m_ctlut, s_ctlut))))]
 *
 *   h_t[h] = output_requantizer.apply((o_t[h] + 128) * tanh_c[h])
 *
 * Gate ordering is {i, f, g, o} matching TFLite. weights buffers are
 * gate-major: w_input[g * NumHidden * NumInputs + h * NumInputs + j].
 *
 * Pure integer at runtime: no float, no <cmath>, no stdlib. Safe to
 * compile under TINYMIND_ENABLE_FLOAT=0, TINYMIND_ENABLE_STD=0.
 *
 * Cell-state precision:
 *   CellStorage_ = int8_t  (default): 256 levels of cell dynamic range;
 *                                     adequate for short sequences.
 *   CellStorage_ = int16_t          : 65k levels; recommended for long
 *                                     unroll horizons. Requires the host
 *                                     calibration build to opt in to
 *                                     TINYMIND_ENABLE_INT16_ACCUM if the
 *                                     deployable target wants the gate
 *                                     advertised (the runtime types are
 *                                     gated only on the C++ type, not the
 *                                     macro, so embedded toolchains
 *                                     without int16 storage need not
 *                                     instantiate this corner).
 */

namespace tinymind {

    /**
     * Gate index convention for QLSTM. Mirrors TFLite's lstm_eval ordering.
     */
    static constexpr std::size_t kQLstmGateInput  = 0;
    static constexpr std::size_t kQLstmGateForget = 1;
    static constexpr std::size_t kQLstmGateCell   = 2;
    static constexpr std::size_t kQLstmGateOutput = 3;
    static constexpr std::size_t kQLstmNumGates   = 4;

    /**
     * Fixed sigmoid LUT output convention: scale 1/256, zero_point -128.
     * Fixed tanh    LUT output convention: scale 1/128, zero_point  0.
     * Embedded as a constant so the cell math is bit-stable; calibration
     * helpers in qcalibration.hpp respect the same convention.
     */
    static constexpr int32_t kQLstmSigmoidZeroPoint = -128;
    static constexpr int32_t kQLstmTanhZeroPoint    = 0;

    namespace detail {

        inline int32_t qLstmSaturateInt8(int32_t v)
        {
            if (v < -128) return -128;
            if (v >  127) return  127;
            return v;
        }

    } // namespace detail

    template<
        typename InputStorage_,
        typename WeightStorage_,
        typename AccumStorage_,
        typename GateActStorage_,
        typename HiddenStorage_,
        typename CellStorage_,
        std::size_t NumInputs_,
        std::size_t NumHidden_>
    struct QLSTMCell
    {
        typedef InputStorage_    InputType;
        typedef WeightStorage_   WeightType;
        typedef AccumStorage_    AccumType;
        typedef GateActStorage_  GateActType;
        typedef HiddenStorage_   HiddenType;
        typedef CellStorage_     CellType;

        static constexpr std::size_t NumInputs = NumInputs_;
        static constexpr std::size_t NumHidden = NumHidden_;
        static constexpr std::size_t NumGates  = kQLstmNumGates;

        // [4][NumHidden][NumInputs], gate-major.
        const WeightType* w_input;
        // [4][NumHidden][NumHidden], gate-major.
        const WeightType* w_recurrent;
        // [4][NumHidden], gate-major, int32 in the LUT input scale.
        const AccumType* biases;

        InputType  input_zero_point;
        HiddenType hidden_zero_point;
        CellType   cell_zero_point;

        // Per-gate rescalers from MAC accumulator to LUT input scale.
        int32_t input_to_lut_multiplier   [kQLstmNumGates];
        int32_t input_to_lut_shift        [kQLstmNumGates];
        int32_t recurrent_to_lut_multiplier[kQLstmNumGates];
        int32_t recurrent_to_lut_shift    [kQLstmNumGates];

        // Sigmoid LUT (i, f, o) and tanh LUT (g). Both 256-entry int8.
        const int8_t* sigmoid_lut;
        const int8_t* tanh_lut;

        // f_t * c_prev -> cell scale.
        int32_t f_times_c_multiplier;
        int32_t f_times_c_shift;
        // i_t * g_t -> cell scale.
        int32_t i_times_g_multiplier;
        int32_t i_times_g_shift;
        CellType cell_qmin;
        CellType cell_qmax;

        // c_t -> tanh LUT input grid.
        int32_t cell_to_tanh_multiplier;
        int32_t cell_to_tanh_shift;
        const int8_t* tanh_cell_lut;

        // o_t * tanh(c_t) -> hidden grid (zero_point + saturation in apply()).
        Requantizer<int32_t, HiddenType> output_requantizer;

        void forward(const InputType* x,
                     HiddenType*      h_state,
                     CellType*        c_state) const
        {
            const int32_t x_zp = static_cast<int32_t>(input_zero_point);
            const int32_t h_zp = static_cast<int32_t>(hidden_zero_point);
            const int32_t c_zp = static_cast<int32_t>(cell_zero_point);

            // Scratch holds the four pre-activation int8 vectors; sized at
            // compile time so the cell stays caller-owned-buffer with no
            // dynamic allocation in the runtime path.
            int8_t pre_act[kQLstmNumGates * NumHidden_];

            for (std::size_t g = 0; g < kQLstmNumGates; ++g)
            {
                const WeightType* w_in_g  = w_input
                    + g * NumHidden_ * NumInputs_;
                const WeightType* w_rec_g = w_recurrent
                    + g * NumHidden_ * NumHidden_;
                const AccumType*  bias_g  = biases + g * NumHidden_;
                const int32_t mi = input_to_lut_multiplier[g];
                const int32_t si = input_to_lut_shift     [g];
                const int32_t mr = recurrent_to_lut_multiplier[g];
                const int32_t sr = recurrent_to_lut_shift     [g];

                for (std::size_t hi = 0; hi < NumHidden_; ++hi)
                {
                    int32_t acc_in = 0;
                    const WeightType* w_row_i = w_in_g + hi * NumInputs_;
                    for (std::size_t j = 0; j < NumInputs_; ++j)
                    {
                        const int32_t x_minus_zp =
                            static_cast<int32_t>(x[j]) - x_zp;
                        acc_in += static_cast<int32_t>(w_row_i[j]) * x_minus_zp;
                    }

                    int32_t acc_rec = 0;
                    const WeightType* w_row_r = w_rec_g + hi * NumHidden_;
                    for (std::size_t k = 0; k < NumHidden_; ++k)
                    {
                        const int32_t h_minus_zp =
                            static_cast<int32_t>(h_state[k]) - h_zp;
                        acc_rec += static_cast<int32_t>(w_row_r[k]) * h_minus_zp;
                    }

                    const int32_t r_in  =
                        multiplyByQuantizedMultiplier(acc_in,  mi, si);
                    const int32_t r_rec =
                        multiplyByQuantizedMultiplier(acc_rec, mr, sr);
                    const int32_t pre_int32 =
                        r_in + r_rec + static_cast<int32_t>(bias_g[hi]);

                    pre_act[g * NumHidden_ + hi] =
                        static_cast<int8_t>(detail::qLstmSaturateInt8(pre_int32));
                }
            }

            // Cell update and hidden update share the gate post-activation
            // vectors, so they fold into one pass over hidden units.
            const int32_t cell_lo = static_cast<int32_t>(cell_qmin);
            const int32_t cell_hi = static_cast<int32_t>(cell_qmax);

            for (std::size_t hi = 0; hi < NumHidden_; ++hi)
            {
                const int8_t i_act = qApplyLUT(
                    pre_act[kQLstmGateInput  * NumHidden_ + hi], sigmoid_lut);
                const int8_t f_act = qApplyLUT(
                    pre_act[kQLstmGateForget * NumHidden_ + hi], sigmoid_lut);
                const int8_t g_act = qApplyLUT(
                    pre_act[kQLstmGateCell   * NumHidden_ + hi], tanh_lut);
                const int8_t o_act = qApplyLUT(
                    pre_act[kQLstmGateOutput * NumHidden_ + hi], sigmoid_lut);

                // Sigmoid output uses fixed zero_point -128; tanh uses 0.
                // Subtracting the LUT zero_point recovers the "centered"
                // int domain so the elementwise products carry sign correctly.
                const int32_t i_centered =
                    static_cast<int32_t>(i_act) - kQLstmSigmoidZeroPoint;
                const int32_t f_centered =
                    static_cast<int32_t>(f_act) - kQLstmSigmoidZeroPoint;
                const int32_t g_centered =
                    static_cast<int32_t>(g_act) - kQLstmTanhZeroPoint;
                const int32_t o_centered =
                    static_cast<int32_t>(o_act) - kQLstmSigmoidZeroPoint;

                const int32_t c_prev_centered =
                    static_cast<int32_t>(c_state[hi]) - c_zp;

                const int32_t prod_fc = f_centered * c_prev_centered;
                const int32_t prod_ig = i_centered * g_centered;

                const int32_t scaled_fc = multiplyByQuantizedMultiplier(
                    prod_fc, f_times_c_multiplier, f_times_c_shift);
                const int32_t scaled_ig = multiplyByQuantizedMultiplier(
                    prod_ig, i_times_g_multiplier, i_times_g_shift);

                int32_t c_new = scaled_fc + scaled_ig + c_zp;
                if (c_new < cell_lo) c_new = cell_lo;
                if (c_new > cell_hi) c_new = cell_hi;
                c_state[hi] = static_cast<CellType>(c_new);

                // tanh(c_new) via LUT keyed by an int8 representation of
                // (c_new - c_zp). The rescale lands in the tanh LUT input
                // grid (typically the same one used by the gate LUTs).
                const int32_t c_for_lut = detail::qLstmSaturateInt8(
                    multiplyByQuantizedMultiplier(
                        c_new - c_zp,
                        cell_to_tanh_multiplier,
                        cell_to_tanh_shift));
                const int8_t tanh_c = qApplyLUT(
                    static_cast<int8_t>(c_for_lut), tanh_cell_lut);
                const int32_t tanh_c_centered =
                    static_cast<int32_t>(tanh_c) - kQLstmTanhZeroPoint;

                const int32_t prod_oc = o_centered * tanh_c_centered;
                h_state[hi] = output_requantizer.apply(prod_oc);
            }
        }
    };

} // namespace tinymind
