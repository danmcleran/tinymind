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
 * Quantized GRU cell (single time step), reset-before-multiply variant.
 *
 *   r_t = sigmoid(W_r x + R_r h_{t-1} + b_r)
 *   z_t = sigmoid(W_z x + R_z h_{t-1} + b_z)
 *   n_t = tanh   (W_n x + R_n (r_t * h_{t-1})           + b_n)
 *   h_t = (1 - z_t) * n_t + z_t * h_{t-1}
 *
 * Gate ordering is {r, z, n}. Reset gate (r) and update gate (z) follow
 * the same input/recurrent two-MAC pattern as QLSTM. The new gate (n)
 * uses r_t * h_{t-1} as its recurrent input; the elementwise product is
 * materialized in the int8 hidden grid before the R_n MAC consumes it.
 *
 * Sigmoid LUT output convention: scale 1/256, zero_point -128
 * Tanh    LUT output convention: scale 1/128, zero_point  0
 *
 * Pure integer at runtime; freestanding-safe under FLOAT=0 / STD=0.
 */

namespace tinymind {

    static constexpr std::size_t kQGruGateReset  = 0;
    static constexpr std::size_t kQGruGateUpdate = 1;
    static constexpr std::size_t kQGruGateNew    = 2;
    static constexpr std::size_t kQGruNumGates   = 3;

    static constexpr int32_t kQGruSigmoidZeroPoint = -128;
    static constexpr int32_t kQGruTanhZeroPoint    = 0;

    namespace detail {

        inline int32_t qGruSaturateInt8(int32_t v)
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
        std::size_t NumInputs_,
        std::size_t NumHidden_>
    struct QGRUCell
    {
        typedef InputStorage_   InputType;
        typedef WeightStorage_  WeightType;
        typedef AccumStorage_   AccumType;
        typedef GateActStorage_ GateActType;
        typedef HiddenStorage_  HiddenType;

        static constexpr std::size_t NumInputs = NumInputs_;
        static constexpr std::size_t NumHidden = NumHidden_;
        static constexpr std::size_t NumGates  = kQGruNumGates;

        // [3][NumHidden][NumInputs], gate-major: r, z, n.
        const WeightType* w_input;
        // [3][NumHidden][NumHidden], gate-major: r, z, n.
        const WeightType* w_recurrent;
        // [3][NumHidden], int32 biases in LUT input scale.
        const AccumType* biases;

        InputType  input_zero_point;
        HiddenType hidden_zero_point;

        int32_t input_to_lut_multiplier   [kQGruNumGates];
        int32_t input_to_lut_shift        [kQGruNumGates];
        int32_t recurrent_to_lut_multiplier[kQGruNumGates];
        int32_t recurrent_to_lut_shift    [kQGruNumGates];

        const int8_t* sigmoid_lut;
        const int8_t* tanh_lut;

        // r_t * h_prev -> hidden scale grid (for feeding R_n MAC).
        int32_t r_times_h_multiplier;
        int32_t r_times_h_shift;

        // (1 - z_t) * n_t -> hidden scale.
        int32_t one_minus_z_times_n_multiplier;
        int32_t one_minus_z_times_n_shift;
        // z_t * h_prev -> hidden scale.
        int32_t z_times_h_multiplier;
        int32_t z_times_h_shift;

        HiddenType output_qmin;
        HiddenType output_qmax;

        void forward(const InputType* x, HiddenType* h_state) const
        {
            const int32_t x_zp = static_cast<int32_t>(input_zero_point);
            const int32_t h_zp = static_cast<int32_t>(hidden_zero_point);

            // Compute r_t (sigmoid) and z_t (sigmoid); cache the resulting
            // int8 activations for downstream use in n_t and h_t.
            int8_t r_act[NumHidden_];
            int8_t z_act[NumHidden_];

            for (std::size_t gate_idx = 0; gate_idx < 2; ++gate_idx)
            {
                const std::size_t g = (gate_idx == 0)
                    ? kQGruGateReset
                    : kQGruGateUpdate;

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
                    const int8_t pre_i8 = static_cast<int8_t>(
                        detail::qGruSaturateInt8(pre_int32));
                    if (gate_idx == 0) r_act[hi] = qApplyLUT(pre_i8, sigmoid_lut);
                    else               z_act[hi] = qApplyLUT(pre_i8, sigmoid_lut);
                }
            }

            // rh[k] = r_t[k] * h_prev[k]   in hidden int8 grid.
            int8_t rh[NumHidden_];
            for (std::size_t k = 0; k < NumHidden_; ++k)
            {
                const int32_t r_centered =
                    static_cast<int32_t>(r_act[k]) - kQGruSigmoidZeroPoint;
                const int32_t h_minus_zp =
                    static_cast<int32_t>(h_state[k]) - h_zp;
                const int32_t prod = r_centered * h_minus_zp;
                const int32_t scaled = multiplyByQuantizedMultiplier(
                    prod, r_times_h_multiplier, r_times_h_shift);
                const int32_t with_zp = scaled + h_zp;
                rh[k] = static_cast<int8_t>(detail::qGruSaturateInt8(with_zp));
            }

            // n_t: same two-MAC structure as r/z but the recurrent MAC
            // consumes rh instead of h_prev.
            const std::size_t g_new = kQGruGateNew;
            const WeightType* w_in_n  = w_input
                + g_new * NumHidden_ * NumInputs_;
            const WeightType* w_rec_n = w_recurrent
                + g_new * NumHidden_ * NumHidden_;
            const AccumType*  bias_n  = biases + g_new * NumHidden_;
            const int32_t mi_n = input_to_lut_multiplier[g_new];
            const int32_t si_n = input_to_lut_shift     [g_new];
            const int32_t mr_n = recurrent_to_lut_multiplier[g_new];
            const int32_t sr_n = recurrent_to_lut_shift     [g_new];

            int8_t n_act[NumHidden_];
            for (std::size_t hi = 0; hi < NumHidden_; ++hi)
            {
                int32_t acc_in = 0;
                const WeightType* w_row_i = w_in_n + hi * NumInputs_;
                for (std::size_t j = 0; j < NumInputs_; ++j)
                {
                    const int32_t x_minus_zp =
                        static_cast<int32_t>(x[j]) - x_zp;
                    acc_in += static_cast<int32_t>(w_row_i[j]) * x_minus_zp;
                }

                int32_t acc_rec = 0;
                const WeightType* w_row_r = w_rec_n + hi * NumHidden_;
                for (std::size_t k = 0; k < NumHidden_; ++k)
                {
                    const int32_t h_minus_zp =
                        static_cast<int32_t>(rh[k]) - h_zp;
                    acc_rec += static_cast<int32_t>(w_row_r[k]) * h_minus_zp;
                }

                const int32_t r_in  =
                    multiplyByQuantizedMultiplier(acc_in,  mi_n, si_n);
                const int32_t r_rec =
                    multiplyByQuantizedMultiplier(acc_rec, mr_n, sr_n);
                const int32_t pre_int32 =
                    r_in + r_rec + static_cast<int32_t>(bias_n[hi]);
                const int8_t pre_i8 = static_cast<int8_t>(
                    detail::qGruSaturateInt8(pre_int32));
                n_act[hi] = qApplyLUT(pre_i8, tanh_lut);
            }

            // h_new = (1 - z_t) * n_t + z_t * h_prev   in hidden int8 grid.
            //
            // (1 - z_t) in the sigmoid grid is just (-z_t) plus 256 offset
            // canceled by the LUT zero_point convention; centering by
            // -kQGruSigmoidZeroPoint (= +128) on the (q - z_t) form yields:
            //   one_minus_z_centered = 128 - z_t   (no zero_point shift)
            // (See QUANTIZATION.md Phase 12 derivation.)
            const int32_t h_lo = static_cast<int32_t>(output_qmin);
            const int32_t h_hi = static_cast<int32_t>(output_qmax);
            for (std::size_t hi = 0; hi < NumHidden_; ++hi)
            {
                const int32_t one_minus_z =
                    -kQGruSigmoidZeroPoint - static_cast<int32_t>(z_act[hi]);
                const int32_t n_centered =
                    static_cast<int32_t>(n_act[hi]) - kQGruTanhZeroPoint;
                const int32_t z_centered =
                    static_cast<int32_t>(z_act[hi]) - kQGruSigmoidZeroPoint;
                const int32_t h_minus_zp =
                    static_cast<int32_t>(h_state[hi]) - h_zp;

                const int32_t prod_a = one_minus_z * n_centered;
                const int32_t prod_b = z_centered  * h_minus_zp;

                const int32_t scaled_a = multiplyByQuantizedMultiplier(
                    prod_a, one_minus_z_times_n_multiplier,
                    one_minus_z_times_n_shift);
                const int32_t scaled_b = multiplyByQuantizedMultiplier(
                    prod_b, z_times_h_multiplier, z_times_h_shift);

                int32_t h_new = scaled_a + scaled_b + h_zp;
                if (h_new < h_lo) h_new = h_lo;
                if (h_new > h_hi) h_new = h_hi;
                h_state[hi] = static_cast<HiddenType>(h_new);
            }
        }
    };

} // namespace tinymind
