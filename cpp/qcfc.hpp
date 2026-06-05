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
 * Quantized Closed-form Continuous-time (CfC) cell (single time step).
 * int8 deployable counterpart of cfc.hpp; solver-free sibling of the LTC cell.
 *
 *   x1   = tanh   ( W_bx x + W_bh h_{t-1} + b_b )      (backbone trunk)
 *   ff1  = tanh   ( W1 x1 + b1 )
 *   ff2  = tanh   ( W2 x1 + b2 )
 *   t    = sigmoid( (W_A x1) * ts + (W_B x1) + b_time ) (time-gate)
 *   h_t  = (1 - t) * ff1 + t * ff2                      (interpolation)
 *
 * This is the REGULAR-SAMPLING deployable form: the elapsed-time `ts` is a
 * compile/calibration-time constant folded into the time-gate-A requantizer
 * (time_a_to_lut_*) and the combined time bias b_time = q((b_A*ts + b_B)).
 * Irregular per-step ts is the float cfc.hpp's domain (ts is a runtime scalar
 * there). The backbone, ff1, ff2, and time-A/time-B MACs each carry their own
 * Requantizer into the shared LUT input scale; the final interpolation reuses
 * the QGRU "(1 - t) == 128 - t in the sigmoid grid" identity.
 *
 * Sigmoid LUT output convention: scale 1/256, zero_point -128
 * Tanh    LUT output convention: scale 1/128, zero_point  0
 *
 * Pure integer at runtime; freestanding-safe under FLOAT=0 / STD=0.
 */

namespace tinymind {

    static constexpr int32_t kQCfcSigmoidZeroPoint = -128;
    static constexpr int32_t kQCfcTanhZeroPoint    = 0;

    namespace detail {

        inline int32_t qCfcSaturateInt8(int32_t v)
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
        typename HiddenStorage_,
        std::size_t NumInputs_,
        std::size_t NumHidden_,
        std::size_t BackboneDim_>
    struct QCfCCell
    {
        typedef InputStorage_  InputType;
        typedef WeightStorage_ WeightType;
        typedef AccumStorage_  AccumType;
        typedef HiddenStorage_ HiddenType;

        static constexpr std::size_t NumInputs   = NumInputs_;
        static constexpr std::size_t NumHidden   = NumHidden_;
        static constexpr std::size_t BackboneDim = BackboneDim_;

        // Backbone weights.
        const WeightType* w_backbone_input;   // [BackboneDim][NumInputs]
        const WeightType* w_backbone_hidden;  // [BackboneDim][NumHidden]
        const AccumType*  b_backbone;          // [BackboneDim], int32 (LUT in-scale)

        // Head + time-gate weights (all over the backbone trunk x1).
        const WeightType* w_ff1;   // [NumHidden][BackboneDim]
        const WeightType* w_ff2;   // [NumHidden][BackboneDim]
        const WeightType* w_time_a;// [NumHidden][BackboneDim]
        const WeightType* w_time_b;// [NumHidden][BackboneDim]
        const AccumType*  b_ff1;   // [NumHidden]
        const AccumType*  b_ff2;   // [NumHidden]
        const AccumType*  b_time;  // [NumHidden], combined q(b_A*ts + b_B)

        InputType  input_zero_point;
        HiddenType hidden_zero_point;

        // Backbone: input-MAC and hidden-MAC into the tanh LUT input scale.
        int32_t backbone_input_multiplier;
        int32_t backbone_input_shift;
        int32_t backbone_hidden_multiplier;
        int32_t backbone_hidden_shift;

        // Heads / time-gate: single MAC over x1 (tanh grid) into LUT in-scale.
        int32_t ff1_multiplier;     int32_t ff1_shift;
        int32_t ff2_multiplier;     int32_t ff2_shift;
        int32_t time_a_multiplier;  int32_t time_a_shift;   // ts folded in
        int32_t time_b_multiplier;  int32_t time_b_shift;

        const int8_t* sigmoid_lut;
        const int8_t* tanh_lut;

        // (1 - t) * ff1 -> hidden scale ;  t * ff2 -> hidden scale.
        int32_t one_minus_t_times_ff1_multiplier;
        int32_t one_minus_t_times_ff1_shift;
        int32_t t_times_ff2_multiplier;
        int32_t t_times_ff2_shift;

        HiddenType output_qmin;
        HiddenType output_qmax;

        void forward(const InputType* x, HiddenType* h_state) const
        {
            const int32_t x_zp = static_cast<int32_t>(input_zero_point);
            const int32_t h_zp = static_cast<int32_t>(hidden_zero_point);

            // --- backbone trunk x1 (tanh grid, zero_point 0) ---------------
            int8_t x1[BackboneDim_];
            for (std::size_t u = 0; u < BackboneDim_; ++u)
            {
                int32_t acc_in = 0;
                const WeightType* w_in_row = w_backbone_input + u * NumInputs_;
                for (std::size_t j = 0; j < NumInputs_; ++j)
                {
                    const int32_t x_minus_zp = static_cast<int32_t>(x[j]) - x_zp;
                    acc_in += static_cast<int32_t>(w_in_row[j]) * x_minus_zp;
                }

                int32_t acc_h = 0;
                const WeightType* w_h_row = w_backbone_hidden + u * NumHidden_;
                for (std::size_t k = 0; k < NumHidden_; ++k)
                {
                    const int32_t h_minus_zp =
                        static_cast<int32_t>(h_state[k]) - h_zp;
                    acc_h += static_cast<int32_t>(w_h_row[k]) * h_minus_zp;
                }

                const int32_t r_in = multiplyByQuantizedMultiplier(
                    acc_in, backbone_input_multiplier, backbone_input_shift);
                const int32_t r_h = multiplyByQuantizedMultiplier(
                    acc_h, backbone_hidden_multiplier, backbone_hidden_shift);
                const int32_t pre = r_in + r_h + static_cast<int32_t>(b_backbone[u]);
                const int8_t pre_i8 =
                    static_cast<int8_t>(detail::qCfcSaturateInt8(pre));
                x1[u] = qApplyLUT(pre_i8, tanh_lut);
            }

            // --- heads + time-gate over x1 ---------------------------------
            int8_t ff1_act[NumHidden_];
            int8_t ff2_act[NumHidden_];
            int8_t t_act  [NumHidden_];
            for (std::size_t i = 0; i < NumHidden_; ++i)
            {
                int32_t acc1 = 0, acc2 = 0, accA = 0, accB = 0;
                const WeightType* r1 = w_ff1    + i * BackboneDim_;
                const WeightType* r2 = w_ff2    + i * BackboneDim_;
                const WeightType* rA = w_time_a + i * BackboneDim_;
                const WeightType* rB = w_time_b + i * BackboneDim_;
                for (std::size_t u = 0; u < BackboneDim_; ++u)
                {
                    // x1 is in the tanh grid (zero_point 0): no centering.
                    const int32_t xv = static_cast<int32_t>(x1[u]);
                    acc1 += static_cast<int32_t>(r1[u]) * xv;
                    acc2 += static_cast<int32_t>(r2[u]) * xv;
                    accA += static_cast<int32_t>(rA[u]) * xv;
                    accB += static_cast<int32_t>(rB[u]) * xv;
                }

                const int32_t p1 = multiplyByQuantizedMultiplier(
                    acc1, ff1_multiplier, ff1_shift) + static_cast<int32_t>(b_ff1[i]);
                ff1_act[i] = qApplyLUT(
                    static_cast<int8_t>(detail::qCfcSaturateInt8(p1)), tanh_lut);

                const int32_t p2 = multiplyByQuantizedMultiplier(
                    acc2, ff2_multiplier, ff2_shift) + static_cast<int32_t>(b_ff2[i]);
                ff2_act[i] = qApplyLUT(
                    static_cast<int8_t>(detail::qCfcSaturateInt8(p2)), tanh_lut);

                // t = sigmoid( (W_A x1)*ts + (W_B x1) + b_time ); ts is folded
                // into time_a_multiplier and into b_time at calibration time.
                const int32_t pA = multiplyByQuantizedMultiplier(
                    accA, time_a_multiplier, time_a_shift);
                const int32_t pB = multiplyByQuantizedMultiplier(
                    accB, time_b_multiplier, time_b_shift);
                const int32_t pt = pA + pB + static_cast<int32_t>(b_time[i]);
                t_act[i] = qApplyLUT(
                    static_cast<int8_t>(detail::qCfcSaturateInt8(pt)), sigmoid_lut);
            }

            // --- interpolation  h = (1 - t) * ff1 + t * ff2 ----------------
            const int32_t h_lo = static_cast<int32_t>(output_qmin);
            const int32_t h_hi = static_cast<int32_t>(output_qmax);
            for (std::size_t i = 0; i < NumHidden_; ++i)
            {
                const int32_t one_minus_t =
                    -kQCfcSigmoidZeroPoint - static_cast<int32_t>(t_act[i]);
                const int32_t t_centered =
                    static_cast<int32_t>(t_act[i]) - kQCfcSigmoidZeroPoint;
                const int32_t ff1_centered =
                    static_cast<int32_t>(ff1_act[i]) - kQCfcTanhZeroPoint;
                const int32_t ff2_centered =
                    static_cast<int32_t>(ff2_act[i]) - kQCfcTanhZeroPoint;

                const int32_t prod_a = one_minus_t * ff1_centered;
                const int32_t prod_b = t_centered  * ff2_centered;

                const int32_t scaled_a = multiplyByQuantizedMultiplier(
                    prod_a, one_minus_t_times_ff1_multiplier,
                    one_minus_t_times_ff1_shift);
                const int32_t scaled_b = multiplyByQuantizedMultiplier(
                    prod_b, t_times_ff2_multiplier, t_times_ff2_shift);

                int32_t h_new = scaled_a + scaled_b + h_zp;
                if (h_new < h_lo) h_new = h_lo;
                if (h_new > h_hi) h_new = h_hi;
                h_state[i] = static_cast<HiddenType>(h_new);
            }
        }
    };

} // namespace tinymind
