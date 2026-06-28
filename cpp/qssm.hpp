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

#include <cstddef>
#include <cstdint>

/*
 * Quantized diagonal state-space layers (linear-recurrent / S4-lite).
 *
 * A diagonal state-space model runs NumChannels independent first-order
 * recurrences -- one scalar IIR per channel, no cross-channel mixing in the
 * recurrence. Per channel c, with input x_t and state s_t:
 *
 *   QStateSpace1D (linear time-invariant):
 *     s_t[c] = a[c] * s_{t-1}[c] + b[c] * x_t[c]
 *     y_t[c] = c[c] * s_t[c]    + d[c] * x_t[c]     (d optional skip)
 *
 *   QSelectiveStateSpace1D (input-gated -- "selective"):
 *     g_t[c] = clamp(wg[c] * x_t[c] + bg[c], 0, 1)  (hard sigmoid, per channel)
 *     s_t[c] = a[c] * s_{t-1}[c] + g_t[c] * b[c] * x_t[c]
 *     y_t[c] = c[c] * s_t[c]     + d[c] * x_t[c]
 *
 * The selective variant makes the input drive content-dependent (the gate is a
 * function of the input), which is the selectivity a Mamba-style model adds on
 * top of a fixed linear recurrence -- here as a cheap per-channel hard-sigmoid
 * gate rather than a full input-dependent transition, so the layer stays
 * LUT-free and freestanding.
 *
 * The decode state is a fixed NumChannels-wide int32 vector (QSSMState),
 * CONSTANT in the sequence length, so streaming inference is O(NumChannels)
 * memory and work per token -- the same property that makes linear attention
 * and these recurrences attractive for always-on sensor / audio streams.
 *
 * Two entry points sharing one token kernel:
 *   step()    -- advance one timestep, emit its output row (the streaming
 *                primitive: O(C) work, O(1) extra memory).
 *   forward() -- reset the state and run step() across a [T x C] block,
 *                byte-identical to T successive step() calls.
 *
 * Coefficients are per-channel integer (multiplier, shift) pairs built host-side
 * (buildQSSMParams / buildQSelectiveGateParams in qcalibration.hpp): a[c] is the
 * dimensionless decay, b/c/d fold the scale ratios. Stability requires |a[c]| <
 * 1, which keeps the int32 state bounded. Pure integer at runtime;
 * freestanding-safe (no LUT, the gate is a clamped affine).
 */

namespace tinymind
{

    /**
     * Diagonal recurrent state for the state-space layers: one int32 channel
     * per recurrence, at the calibrated state scale. reset() clears it to
     * start a fresh sequence.
     */
    template<typename AccumStorage_, std::size_t NumChannels_>
    struct QSSMState
    {
        typedef AccumStorage_ AccumType;

        static constexpr std::size_t NumChannels = NumChannels_;

        AccumType s[NumChannels_];

        void reset()
        {
            for (std::size_t c = 0; c < NumChannels_; ++c)
            {
                s[c] = static_cast<AccumType>(0);
            }
        }

        static_assert(NumChannels_ > 0, "NumChannels must be > 0.");
    };

    // ----------------------------------------------------------------------
    // Linear time-invariant diagonal state space.
    // ----------------------------------------------------------------------
    template<
        typename InStorage_,
        typename AccumStorage_,
        typename OutStorage_,
        std::size_t SeqLength_,
        std::size_t NumChannels_>
    struct QStateSpace1D
    {
        typedef InStorage_    InputType;
        typedef AccumStorage_ AccumType;
        typedef OutStorage_   OutputType;

        typedef QSSMState<AccumStorage_, NumChannels_> State;

        static constexpr std::size_t SeqLength    = SeqLength_;
        static constexpr std::size_t NumChannels  = NumChannels_;
        static constexpr std::size_t InputSize    = SeqLength_ * NumChannels_;
        static constexpr std::size_t OutputSize   = SeqLength_ * NumChannels_;

        InputType  input_zero_point;
        OutputType output_zero_point;
        OutputType qmin;
        OutputType qmax;

        // Per-channel integer coefficients (caller-owned, length NumChannels):
        //   a -> decay (state -> state), b -> input drive (input -> state),
        //   c -> readout (state -> output), d -> skip (input -> output).
        // d_multiplier may be nullptr to disable the skip path.
        const int32_t* a_multiplier;
        const int32_t* a_shift;
        const int32_t* b_multiplier;
        const int32_t* b_shift;
        const int32_t* c_multiplier;
        const int32_t* c_shift;
        const int32_t* d_multiplier;
        const int32_t* d_shift;

        /**
         * Advance one timestep. x and y are NumChannels wide; state carries the
         * recurrence across calls and must be reset() before the first token.
         */
        void step(const InputType* x, State& state, OutputType* y) const
        {
            const int32_t in_zp  = static_cast<int32_t>(input_zero_point);
            const int32_t out_zp = static_cast<int32_t>(output_zero_point);
            const int32_t lo     = static_cast<int32_t>(qmin);
            const int32_t hi     = static_cast<int32_t>(qmax);

            for (std::size_t c = 0; c < NumChannels_; ++c)
            {
                const int32_t xv = static_cast<int32_t>(x[c]) - in_zp;

                const int32_t a_term = multiplyByQuantizedMultiplier(
                    static_cast<int32_t>(state.s[c]), a_multiplier[c], a_shift[c]);
                const int32_t b_term = multiplyByQuantizedMultiplier(
                    xv, b_multiplier[c], b_shift[c]);
                state.s[c] = static_cast<AccumType>(a_term + b_term);

                int32_t acc = multiplyByQuantizedMultiplier(
                    static_cast<int32_t>(state.s[c]), c_multiplier[c], c_shift[c]);
                if (d_multiplier != nullptr)
                {
                    acc += multiplyByQuantizedMultiplier(xv, d_multiplier[c], d_shift[c]);
                }

                int32_t yv = acc + out_zp;
                if (yv < lo) yv = lo;
                if (yv > hi) yv = hi;
                y[c] = static_cast<OutputType>(yv);
            }
        }

        /**
         * Full-sequence pass over a [SeqLength x NumChannels] block. Resets the
         * state and steps; byte-identical to SeqLength successive step() calls.
         */
        void forward(const InputType* seq, State& state, OutputType* out) const
        {
            state.reset();
            for (std::size_t t = 0; t < SeqLength_; ++t)
            {
                step(seq + t * NumChannels_, state, out + t * NumChannels_);
            }
        }

        static_assert(SeqLength_   > 0, "Sequence length must be > 0.");
        static_assert(NumChannels_ > 0, "NumChannels must be > 0.");
    };

    // ----------------------------------------------------------------------
    // Input-gated ("selective") diagonal state space.
    // ----------------------------------------------------------------------
    template<
        typename InStorage_,
        typename AccumStorage_,
        typename OutStorage_,
        std::size_t SeqLength_,
        std::size_t NumChannels_>
    struct QSelectiveStateSpace1D
    {
        typedef InStorage_    InputType;
        typedef AccumStorage_ AccumType;
        typedef OutStorage_   OutputType;

        typedef QSSMState<AccumStorage_, NumChannels_> State;

        static constexpr std::size_t SeqLength   = SeqLength_;
        static constexpr std::size_t NumChannels = NumChannels_;
        static constexpr std::size_t InputSize   = SeqLength_ * NumChannels_;
        static constexpr std::size_t OutputSize  = SeqLength_ * NumChannels_;

        // The per-channel hard-sigmoid gate is computed in Q15 then clamped to
        // [0, 1]; 32767 == 1.0.
        static constexpr int32_t kGateOne = 32767;
        static constexpr int32_t kGateFracBits = 15;

        InputType  input_zero_point;
        OutputType output_zero_point;
        OutputType qmin;
        OutputType qmax;

        const int32_t* a_multiplier;
        const int32_t* a_shift;
        const int32_t* b_multiplier;
        const int32_t* b_shift;
        const int32_t* c_multiplier;
        const int32_t* c_shift;
        const int32_t* d_multiplier;
        const int32_t* d_shift;

        // Per-channel gate: g_pre = mulByQuant(xv, gate_multiplier, gate_shift)
        //                          + gate_bias, then clamp to [0, kGateOne].
        // gate_* are in Q15 (so g_pre is the gate value in [0,1] fixed point).
        const int32_t* gate_multiplier;
        const int32_t* gate_shift;
        const int32_t* gate_bias;

        void step(const InputType* x, State& state, OutputType* y) const
        {
            const int32_t in_zp  = static_cast<int32_t>(input_zero_point);
            const int32_t out_zp = static_cast<int32_t>(output_zero_point);
            const int32_t lo     = static_cast<int32_t>(qmin);
            const int32_t hi     = static_cast<int32_t>(qmax);

            for (std::size_t c = 0; c < NumChannels_; ++c)
            {
                const int32_t xv = static_cast<int32_t>(x[c]) - in_zp;

                const int32_t a_term = multiplyByQuantizedMultiplier(
                    static_cast<int32_t>(state.s[c]), a_multiplier[c], a_shift[c]);
                const int32_t b_term = multiplyByQuantizedMultiplier(
                    xv, b_multiplier[c], b_shift[c]);

                // Per-channel hard-sigmoid input gate (Q15, clamped to [0,1]).
                int32_t g = multiplyByQuantizedMultiplier(
                    xv, gate_multiplier[c], gate_shift[c]) + gate_bias[c];
                if (g < 0)        g = 0;
                if (g > kGateOne) g = kGateOne;

                // Gate the input drive: (b_term * g) >> 15, rounded.
                const int64_t gated =
                    (static_cast<int64_t>(b_term) * static_cast<int64_t>(g) +
                     (static_cast<int64_t>(1) << (kGateFracBits - 1)))
                    >> kGateFracBits;

                state.s[c] = static_cast<AccumType>(a_term + static_cast<int32_t>(gated));

                int32_t acc = multiplyByQuantizedMultiplier(
                    static_cast<int32_t>(state.s[c]), c_multiplier[c], c_shift[c]);
                if (d_multiplier != nullptr)
                {
                    acc += multiplyByQuantizedMultiplier(xv, d_multiplier[c], d_shift[c]);
                }

                int32_t yv = acc + out_zp;
                if (yv < lo) yv = lo;
                if (yv > hi) yv = hi;
                y[c] = static_cast<OutputType>(yv);
            }
        }

        void forward(const InputType* seq, State& state, OutputType* out) const
        {
            state.reset();
            for (std::size_t t = 0; t < SeqLength_; ++t)
            {
                step(seq + t * NumChannels_, state, out + t * NumChannels_);
            }
        }

        static_assert(SeqLength_   > 0, "Sequence length must be > 0.");
        static_assert(NumChannels_ > 0, "NumChannels must be > 0.");
    };

} // namespace tinymind
