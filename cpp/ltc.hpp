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
#include <cstddef>
#include <cstdint>

// Liquid Time-Constant (LTC) recurrent cell -- Hasani, Lechner, Amini, Rus, Grosu
// "Liquid Time-constant Networks" (AAAI 2021), the fused (semi-implicit Euler)
// ODE solver step.
//
// Written in the same scalar-templated style as pinn::PinnMlp: the cell's
// parameters live in a caller-owned flat array of scalar type S, and step<S> is
// templated on S. The SAME code therefore:
//   - runs plain inference          (S = double / float, or fixed-point QValue),
//   - yields input derivatives       (S = Dual<...>),
//   - yields the weight gradient      (S = MultiDual<double,N>  -> pinn::sgdStep,
//                                      S = RevVar               -> pinn::sgdStepReverse).
// No bespoke backprop: differentiability comes for free from the dual machinery,
// exactly like the PINN building blocks. The caller owns the time loop and the
// state buffer, matching the QLSTMCell / QGRUCell convention.
//
// Continuous-time dynamics (per neuron i):
//   dx_i/dt = -[ 1/tau_i + f_i(x, I) ] * x_i + f_i(x, I) * A_i
// with the synaptic activation f_i = Act( W_in[i,:] . I + W_rec[i,:] . x + b_i ).
// The fused solver advances one step of size dt in closed form (no inner
// iteration, denominator > 1 so it is unconditionally stable for dt, tau > 0):
//   f_i        = Act( W_in[i,:] . I + W_rec[i,:] . x + b_i )
//   x_i(t+dt)  = ( x_i + dt * f_i * A_i ) / ( 1 + dt * ( 1/tau_i + f_i ) )
// `unfolds` repeats the fused step within one input sample for a finer effective
// step (dt is the per-unfold size). The LTC reference uses Act = sigmoid.
//
// SCOPE: float/double inference + autodiff training are the primary regime
// (TINYMIND_ENABLE_FLOAT). Fixed-point QValue inference also compiles (sigmoid
// LUT + QValue divide). Reverse-mode (RevVar) training additionally needs the
// Constant<RevVar> lift, enabled by -DTINYMIND_LTC_REVERSE_TRAINING=1 (which
// pulls in the host-only revdual.hpp tape). Pure int8 affine (QUANT) is NOT a
// target for LTC -- use the CfC cell there.

#include "include/tinymind_platform.hpp"
#include "pinn.hpp"          // pinn::Constant, pinn::SigmoidActivation, DualScalarActivation
#include "dualmath.hpp"      // DualScalarMath (unused constants kept consistent)

#include <cstddef>

// Optional: reverse-mode (adjoint) training support. Defining this before
// including ltc.hpp pulls in the host-only RevVar tape and teaches
// pinn::Constant how to lift a literal into a RevVar leaf-free constant, so the
// fused-step constants (dt, 1) compose under reverse-over-... evaluation.
#if defined(TINYMIND_LTC_REVERSE_TRAINING) && TINYMIND_LTC_REVERSE_TRAINING
#include "revdual.hpp"
namespace tinymind {
namespace pinn {
    template<> struct Constant<RevVar>
    {
        static RevVar of(double d) { return RevVar::constant(d); }
    };
} // namespace pinn
} // namespace tinymind
#endif

namespace tinymind {
namespace ltc {

    /**
     * One LTC cell with NumInputs external inputs and NumState neurons (the
     * state vector IS the cell output; add a readout layer downstream).
     *
     * Caller-owned parameter array `p`, flat layout, all of scalar type S:
     *   [ W_in : NumState * NumInputs ]   row-major, W_in[i*NumInputs + j]
     *   [ W_rec: NumState * NumState  ]   row-major, W_rec[i*NumState + k]
     *   [ b    : NumState ]               gate bias
     *   [ tau  : NumState ]               per-neuron time constant (keep > 0)
     *   [ A    : NumState ]               per-neuron synaptic reversal target
     *
     * Act is a policy with `template<typename S> static S apply(const S&)`,
     * e.g. pinn::SigmoidActivation (LTC default) or pinn::TanhActivation.
     */
    template<std::size_t NumInputs, std::size_t NumState,
             typename Act = pinn::SigmoidActivation>
    struct LtcCell
    {
        static const std::size_t WInCount  = NumState * NumInputs;
        static const std::size_t WRecCount = NumState * NumState;
        static const std::size_t NumParams =
            WInCount + WRecCount + NumState /*b*/ + NumState /*tau*/ + NumState /*A*/;

        // Offsets into the flat parameter array.
        static const std::size_t OffWIn  = 0;
        static const std::size_t OffWRec = OffWIn  + WInCount;
        static const std::size_t OffB    = OffWRec + WRecCount;
        static const std::size_t OffTau  = OffB    + NumState;
        static const std::size_t OffA    = OffTau  + NumState;

        /**
         * Advance the state by one input sample using `unfolds` fused steps of
         * size dt. `state` and `stateOut` are length NumState and may alias
         * (stateOut is written only after each unfold completes). `input` is
         * length NumInputs.
         */
        template<typename S>
        static void step(const S* p, const S* input, const S* state, S* stateOut,
                         double dt, std::size_t unfolds = 1)
        {
            const S* Win  = p + OffWIn;
            const S* Wrec = p + OffWRec;
            const S* b    = p + OffB;
            const S* tau  = p + OffTau;
            const S* A    = p + OffA;

            const S one = pinn::Constant<S>::of(1.0);
            const S h   = pinn::Constant<S>::of(dt);

            S work[NumState];
            for (std::size_t i = 0; i < NumState; ++i) work[i] = state[i];

            for (std::size_t u = 0; u < unfolds; ++u)
            {
                S next[NumState];
                for (std::size_t i = 0; i < NumState; ++i)
                {
                    S z = b[i];
                    for (std::size_t j = 0; j < NumInputs; ++j)
                        z = z + Win[i * NumInputs + j] * input[j];
                    for (std::size_t k = 0; k < NumState; ++k)
                        z = z + Wrec[i * NumState + k] * work[k];

                    const S f   = Act::template apply<S>(z);
                    const S num = work[i] + h * f * A[i];
                    const S den = one + h * (one / tau[i] + f);
                    next[i] = num / den;
                }
                for (std::size_t i = 0; i < NumState; ++i) work[i] = next[i];
            }

            for (std::size_t i = 0; i < NumState; ++i) stateOut[i] = work[i];
        }
    };

} // namespace ltc
} // namespace tinymind
