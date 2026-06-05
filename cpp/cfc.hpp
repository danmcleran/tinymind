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

// Closed-form Continuous-time (CfC) recurrent cell -- Hasani, Lechner, Amini,
// Liebenwein, Ray, Tschaikowski, Teschl, Rus, "Closed-form continuous-time
// neural networks" (Nature Machine Intelligence 2022). The solver-free sibling
// of the LTC cell (ltc.hpp): CfC approximates the LTC ODE rollout in closed
// form, so a single step has NO inner ODE iteration -- it is a fixed sequence
// of matmuls + tanh/sigmoid + a time-gated interpolation.
//
// Written scalar-templated in the same style as pinn::PinnMlp / ltc::LtcCell:
// step<S> differentiates for free (Dual / MultiDual / RevVar) and infers in
// double / float / fixed-point QValue. The caller owns the state buffer and the
// time loop, matching the QLSTMCell / QGRUCell convention.
//
// Forward (per step), with backbone trunk x1 over [input ++ h_prev]:
//   x1   = BackboneAct( W_bx . input + W_bh . h_prev + b_b )      (BackboneDim)
//   ff1  = HeadAct( W1 . x1 + b1 )                                (NumState)
//   ff2  = HeadAct( W2 . x1 + b2 )                                (NumState)
//   tA   = W_A . x1 + b_A                                         (NumState, linear)
//   tB   = W_B . x1 + b_B                                         (NumState, linear)
//   t    = Gate( tA * ts + tB )                                   (time-gate)
//   h'   = (1 - t) * ff1 + t * ff2                                (interpolate)
// `ts` is the elapsed-time since the previous sample. CfC's headline feature is
// irregular sampling: ts varies per step and feeds the time-gate. With ts held
// constant the cell degenerates to a backbone-gated RNN. The int8 deployable
// counterpart is qcfc.hpp (regular-sampling form; ts folded at calibration).

#include "include/tinymind_platform.hpp"
#include "pinn.hpp"          // pinn::Constant, pinn::TanhActivation/SigmoidActivation, DualScalarActivation
#include "dualmath.hpp"

#include <cstddef>

#if defined(TINYMIND_CFC_REVERSE_TRAINING) && TINYMIND_CFC_REVERSE_TRAINING
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
namespace cfc {

    /**
     * One CfC cell: NumInputs external inputs, NumState neurons (the state
     * vector IS the cell output), a backbone trunk of BackboneDim units.
     *
     * Caller-owned flat parameter array `p`, all of scalar type S, layout:
     *   [ W_bx : BackboneDim * NumInputs ]   backbone, input  half
     *   [ W_bh : BackboneDim * NumState  ]   backbone, hidden half
     *   [ b_b  : BackboneDim ]
     *   [ W1   : NumState * BackboneDim ][ b1 : NumState ]   ff1 head (tanh)
     *   [ W2   : NumState * BackboneDim ][ b2 : NumState ]   ff2 head (tanh)
     *   [ W_A  : NumState * BackboneDim ][ b_A: NumState ]   time-gate A (linear)
     *   [ W_B  : NumState * BackboneDim ][ b_B: NumState ]   time-gate B (linear)
     *
     * Each activation policy has `template<typename S> static S apply(const S&)`.
     */
    template<std::size_t NumInputs, std::size_t NumState, std::size_t BackboneDim,
             typename BackboneAct = pinn::TanhActivation,
             typename HeadAct     = pinn::TanhActivation,
             typename GateAct     = pinn::SigmoidActivation>
    struct CfCCell
    {
        static const std::size_t WbxCount = BackboneDim * NumInputs;
        static const std::size_t WbhCount = BackboneDim * NumState;
        static const std::size_t HeadCount = NumState * BackboneDim;
        static const std::size_t NumParams =
            WbxCount + WbhCount + BackboneDim
            + 4 * (HeadCount + NumState);   // ff1, ff2, A, B

        static const std::size_t OffWbx = 0;
        static const std::size_t OffWbh = OffWbx + WbxCount;
        static const std::size_t OffBb  = OffWbh + WbhCount;
        static const std::size_t OffW1  = OffBb  + BackboneDim;
        static const std::size_t OffB1  = OffW1  + HeadCount;
        static const std::size_t OffW2  = OffB1  + NumState;
        static const std::size_t OffB2  = OffW2  + HeadCount;
        static const std::size_t OffWA  = OffB2  + NumState;
        static const std::size_t OffBA  = OffWA  + HeadCount;
        static const std::size_t OffWB  = OffBA  + NumState;
        static const std::size_t OffBB  = OffWB  + HeadCount;

        /**
         * Advance the state by one input sample with elapsed time `ts`.
         * `state` / `stateOut` are length NumState (may alias); `input` is
         * length NumInputs.
         */
        template<typename S>
        static void step(const S* p, const S* input, const S* state, S* stateOut,
                         double ts = 1.0)
        {
            const S* Wbx = p + OffWbx;
            const S* Wbh = p + OffWbh;
            const S* bb  = p + OffBb;
            const S* W1  = p + OffW1;  const S* b1 = p + OffB1;
            const S* W2  = p + OffW2;  const S* b2 = p + OffB2;
            const S* WA  = p + OffWA;  const S* bA = p + OffBA;
            const S* WB  = p + OffWB;  const S* bB = p + OffBB;

            const S one  = pinn::Constant<S>::of(1.0);
            const S tsS  = pinn::Constant<S>::of(ts);

            // Backbone trunk over [input ++ h_prev].
            S x1[BackboneDim];
            for (std::size_t u = 0; u < BackboneDim; ++u)
            {
                S z = bb[u];
                for (std::size_t j = 0; j < NumInputs; ++j)
                    z = z + Wbx[u * NumInputs + j] * input[j];
                for (std::size_t k = 0; k < NumState; ++k)
                    z = z + Wbh[u * NumState + k] * state[k];
                x1[u] = BackboneAct::template apply<S>(z);
            }

            // Heads + time-gated interpolation.
            for (std::size_t i = 0; i < NumState; ++i)
            {
                S a1 = b1[i], a2 = b2[i], aA = bA[i], aB = bB[i];
                const S* r1 = W1 + i * BackboneDim;
                const S* r2 = W2 + i * BackboneDim;
                const S* rA = WA + i * BackboneDim;
                const S* rB = WB + i * BackboneDim;
                for (std::size_t u = 0; u < BackboneDim; ++u)
                {
                    a1 = a1 + r1[u] * x1[u];
                    a2 = a2 + r2[u] * x1[u];
                    aA = aA + rA[u] * x1[u];
                    aB = aB + rB[u] * x1[u];
                }

                const S ff1 = HeadAct::template apply<S>(a1);
                const S ff2 = HeadAct::template apply<S>(a2);
                const S t   = GateAct::template apply<S>(aA * tsS + aB);

                stateOut[i] = (one - t) * ff1 + t * ff2;
            }
        }
    };

} // namespace cfc
} // namespace tinymind
