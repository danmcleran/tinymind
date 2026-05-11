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

#include <cstddef>

/*
 * Quantized 2D constant padding (NHWC).
 *
 * Required to express SAME-padding convolutions on top of QConv2D (which
 * is VALID-only). The pad_value field carries the affine "real zero":
 * for an int8 affine input, that is the input's zero_point. Padding with
 * raw 0 instead would inject -scale * zp_input bias into every receptive
 * field that touched the border, which is a common int8 deployment bug.
 *
 * Tensor layout:
 *
 *   input    NHWC   [H_IN][W_IN][C]
 *   output   NHWC   [H_IN + PAD_T + PAD_B][W_IN + PAD_L + PAD_R][C]
 *
 * Pure integer at runtime; freestanding-safe. No <cstring>; the inner
 * loops use scalar writes so the same code compiles on all targets.
 */

namespace tinymind {

    template<
        typename Storage_,
        std::size_t H_IN_,
        std::size_t W_IN_,
        std::size_t C_,
        std::size_t PadTop_,
        std::size_t PadBottom_,
        std::size_t PadLeft_,
        std::size_t PadRight_>
    struct QPad2D
    {
        typedef Storage_ StorageType;

        static constexpr std::size_t InputHeight    = H_IN_;
        static constexpr std::size_t InputWidth     = W_IN_;
        static constexpr std::size_t Channels       = C_;
        static constexpr std::size_t PadTop         = PadTop_;
        static constexpr std::size_t PadBottom      = PadBottom_;
        static constexpr std::size_t PadLeft        = PadLeft_;
        static constexpr std::size_t PadRight       = PadRight_;

        static constexpr std::size_t OutputHeight   = H_IN_ + PadTop_ + PadBottom_;
        static constexpr std::size_t OutputWidth    = W_IN_ + PadLeft_ + PadRight_;
        static constexpr std::size_t OutputChannels = C_;
        static constexpr std::size_t InputSize      = H_IN_ * W_IN_ * C_;
        static constexpr std::size_t OutputSize     = OutputHeight * OutputWidth * C_;

        static_assert(H_IN_ > 0 && W_IN_ > 0, "Input spatial dims must be > 0.");
        static_assert(C_ > 0, "Channels must be > 0.");

        StorageType pad_value;

        void forward(const StorageType* in, StorageType* out) const
        {
            for (std::size_t oh = 0; oh < OutputHeight; ++oh)
            {
                const bool inside_h = (oh >= PadTop_) && (oh < PadTop_ + H_IN_);
                for (std::size_t ow = 0; ow < OutputWidth; ++ow)
                {
                    const bool inside_w = (ow >= PadLeft_) && (ow < PadLeft_ + W_IN_);
                    const std::size_t outPixelOffset =
                        (oh * OutputWidth + ow) * C_;

                    if (inside_h && inside_w)
                    {
                        const std::size_t ih = oh - PadTop_;
                        const std::size_t iw = ow - PadLeft_;
                        const std::size_t inPixelOffset = (ih * W_IN_ + iw) * C_;
                        for (std::size_t c = 0; c < C_; ++c)
                        {
                            out[outPixelOffset + c] = in[inPixelOffset + c];
                        }
                    }
                    else
                    {
                        for (std::size_t c = 0; c < C_; ++c)
                        {
                            out[outPixelOffset + c] = pad_value;
                        }
                    }
                }
            }
        }
    };

} // namespace tinymind
