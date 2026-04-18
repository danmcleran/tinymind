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
#include "bench/platform.hpp"

// Compile-time size reporting for TinyMind pipelines.
//
// Each layer row contributes four numbers reported in CSV:
//   name, weight_bytes, activation_bytes, cycles
//
// Weight bytes come from sizeof(Layer) minus any known mutable state
// (e.g., argmax indices in MaxPool1D/2D). Activation bytes must be
// supplied by the caller because output buffers are external to the
// layer objects in TinyMind's design. This keeps the harness simple
// and matches the way Conv1D / Conv2D are actually composed.

namespace tinymind {
namespace bench {

struct LayerStat
{
    const char* name;
    size_t weightBytes;
    size_t activationBytes;
    Cycles cycles;
};

// Minimal printer that writes a CSV row to any sink with an
// operator<< for const char* and size_t/uint32_t. std::ostream
// satisfies this; on an MCU, a lightweight UART wrapper that
// implements those overloads can be used instead without
// dragging in <iostream>.
template<typename Sink>
void writeHeader(Sink& sink)
{
    sink << "name,weight_bytes,activation_bytes,cycles\n";
}

template<typename Sink>
void writeRow(Sink& sink, const LayerStat& row)
{
    sink << row.name << ","
         << row.weightBytes << ","
         << row.activationBytes << ","
         << row.cycles << "\n";
}

// Scoped measurement helper. Record cycles between construction
// and readElapsed(). Used inside the per-layer timing blocks in
// the benchmark runners.
struct ScopedTimer
{
    Cycles start;

    ScopedTimer() : start(readCycleCounter()) {}

    Cycles readElapsed() const
    {
        return readCycleCounter() - start;
    }
};

} // namespace bench
} // namespace tinymind
