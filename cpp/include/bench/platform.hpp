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

#include <cstdint>
#include <cstddef>

#if !defined(TINYMIND_BENCH_CORTEX_M)
#if __has_include(<chrono>)
#include <chrono>
#define TINYMIND_BENCH_HAVE_CHRONO 1
#endif
#endif

// Platform selection for cycle counters and stack watermarking.
//
// Define TINYMIND_BENCH_CORTEX_M to enable DWT->CYCCNT reads on
// ARM Cortex-M3/M4/M7/M33/M55 targets. Otherwise the host implementation
// uses std::chrono::steady_clock and reports nanoseconds as "cycles".
//
// Both implementations expose the same API so benchmark code is portable:
//   bench::Cycles c = bench::readCycleCounter();
//   ... work ...
//   bench::Cycles elapsed = bench::readCycleCounter() - c;
//
// Stack watermarking:
//   bench::paintStack(buffer, size);          // call once before work
//   size_t used = bench::stackHighWater(buffer, size);

namespace tinymind {
namespace bench {

typedef uint32_t Cycles;

#if defined(TINYMIND_BENCH_CORTEX_M)

// Cortex-M DWT cycle counter. Must be enabled once at startup via
// enableCycleCounter(). DWT is present on M3/M4/M7/M33/M55; M0/M0+ do
// not have it and callers should fall back to SysTick.
struct Dwt
{
    volatile uint32_t ctrl;
    volatile uint32_t cyccnt;
};

struct Demcr
{
    volatile uint32_t value;
};

inline void enableCycleCounter()
{
    // DEMCR at 0xE000EDFC, TRCENA bit 24
    volatile uint32_t* demcr = reinterpret_cast<volatile uint32_t*>(0xE000EDFCu);
    *demcr |= (1u << 24);
    // DWT_CTRL at 0xE0001000
    volatile uint32_t* dwtCtrl = reinterpret_cast<volatile uint32_t*>(0xE0001000u);
    volatile uint32_t* dwtCyccnt = reinterpret_cast<volatile uint32_t*>(0xE0001004u);
    *dwtCyccnt = 0;
    *dwtCtrl |= 1u; // CYCCNTENA
}

inline Cycles readCycleCounter()
{
    volatile uint32_t* dwtCyccnt = reinterpret_cast<volatile uint32_t*>(0xE0001004u);
    return *dwtCyccnt;
}

#else // host

#if defined(TINYMIND_BENCH_HAVE_CHRONO)

inline void enableCycleCounter() {}

inline Cycles readCycleCounter()
{
    // Host fallback: report elapsed nanoseconds since first call.
    // Truncated to 32 bits; wraps every ~4.29 seconds, which is fine
    // for the short inference windows the harness targets.
    //
    // Uses .time_since_epoch().count() to avoid time_point subtraction,
    // which triggers ADL against the unconstrained tinymind::operator-
    // template and yields an ambiguous overload.
    using clock = ::std::chrono::steady_clock;
    const long long now = ::std::chrono::duration_cast< ::std::chrono::nanoseconds>(
                              clock::now().time_since_epoch()).count();
    static const long long base = now;
    return static_cast<Cycles>(static_cast<uint64_t>(now - base) & 0xFFFFFFFFu);
}

#else

inline void enableCycleCounter() {}
inline Cycles readCycleCounter() { return 0; }

#endif

#endif

// Fill a user-supplied stack-resident buffer with a recognizable pattern
// before the measured work runs. stackHighWater() then walks the buffer
// and reports how many bytes were touched.
//
// Typical usage in an MCU example:
//   static uint8_t sScratch[4096];
//   bench::paintStack(sScratch, sizeof(sScratch));
//   runInference();
//   size_t used = bench::stackHighWater(sScratch, sizeof(sScratch));
//
// The buffer should be placed where worst-case activation tensors live
// (e.g., passed as the scratch region into layer forward() calls).
inline void paintStack(void* buffer, size_t sizeBytes)
{
    uint8_t* p = static_cast<uint8_t*>(buffer);
    for (size_t i = 0; i < sizeBytes; ++i)
    {
        p[i] = 0xA5u;
    }
}

inline size_t stackHighWater(const void* buffer, size_t sizeBytes)
{
    const uint8_t* p = static_cast<const uint8_t*>(buffer);
    size_t firstTouched = sizeBytes;
    for (size_t i = 0; i < sizeBytes; ++i)
    {
        if (p[i] != 0xA5u)
        {
            firstTouched = i;
            break;
        }
    }
    return sizeBytes - firstTouched;
}

} // namespace bench
} // namespace tinymind
