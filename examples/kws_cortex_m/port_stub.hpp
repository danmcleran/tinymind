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
*/

// Vendor-neutral port stub. Drop this into your MCU project and fill in
// the three functions for your HAL / SDK. Kept separate from
// kws_cortex_m.cpp so the model pipeline stays portable.

#pragma once

#include <cstddef>
#include <cstdint>

namespace kws_port {

// Pull one sample (INT16 PCM, mono) from the microphone front-end.
// Return false on underrun so the caller can decide whether to block.
bool readMicSample(int16_t& sample);

// Emit one character of a CSV / log line. Typically a UART TX.
void putChar(char c);

// Optional: called once at startup. Initialize the microphone,
// UART, DWT cycle counter, etc.
void platformInit();

} // namespace kws_port

// Example thin ostream-like sink so bench::writeRow can target UART
// without pulling in <iostream>. Implement operator<< for the types
// the harness prints.
struct UartSink
{
    UartSink& operator<<(const char* s)
    {
        while (*s) { kws_port::putChar(*s++); }
        return *this;
    }

    UartSink& operator<<(size_t v)
    {
        char buf[24];
        size_t n = 0;
        if (v == 0) { buf[n++] = '0'; }
        while (v)
        {
            buf[n++] = static_cast<char>('0' + (v % 10));
            v /= 10;
        }
        while (n) { kws_port::putChar(buf[--n]); }
        return *this;
    }

    UartSink& operator<<(uint32_t v)
    {
        return (*this) << static_cast<size_t>(v);
    }
};
