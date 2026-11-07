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

// Arm semihosting (M-profile: BKPT 0xAB, operation in r0, argument in r1).
// QEMU services these when run with -semihosting-config enable=on.

#include <cstdint>

namespace semihost {

    inline uint32_t call(uint32_t op, const void* arg)
    {
        uint32_t result;
        __asm__ volatile("mov r0, %1\n\t"
                         "mov r1, %2\n\t"
                         "bkpt 0xab\n\t"
                         "mov %0, r0"
                         : "=r"(result)
                         : "r"(op), "r"(arg)
                         : "r0", "r1", "memory");
        return result;
    }

    // SYS_WRITE0: write a NUL-terminated string to the host console.
    inline void write(const char* s)
    {
        call(0x04u, s);
    }

    // SYS_EXIT with ADP_Stopped_ApplicationExit. On 32-bit Arm the call
    // carries no status word, so QEMU exits 0 for this reason code and 1
    // for any other; RunTimeErrorUnknown stands in for "failed".
    [[noreturn]] inline void exit(int status)
    {
        const uint32_t reason = (status == 0) ? 0x20026u : 0x20023u;
        call(0x18u, reinterpret_cast<const void*>(reason));
        for (;;)
        {
        }
    }

    inline void writeUnsigned(uint32_t v)
    {
        char buf[11];
        int i = 10;
        buf[i] = '\0';
        do
        {
            buf[--i] = static_cast<char>('0' + (v % 10u));
            v /= 10u;
        } while (v != 0u);
        write(buf + i);
    }

    inline void writeSigned(int32_t v)
    {
        if (v < 0)
        {
            write("-");
            writeUnsigned(0u - static_cast<uint32_t>(v));
        }
        else
        {
            writeUnsigned(static_cast<uint32_t>(v));
        }
    }

} // namespace semihost
