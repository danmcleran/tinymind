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

// Bare-metal startup for QEMU mps2-an386 (Cortex-M4), plus the two Arm
// semihosting calls the test needs: print a string and exit with a status.
// No newlib: the test links -nostdlib, so memcpy/memset are supplied here
// for any calls the compiler emits on its own.

#include "semihost.hpp"

#include <cstddef>
#include <cstdint>

extern "C"
{
    extern uint32_t _stack_top;
    extern uint32_t __data_load;
    extern uint32_t __data_start;
    extern uint32_t __data_end;
    extern uint32_t __bss_start;
    extern uint32_t __bss_end;
    extern void (*__init_array_start[])();
    extern void (*__init_array_end[])();

    // The test entry point. Not main(): under -ffreestanding Clang treats
    // main as an ordinary C++ function and mangles it.
    int run_tests();

    void* memcpy(void* dst, const void* src, std::size_t n)
    {
        unsigned char* d = static_cast<unsigned char*>(dst);
        const unsigned char* s = static_cast<const unsigned char*>(src);
        while (n--)
        {
            *d++ = *s++;
        }
        return dst;
    }

    void* memset(void* dst, int c, std::size_t n)
    {
        unsigned char* d = static_cast<unsigned char*>(dst);
        while (n--)
        {
            *d++ = static_cast<unsigned char>(c);
        }
        return dst;
    }

    [[noreturn]] void Reset_Handler()
    {
        const uint32_t* src = &__data_load;
        for (uint32_t* dst = &__data_start; dst < &__data_end; )
        {
            *dst++ = *src++;
        }
        for (uint32_t* dst = &__bss_start; dst < &__bss_end; )
        {
            *dst++ = 0;
        }
        for (void (**ctor)() = __init_array_start; ctor < __init_array_end; ++ctor)
        {
            (*ctor)();
        }

        semihost::exit(run_tests());
    }

    [[noreturn]] void Fault_Handler()
    {
        semihost::write("FAULT: core took an exception\n");
        semihost::exit(2);
    }

    // Initial SP, reset, then NMI / HardFault / MemManage / BusFault /
    // UsageFault. A fault (e.g. an unaligned access the core refused)
    // reports and fails rather than hanging the emulator.
    __attribute__((section(".isr_vector"), used))
    const void* const g_vectors[] =
    {
        &_stack_top,
        reinterpret_cast<const void*>(&Reset_Handler),
        reinterpret_cast<const void*>(&Fault_Handler),
        reinterpret_cast<const void*>(&Fault_Handler),
        reinterpret_cast<const void*>(&Fault_Handler),
        reinterpret_cast<const void*>(&Fault_Handler),
        reinterpret_cast<const void*>(&Fault_Handler),
    };
}
