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
// Bare-metal NUCLEO-L031K6 support: vector table, reset handler, 32 MHz
// clock tree, USART2 on the ST-LINK virtual COM port, a SysTick-based cycle
// counter and stack painting. No vendor HAL or CMSIS device headers; register
// addresses are from RM0377.
//
// Clock: HSI16 -> PLL (x4, /2) -> SYSCLK 32 MHz, the L031's maximum. 32 MHz
// needs voltage range 1 and one flash wait state; the prefetch buffer is on.
//
// Cortex-M0+ has no DWT cycle counter. SysTick counts core clocks down from
// 2^24 - 1; its wrap interrupt carries the upper bits, so board_cycles() is a
// free-running 32-bit cycle count like DWT CYCCNT on the F411.

#include "board.hpp"

extern "C"
{
extern uint32_t _sidata;
extern uint32_t _sdata;
extern uint32_t _edata;
extern uint32_t _sbss;
extern uint32_t _ebss;
extern uint32_t _estack;

void __libc_init_array();
void Reset_Handler();
void Default_Handler();
void SysTick_Handler();

// __libc_init_array calls _init; with -nostartfiles nothing else defines it.
void _init()
{
}
}

namespace
{

inline volatile uint32_t& reg(uint32_t address)
{
    return *reinterpret_cast<volatile uint32_t*>(address);
}

constexpr uint32_t RCC_CR      = 0x40021000u;
constexpr uint32_t RCC_CFGR    = 0x4002100Cu;
constexpr uint32_t RCC_IOPENR  = 0x4002102Cu;
constexpr uint32_t RCC_APB1ENR = 0x40021038u;
constexpr uint32_t FLASH_ACR   = 0x40022000u;
constexpr uint32_t PWR_CR      = 0x40007000u;
constexpr uint32_t PWR_CSR     = 0x40007004u;
constexpr uint32_t GPIOA_MODER = 0x50000000u;
constexpr uint32_t GPIOA_AFRL  = 0x50000020u;
constexpr uint32_t USART2_CR1  = 0x40004400u;
constexpr uint32_t USART2_BRR  = 0x4000440Cu;
constexpr uint32_t USART2_ISR  = 0x4000441Cu;
constexpr uint32_t USART2_TDR  = 0x40004428u;
constexpr uint32_t SYST_CSR    = 0xE000E010u;
constexpr uint32_t SYST_RVR    = 0xE000E014u;
constexpr uint32_t SYST_CVR    = 0xE000E018u;
constexpr uint32_t SCB_ICSR    = 0xE000ED04u;
constexpr uint32_t kPendSTSet  = 1u << 26;
constexpr uint32_t kSysTickMax = 0x00FFFFFFu;
constexpr uint32_t kPaint      = 0xA5A5A5A5u;

volatile uint32_t gSysTickWraps = 0;

void clockInit()
{
    // Voltage range 1 (1.8 V) is required above 16 MHz; wait for the regulator.
    reg(RCC_APB1ENR) |= (1u << 28);   // PWREN
    reg(PWR_CR) = (reg(PWR_CR) & ~(3u << 11)) | (1u << 11);
    while ((reg(PWR_CSR) & (1u << 4)) != 0u) // VOSF
    {
    }

    // One wait state + prefetch, set before raising the clock.
    reg(FLASH_ACR) |= 1u | (1u << 1);
    while ((reg(FLASH_ACR) & 1u) == 0u)
    {
    }

    reg(RCC_CR) |= 1u;                // HSI16ON
    while ((reg(RCC_CR) & (1u << 2)) == 0u)
    {
    }

    // PLL from HSI16: x4 = 64 MHz VCO, /2 = 32 MHz. AHB and APB undivided.
    reg(RCC_CFGR) = (reg(RCC_CFGR) & ~((1u << 16) | (0xFu << 18) | (3u << 22))) | (1u << 18) | (1u << 22);
    reg(RCC_CR) |= (1u << 24);
    while ((reg(RCC_CR) & (1u << 25)) == 0u)
    {
    }

    reg(RCC_CFGR) |= 3u;
    while (((reg(RCC_CFGR) >> 2) & 3u) != 3u)
    {
    }
}

void uartInit()
{
    reg(RCC_IOPENR) |= 1u;            // IOPAEN
    reg(RCC_APB1ENR) |= (1u << 17);   // USART2EN

    // PA2 = USART2_TX (alternate function 4), wired to the ST-LINK VCP.
    reg(GPIOA_MODER) = (reg(GPIOA_MODER) & ~(3u << 4)) | (2u << 4);
    reg(GPIOA_AFRL) = (reg(GPIOA_AFRL) & ~(0xFu << 8)) | (4u << 8);

    // 115200 baud from PCLK1 = 32 MHz, 16x oversampling.
    reg(USART2_BRR) = 278u;
    reg(USART2_CR1) = (1u << 3) | 1u; // TE, UE
}

void sysTickInit()
{
    reg(SYST_RVR) = kSysTickMax;
    reg(SYST_CVR) = 0u;
    reg(SYST_CSR) = 7u;               // CLKSOURCE = core clock, TICKINT, ENABLE
}

void putChar(char c)
{
    if (c == '\n')
    {
        putChar('\r');
    }
    while ((reg(USART2_ISR) & (1u << 7)) == 0u) // TXE
    {
    }
    reg(USART2_TDR) = static_cast<uint8_t>(c);
}

} // namespace

extern "C"
{

void board_init(void)
{
    clockInit();
    uartInit();
    sysTickInit();
}

uint32_t board_cycles(void)
{
    // With interrupts masked, a wrap that has happened but not yet been
    // counted shows as a pending SysTick; re-read the counter after seeing it
    // so the low bits are from after the reload too.
    __asm volatile ("cpsid i" ::: "memory");
    uint32_t high = gSysTickWraps;
    uint32_t low = reg(SYST_CVR);
    if (reg(SCB_ICSR) & kPendSTSet)
    {
        low = reg(SYST_CVR);
        ++high;
    }
    __asm volatile ("cpsie i" ::: "memory");
    return (high << 24) | (kSysTickMax - low);
}

void board_puts(const char* s)
{
    while (*s)
    {
        putChar(*s++);
    }
}

void board_put_u32(uint32_t v)
{
    char buffer[11];
    int i = 10;
    buffer[i] = 0;
    do
    {
        buffer[--i] = static_cast<char>('0' + v % 10u);
        v /= 10u;
    }
    while (v);
    board_puts(buffer + i);
}

void board_put_i32(int32_t v)
{
    if (v < 0)
    {
        putChar('-');
        board_put_u32(static_cast<uint32_t>(-(v + 1)) + 1u);
    }
    else
    {
        board_put_u32(static_cast<uint32_t>(v));
    }
}

void board_put_hex(uint32_t v)
{
    board_puts("0x");
    for (int s = 28; s >= 0; s -= 4)
    {
        putChar("0123456789ABCDEF"[(v >> s) & 0xFu]);
    }
}

void board_flush(void)
{
    while ((reg(USART2_ISR) & (1u << 6)) == 0u) // TC
    {
    }
}

__attribute__((noinline)) void board_stack_paint(void)
{
    uint32_t sp;
    __asm volatile ("mov %0, sp" : "=r"(sp));
    for (uint32_t* p = &_ebss; p < reinterpret_cast<uint32_t*>(sp - 64u); ++p)
    {
        *p = kPaint;
    }
}

uint32_t board_stack_used(void)
{
    const uint32_t* p = &_ebss;
    while (p < &_estack && *p == kPaint)
    {
        ++p;
    }
    return static_cast<uint32_t>(reinterpret_cast<uintptr_t>(&_estack) - reinterpret_cast<uintptr_t>(p));
}

uint32_t board_static_ram(void)
{
    return static_cast<uint32_t>(reinterpret_cast<uintptr_t>(&_ebss) - reinterpret_cast<uintptr_t>(&_sdata));
}

void SysTick_Handler()
{
    gSysTickWraps = gSysTickWraps + 1u;
}

void Reset_Handler()
{
    uint32_t* src = &_sidata;
    for (uint32_t* dst = &_sdata; dst < &_edata;)
    {
        *dst++ = *src++;
    }
    for (uint32_t* dst = &_sbss; dst < &_ebss;)
    {
        *dst++ = 0u;
    }

    __libc_init_array();

    // ISO C++ forbids calling main from the program, so branch to it the way
    // a C runtime's crt0 would.
    __asm volatile ("bl main" ::: "r0", "r1", "r2", "r3", "r12", "lr", "memory", "cc");

    for (;;)
    {
    }
}

void Default_Handler()
{
    board_puts("\n# FAULT\n");
    for (;;)
    {
    }
}

} // extern "C"

// The core exceptions only (ARMv6-M has no MemManage/BusFault/UsageFault);
// SysTick is the one interrupt enabled.
__attribute__((section(".isr_vector"), used))
void (* const gVectorTable[16])() =
{
    reinterpret_cast<void (*)()>(&_estack),
    Reset_Handler,
    Default_Handler, // NMI
    Default_Handler, // HardFault
    nullptr,
    nullptr,
    nullptr,
    nullptr,
    nullptr,
    nullptr,
    nullptr,
    Default_Handler, // SVCall
    nullptr,
    nullptr,
    Default_Handler, // PendSV
    SysTick_Handler,
};
