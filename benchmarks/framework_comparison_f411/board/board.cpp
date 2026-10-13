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

// Bare-metal NUCLEO-F411RE support: vector table, reset handler, 100 MHz
// clock tree, USART2 on the ST-LINK virtual COM port, DWT cycle counter and
// stack painting. No vendor HAL or CMSIS device headers; register addresses
// are from RM0383.
//
// Clock: HSI 16 MHz -> PLL (M=16, N=400, P=4) -> SYSCLK 100 MHz, the F411's
// maximum. APB1 = 50 MHz (its limit). 100 MHz needs voltage scale 1 and three
// flash wait states; the ART accelerator (prefetch + I/D caches) is enabled.

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

constexpr uint32_t RCC_CR      = 0x40023800u;
constexpr uint32_t RCC_PLLCFGR = 0x40023804u;
constexpr uint32_t RCC_CFGR    = 0x40023808u;
constexpr uint32_t RCC_AHB1ENR = 0x40023830u;
constexpr uint32_t RCC_APB1ENR = 0x40023840u;
constexpr uint32_t FLASH_ACR   = 0x40023C00u;
constexpr uint32_t PWR_CR      = 0x40007000u;
constexpr uint32_t GPIOA_MODER = 0x40020000u;
constexpr uint32_t GPIOA_AFRL  = 0x40020020u;
constexpr uint32_t USART2_SR   = 0x40004400u;
constexpr uint32_t USART2_DR   = 0x40004404u;
constexpr uint32_t USART2_BRR  = 0x40004408u;
constexpr uint32_t USART2_CR1  = 0x4000440Cu;
constexpr uint32_t SCB_CPACR   = 0xE000ED88u;
constexpr uint32_t DEMCR       = 0xE000EDFCu;
constexpr uint32_t DWT_CTRL    = 0xE0001000u;
constexpr uint32_t DWT_CYCCNT  = 0xE0001004u;
constexpr uint32_t kPaint      = 0xA5A5A5A5u;

void clockInit()
{
    // Voltage scale 1 is required above 84 MHz.
    reg(RCC_APB1ENR) |= (1u << 28);
    reg(PWR_CR) = (reg(PWR_CR) & ~(3u << 14)) | (3u << 14);

    // Three wait states + prefetch + I-cache + D-cache, set before raising the clock.
    reg(FLASH_ACR) = 3u | (1u << 8) | (1u << 9) | (1u << 10);
    while ((reg(FLASH_ACR) & 0xFu) != 3u)
    {
    }

    // PLL from HSI: VCO in 1 MHz, VCO out 400 MHz, SYSCLK = 400 / 4.
    reg(RCC_PLLCFGR) = 16u | (400u << 6) | (1u << 16) | (0u << 22) | (8u << 24);
    reg(RCC_CR) |= (1u << 24);
    while ((reg(RCC_CR) & (1u << 25)) == 0u)
    {
    }

    // APB1 /2, then switch SYSCLK to the PLL.
    reg(RCC_CFGR) = (reg(RCC_CFGR) & ~(7u << 10)) | (4u << 10);
    reg(RCC_CFGR) = (reg(RCC_CFGR) & ~3u) | 2u;
    while (((reg(RCC_CFGR) >> 2) & 3u) != 2u)
    {
    }
}

void uartInit()
{
    reg(RCC_AHB1ENR) |= 1u;           // GPIOAEN
    reg(RCC_APB1ENR) |= (1u << 17);   // USART2EN

    // PA2 = USART2_TX, PA3 = USART2_RX: alternate function 7.
    reg(GPIOA_MODER) = (reg(GPIOA_MODER) & ~(0xFu << 4)) | (0xAu << 4);
    reg(GPIOA_AFRL) = (reg(GPIOA_AFRL) & ~(0xFFu << 8)) | (0x77u << 8);

    // 115200 baud from PCLK1 = 50 MHz: mantissa 27, fraction 2.
    reg(USART2_BRR) = (27u << 4) | 2u;
    reg(USART2_CR1) = (1u << 13) | (1u << 3); // UE, TE
}

void putChar(char c)
{
    if (c == '\n')
    {
        putChar('\r');
    }
    while ((reg(USART2_SR) & (1u << 7)) == 0u) // TXE
    {
    }
    reg(USART2_DR) = static_cast<uint8_t>(c);
}

} // namespace

extern "C"
{

void board_init(void)
{
    clockInit();
    uartInit();
    reg(DEMCR) |= (1u << 24);  // TRCENA
    reg(DWT_CYCCNT) = 0;
    reg(DWT_CTRL) |= 1u;       // CYCCNTENA
}

uint32_t board_cycles(void)
{
    return reg(DWT_CYCCNT);
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
    while ((reg(USART2_SR) & (1u << 6)) == 0u) // TC
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

void Reset_Handler()
{
    // Full access to CP10/CP11 before any floating-point instruction runs.
    reg(SCB_CPACR) |= (0xFu << 20);
    __asm volatile ("dsb\n\tisb" ::: "memory");

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

// The core exceptions only; no peripheral interrupts are enabled.
__attribute__((section(".isr_vector"), used))
void (* const gVectorTable[16])() =
{
    reinterpret_cast<void (*)()>(&_estack),
    Reset_Handler,
    Default_Handler, // NMI
    Default_Handler, // HardFault
    Default_Handler, // MemManage
    Default_Handler, // BusFault
    Default_Handler, // UsageFault
    nullptr,
    nullptr,
    nullptr,
    nullptr,
    Default_Handler, // SVCall
    Default_Handler, // DebugMonitor
    nullptr,
    Default_Handler, // PendSV
    Default_Handler, // SysTick
};
