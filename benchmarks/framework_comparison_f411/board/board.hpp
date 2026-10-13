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

// Minimal NUCLEO-F411RE support shared by every benchmark image.
#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C"
{
#endif

void     board_init(void);           // 100 MHz clock, USART2 115200, DWT on
uint32_t board_cycles(void);         // DWT CYCCNT
void     board_puts(const char* s);
void     board_put_u32(uint32_t v);
void     board_put_i32(int32_t v);
void     board_put_hex(uint32_t v);
void     board_flush(void);
void     board_stack_paint(void);    // fill unused stack with a pattern
uint32_t board_stack_used(void);     // deepest stack use since paint, bytes
uint32_t board_static_ram(void);     // .data + .bss bytes

#ifdef __cplusplus
}
#endif
