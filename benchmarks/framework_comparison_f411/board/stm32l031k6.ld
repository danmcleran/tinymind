/*
 * Copyright (c) 2026 Dan McLeran
 *
 * Linker script for the STM32L031K6: 32 KB flash, 8 KB SRAM.
 * The stack starts at the top of SRAM and grows down toward .bss; there is
 * no heap, so _sbrk-backed allocation is deliberately left without room.
 */

ENTRY(Reset_Handler)

MEMORY
{
    FLASH (rx)  : ORIGIN = 0x08000000, LENGTH = 32K
    RAM   (rwx) : ORIGIN = 0x20000000, LENGTH = 8K
}

_estack = ORIGIN(RAM) + LENGTH(RAM);

INCLUDE sections.ld
