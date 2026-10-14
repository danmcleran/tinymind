/*
 * Copyright (c) 2026 Dan McLeran
 *
 * Linker script for the STM32F411RE: 512 KB flash, 128 KB SRAM.
 * The stack starts at the top of SRAM and grows down toward .bss; there is
 * no heap, so _sbrk-backed allocation is deliberately left without room.
 */

ENTRY(Reset_Handler)

MEMORY
{
    FLASH (rx)  : ORIGIN = 0x08000000, LENGTH = 512K
    RAM   (rwx) : ORIGIN = 0x20000000, LENGTH = 128K
}

_estack = ORIGIN(RAM) + LENGTH(RAM);

INCLUDE sections.ld
