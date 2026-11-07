/*
 * Copyright (c) 2026 Dan McLeran
 *
 * Linker script for QEMU's mps3-an547 machine (Cortex-M55). Code and the
 * vector table live in the 512 KB ITCM at 0x00000000, where the core fetches
 * its initial SP and reset vector; data, bss and the stack live in the
 * 512 KB DTCM at 0x20000000. Neither TCM sits behind a memory protection
 * controller, so no MPC setup is needed. Only what the freestanding test
 * needs: no heap, no constructors beyond .init_array, no exception tables.
 */

ENTRY(Reset_Handler)

MEMORY
{
    CODE (rx)  : ORIGIN = 0x00000000, LENGTH = 512K
    RAM  (rwx) : ORIGIN = 0x20000000, LENGTH = 512K
}

_stack_top = ORIGIN(RAM) + LENGTH(RAM);

SECTIONS
{
    .text :
    {
        KEEP(*(.isr_vector))
        *(.text*)
        *(.rodata*)
        . = ALIGN(4);
        __init_array_start = .;
        KEEP(*(.init_array*))
        __init_array_end = .;
        . = ALIGN(4);
    } > CODE

    .data :
    {
        __data_start = .;
        *(.data*)
        . = ALIGN(4);
        __data_end = .;
    } > RAM AT > CODE
    __data_load = LOADADDR(.data);

    .bss (NOLOAD) :
    {
        __bss_start = .;
        *(.bss*)
        *(COMMON)
        . = ALIGN(4);
        __bss_end = .;
    } > RAM

    /DISCARD/ :
    {
        *(.ARM.exidx*)
        *(.ARM.extab*)
    }
}
