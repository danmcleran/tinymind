/*
 * Copyright (c) 2026 Dan McLeran
 *
 * Memory map with no practical limit, for `make fit`: links an image that
 * overflows its board so `size` can report what it would need. Never flashed.
 */

ENTRY(Reset_Handler)

MEMORY
{
    FLASH (rx)  : ORIGIN = 0x08000000, LENGTH = 16M
    RAM   (rwx) : ORIGIN = 0x20000000, LENGTH = 16M
}

_estack = ORIGIN(RAM) + LENGTH(RAM);

INCLUDE sections.ld
