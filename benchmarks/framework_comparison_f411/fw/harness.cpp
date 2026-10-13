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

// Times one backend on the F411 and prints a key=value report over the VCP.
#include "board.hpp"
#include "backend.hpp"

#ifndef BENCH_REPS
#define BENCH_REPS 33
#endif

namespace
{

void kv(const char* key, uint32_t value)
{
    board_puts(key);
    board_puts("=");
    board_put_u32(value);
    board_puts("\n");
}

void sort(uint32_t* a, int n)
{
    for (int i = 1; i < n; ++i)
    {
        const uint32_t x = a[i];
        int j = i - 1;
        while (j >= 0 && a[j] > x)
        {
            a[j + 1] = a[j];
            --j;
        }
        a[j + 1] = x;
    }
}

[[noreturn]] void halt()
{
    board_flush();
    for (;;)
    {
    }
}

} // namespace

int main()
{
    board_init();
    board_stack_paint();

    const uint32_t s0 = board_cycles();
    const int rc = be_setup();
    const uint32_t setup = board_cycles() - s0;

    if (rc != 0)
    {
        board_puts("\n# begin bench\nname=");
        board_puts(be_name);
        board_puts("\n");
        kv("setup_rc", static_cast<uint32_t>(rc));
        board_puts("# end bench\n");
        halt();
    }

    // Warm-up (first touch, lazy init), then BENCH_REPS timed inferences.
    static uint32_t t[BENCH_REPS];
    be_load(0);
    be_invoke();
    for (int r = 0; r < BENCH_REPS; ++r)
    {
        be_load(r % be_n_test);
        const uint32_t a = board_cycles();
        be_invoke();
        t[r] = board_cycles() - a;
    }
    sort(t, BENCH_REPS);

    // Whole test set: output hash and accuracy.
    uint32_t h = 0x811C9DC5u;
    int correct = 0;
    for (int s = 0; s < be_n_test; ++s)
    {
        be_load(s);
        be_invoke();
        h = be_hash(h);
        if (be_labels && be_argmax() == be_labels[s])
        {
            ++correct;
        }
    }
    const uint32_t stack = board_stack_used();

    board_puts("\n# begin bench\n");
    board_puts("name=");
    board_puts(be_name);
    board_puts("\n");
    kv("setup_rc", static_cast<uint32_t>(rc));
    kv("setup_cycles", setup);
    kv("infer_min", t[0]);
    kv("infer_med", t[BENCH_REPS / 2]);
    kv("infer_max", t[BENCH_REPS - 1]);
    kv("n_test", static_cast<uint32_t>(be_n_test));
    if (be_labels)
    {
        kv("correct", static_cast<uint32_t>(correct));
    }
    board_puts("hash=");
    board_put_hex(h);
    board_puts("\n");
    if (be_expect_hash)
    {
        kv("hash_match", h == be_expect_hash ? 1u : 0u);
    }
    kv("stack_bytes", stack);
    kv("static_ram", board_static_ram());
    kv("arena_used", be_arena_used());
#ifdef PROFILE
    {
        uint32_t c[16];
        const char* tags[16];
        be_load(0);
        const int n = be_profile(c, tags, 16);
        for (int i = 0; i < n; ++i)
        {
            board_puts("layer.");
            board_put_u32(static_cast<uint32_t>(i));
            board_puts(".");
            board_puts(tags[i]);
            board_puts("=");
            board_put_u32(c[i]);
            board_puts("\n");
        }
    }
#endif
    board_puts("# end bench\n");
    halt();
}
