/*
 * Shared property-check and output-digest helpers for the fuzz harnesses.
 *
 * A harness that only feeds its input to a kernel proves only that the
 * kernel does not crash or trip a sanitizer. A harness that checks a
 * property and traps when it is false turns every input the fuzzer (or the
 * replay gate) runs into a test of that property. TINYMIND_FUZZ_CHECK is the
 * one way the harnesses state a property:
 *
 *   - It is not assert(): a build that defines NDEBUG would remove it, and
 *     the cross-build agreement replay builds at -O3.
 *   - It prints the failing expression before trapping, so a crash file
 *     from the nightly run says which property broke, not just "SIGILL".
 *   - It traps rather than returning a nonzero status: libFuzzer treats a
 *     return of -1 as "do not add to corpus", not as a failure.
 *
 * emit() folds a harness's outputs into a per-input FNV-1a digest. Under
 * libFuzzer it only keeps the outputs live (it replaces the old volatile
 * sinks). Under replay_main.cpp the digest is printed per input, so the
 * same corpus replayed through builds from different compilers and
 * optimization levels can be compared byte for byte (`make agree`).
 *
 * makePermutation / permuteMatrix relabel the axes of a layer's weights and
 * activations for the permutation-equivariance checks: integer sums are
 * order-independent, so a relabelled layer must produce the relabelled
 * output bit for bit, and any difference is an indexing or stride error.
 *
 * Copyright (c) 2026 Dan McLeran. MIT (see LICENSE.txt).
 */
#pragma once

#include <cstddef>
#include <cstdint>
#include <cstdio>

#define TINYMIND_FUZZ_CHECK(cond)                                              \
    do                                                                         \
    {                                                                          \
        if (!(cond))                                                           \
        {                                                                      \
            std::fprintf(stderr, "%s:%d: property failed: %s\n",               \
                         __FILE__, __LINE__, #cond);                           \
            __builtin_trap();                                                  \
        }                                                                      \
    } while (0)

namespace tinymind_fuzz {

inline uint64_t& digest()
{
    static uint64_t state = 0xCBF29CE484222325ull;
    return state;
}

inline void resetDigest()
{
    digest() = 0xCBF29CE484222325ull;
}

// Fold raw bytes into the digest. Callers pass plain integer arrays, whose
// object representation is the value on every target the replay runs on.
inline void emit(const void* data, std::size_t size)
{
    const uint8_t* p = static_cast<const uint8_t*>(data);
    uint64_t h = digest();
    for (std::size_t i = 0; i < size; ++i)
    {
        h ^= p[i];
        h *= 0x100000001B3ull;
    }
    digest() = h;
}

template<typename T>
inline void emitValue(const T& value)
{
    emit(&value, sizeof(value));
}

// A permutation of [0, n), Fisher-Yates over a 32-bit LCG seeded by the
// fuzz input, so the corpus can steer it.
inline void makePermutation(std::size_t* perm, std::size_t n, uint32_t seed)
{
    for (std::size_t i = 0; i < n; ++i)
    {
        perm[i] = i;
    }
    uint32_t state = seed * 2654435761u + 1u;
    for (std::size_t i = n; i > 1; --i)
    {
        state = state * 1664525u + 1013904223u;
        const std::size_t j = (state >> 8) % i;
        const std::size_t t = perm[i - 1];
        perm[i - 1] = perm[j];
        perm[j] = t;
    }
}

// dst[row_perm[r]][col_perm[c]] = src[r][c] over `blocks` stacked
// rows x cols matrices (gate-major weight layouts). A null permutation is
// the identity.
template<typename T>
inline void permuteMatrix(const T* src, T* dst, std::size_t blocks, std::size_t rows,
                          std::size_t cols, const std::size_t* row_perm,
                          const std::size_t* col_perm)
{
    for (std::size_t b = 0; b < blocks; ++b)
    {
        for (std::size_t r = 0; r < rows; ++r)
        {
            const std::size_t pr = row_perm ? row_perm[r] : r;
            for (std::size_t c = 0; c < cols; ++c)
            {
                const std::size_t pc = col_perm ? col_perm[c] : c;
                dst[(b * rows + pr) * cols + pc] = src[(b * rows + r) * cols + c];
            }
        }
    }
}

} // namespace tinymind_fuzz
