/*
 * Standalone replay driver for the fuzz harnesses, without libFuzzer.
 *
 * Links against one harness's LLVMFuzzerTestOneInput, replays every corpus
 * file named on the command line (a directory is expanded to its files, in
 * name order), and prints one line per input: the FNV-1a digest of the
 * outputs the harness passed to tinymind_fuzz::emit, then the file name.
 *
 * Built without sanitizers by g++ and clang++ at -O0 and -O3, it backs
 * `make agree`: the same corpus through every build must print the same
 * digests. A difference is undefined behaviour the optimizer exploited, or
 * arithmetic whose result depends on the compiler -- both invisible to a
 * sanitizer build, which only runs one compiler at one optimization level.
 * The harnesses' property checks stay live in these builds too.
 *
 * Built with -fsanitize=memory it backs `make msan-replay`, which libFuzzer
 * cannot: libFuzzer's own runtime is not MSan-instrumented on a stock
 * toolchain.
 *
 * Copyright (c) 2026 Dan McLeran. MIT (see LICENSE.txt).
 */
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <dirent.h>
#include <sys/stat.h>

#include "fuzz_check.hpp"

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size);

namespace {

// Plain C I/O only: the MSan build links an uninstrumented libstdc++, and
// std::string / std::vector members instantiated there would report false
// positives.
const std::size_t kMaxPath = 4096;
const std::size_t kMaxInput = 1u << 20;

int comparePaths(const void* a, const void* b)
{
    return std::strcmp(*static_cast<char* const*>(a), *static_cast<char* const*>(b));
}

// Replays one file; returns false if it cannot be read.
bool replayFile(const char* path, uint8_t* buffer)
{
    FILE* f = std::fopen(path, "rb");
    if (f == nullptr)
    {
        std::fprintf(stderr, "replay: cannot read %s\n", path);
        return false;
    }
    const std::size_t size = std::fread(buffer, 1, kMaxInput, f);
    std::fclose(f);

    tinymind_fuzz::resetDigest();
    LLVMFuzzerTestOneInput(buffer, size);
    std::printf("%016llx %s\n", static_cast<unsigned long long>(tinymind_fuzz::digest()), path);
    return true;
}

bool replayPath(const char* path, uint8_t* buffer)
{
    struct stat st;
    if ((stat(path, &st) != 0) || !S_ISDIR(st.st_mode))
    {
        return replayFile(path, buffer);
    }

    DIR* dir = opendir(path);
    if (dir == nullptr)
    {
        std::fprintf(stderr, "replay: cannot open %s\n", path);
        return false;
    }
    std::size_t count = 0;
    std::size_t capacity = 64;
    char** names = static_cast<char**>(std::malloc(capacity * sizeof(char*)));
    for (dirent* entry = readdir(dir); entry != nullptr; entry = readdir(dir))
    {
        if ((std::strcmp(entry->d_name, ".") == 0) || (std::strcmp(entry->d_name, "..") == 0))
        {
            continue;
        }
        if (count == capacity)
        {
            capacity *= 2;
            names = static_cast<char**>(std::realloc(names, capacity * sizeof(char*)));
        }
        names[count] = static_cast<char*>(std::malloc(kMaxPath));
        std::snprintf(names[count], kMaxPath, "%s/%s", path, entry->d_name);
        ++count;
    }
    closedir(dir);

    std::qsort(names, count, sizeof(char*), comparePaths);
    bool ok = true;
    for (std::size_t i = 0; i < count; ++i)
    {
        ok = replayFile(names[i], buffer) && ok;
        std::free(names[i]);
    }
    std::free(names);
    return ok;
}

} // namespace

int main(int argc, char** argv)
{
    uint8_t* buffer = static_cast<uint8_t*>(std::malloc(kMaxInput));
    bool ok = true;
    for (int i = 1; i < argc; ++i)
    {
        ok = replayPath(argv[i], buffer) && ok;
    }
    std::free(buffer);
    return ok ? 0 : 1;
}
