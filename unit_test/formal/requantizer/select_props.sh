#!/bin/sh
# Print the --property arguments for one proof job: every check CBMC
# generates (library and reference) except the 64-bit multiply overflow
# checks, which the shift-independent kernels proof covers for every operand.
# Those two are the expensive obligations; repeating them in each of the
# per-shift jobs would multiply the run time several times over.
#
# usage: select_props.sh <cbmc command line without property selection>
"$@" --show-properties 2>/dev/null | awk '
    /^Property / { name = $2; sub(/:$/, "", name); line = 0; next }
    name != "" {
        line++
        if (line == 2) {
            if ($0 !~ /overflow on signed \* in \(int64_t\)/ && $0 !~ /overflow on signed \* in a \* b/)
                print "--property " name
            name = ""
        }
    }'
