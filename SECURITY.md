# Security Policy

## Reporting a vulnerability

Please report security issues privately through GitHub's private vulnerability
reporting: open the repository's **Security** tab and choose **Report a
vulnerability** (or go to
<https://github.com/danmcleran/tinymind/security/advisories/new>). Do not open a
public issue for a suspected vulnerability.

Include what you can of:

- the affected header, layer or tool, and the tinymind commit or release;
- the build configuration (compiler, flags, and any `TINYMIND_ENABLE_*` gates);
- a minimal reproducer (an input, model or fuzz corpus entry) and what goes wrong.

tinymind is maintained by one person, so responses are best effort. You can expect
an acknowledgement within a week and an assessment once the report has been
reproduced. Fixes land on `master` and ship in the next release, and the advisory
credits the reporter unless they prefer otherwise.

## Supported versions

Security fixes go to `master` and the latest release. Older releases are not
patched.

## Scope

In scope: memory-safety and undefined-behaviour defects in the library headers
under `cpp/` (out-of-bounds access, overflow that the code does not intend,
uninitialised reads) and in the host tools under `apps/` that parse model or
dataset files, for example the PyTorch, ONNX and tree-ensemble importers.

Out of scope: numerical accuracy of a trained model, and the examples and
benchmarks, which are demonstrations rather than hardened code. Reports about
them are still welcome as ordinary issues.
