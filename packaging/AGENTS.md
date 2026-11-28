# packaging

Projects that consume tinymind through CMake, PlatformIO, Arduino, Zephyr and
ESP-IDF, built by the `Packaging` workflow (`.github/workflows/packaging.yml`).
Repo-wide rules are in the root `AGENTS.md`; what each package is and how the
checks run are in `packaging/README.md`.

- The package files live outside this directory, where each build system looks
  for them: `CMakeLists.txt` (CMake, and ESP-IDF in its `ESP_PLATFORM` branch),
  `cmake/`, `library.json` (PlatformIO), `library.properties` (Arduino),
  `idf_component.yml` (ESP-IDF), `zephyr/` (Zephyr module) and `src/` (Arduino
  and PlatformIO layout).
- `src/` is generated: run `tools/gen_src_forwarders.py` after adding or
  removing a header in `cpp/` or `cpp/include/`; the metadata job runs it with
  `--check`.
- A header in `cpp/` includes another by a path relative to itself, never one
  that needs `-Icpp/include`: the Arduino layout and the no-include-path
  builds depend on it.
- A release sets the version in all four manifests before the tag;
  `tools/check_package_versions.py --tag vX.Y` checks them.
- Every consumer runs `selftest/tinymind_selftest.hpp` and the job requires its
  PASS line where the target runs. A change to the self-test, or a new
  package, must be shown to fail on a planted defect before it is trusted.
- The jobs are not required checks; `make check` is still the gate.
