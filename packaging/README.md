# packaging

tinymind packaged for other build systems, and the projects CI builds to prove
each package works. The packages themselves live where each build system looks
for them:

| Build system | Package files | Consumer here | What CI does with it |
|---|---|---|---|
| CMake | `CMakeLists.txt`, `cmake/tinymindConfig.cmake.in` | `cmake/` | builds and runs it through `add_subdirectory()`, then through `find_package()` on an installed copy |
| PlatformIO | `library.json`, `src/` | `platformio/` | builds and runs it on the host (`native`); builds it for a NUCLEO-F411RE (STM32Cube) |
| Arduino | `library.properties`, `src/` | `arduino/tinymind_selftest/` | compiles it for a Nano 33 BLE (Arduino Mbed OS core) |
| Zephyr | `zephyr/module.yml`, `zephyr/Kconfig`, `zephyr/CMakeLists.txt` | `zephyr/` | builds it for `qemu_cortex_m3` (Zephyr v4.3.2) and runs it under QEMU |
| ESP-IDF | `CMakeLists.txt` (its `ESP_PLATFORM` branch), `idf_component.yml` | `esp-idf/` | builds it for the ESP32-S3 (ESP-IDF v5.5.6) and runs it under Espressif's QEMU |

Each consumer runs the same self-test, `selftest/tinymind_selftest.hpp`: the
generated models of `examples/tflite_streaming_int8` are fed that example's
16-second test recording, `step()` one frame at a time and `forward()` over
the window ending at each frame, and all 3,072 int8 outputs must equal the
ones TensorFlow Lite for Microcontrollers computed. The consumers also read
the Q8.8 tanh lookup table (`TINYMIND_USE_TANH_8_8`), so the job fails if
the package does not compile `cpp/lookupTables.cpp` with the flags the
application sees. The run prints `tinymind self-test: PASS`, and the job
requires that line.

How each build system is used from a project of your own is in the README,
under "Using tinymind from your build system".

## Running the checks locally

```bash
# CMake (3.14 or later; cmake and ninja from a venv work)
cmake -S packaging/cmake -B build/sub -DTINYMIND_FROM=subdirectory
cmake --build build/sub && ctest --test-dir build/sub --output-on-failure

# PlatformIO (pip install platformio in a venv)
cd packaging/platformio && pio run -e native && .pio/build/native/program

# src/ and the manifests
python3 -I tools/gen_src_forwarders.py --check
python3 -I tools/check_package_versions.py
```

The Zephyr, ESP-IDF and Arduino jobs need their SDKs (west and the Zephyr
SDK, the `espressif/idf` container, `arduino-cli` with the Mbed OS core); the
commands are in `.github/workflows/packaging.yml`.

## Keeping the packages current

- **Adding or removing a header in `cpp/` or `cpp/include/`:** run
  `tools/gen_src_forwarders.py` and commit `src/`. Arduino puts only `src/`
  on the include path, so a header without a forwarder there cannot be
  included from a sketch, or from a generated `model.hpp` built in one.
- **A release:** set the version in `CMakeLists.txt`, `library.json`,
  `library.properties` and `idf_component.yml` to the new one (`X.Y.0` for
  tag `vX.Y`) before tagging; `tools/check_package_versions.py --tag vX.Y`
  checks it.
- **Library headers include each other by paths relative to themselves**
  (`"include/tinymind_platform.hpp"`, `"../qaffine.hpp"`), never through an
  include path. That is what lets the Arduino forwarders, and any build that
  puts only one directory on the include path, work.
