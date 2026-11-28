---
title: Using tinymind from Your Build System
layout: default
parent: Getting Started
nav_order: 0
---

# Using tinymind from Your Build System

tinymind is header-only: put `cpp/` and `cpp/include/` on the include path and set the feature macros you need. The one source file, `cpp/lookupTables.cpp`, holds the fixed-point activation lookup tables and compiles to an empty object unless a `TINYMIND_USE_<FUNCTION>_<I>_<F>` flag is set. Whatever the build system, define the feature macros (`TINYMIND_ENABLE_*`, all 0 by default) and the table flags for the whole build, so the library's source and yours agree. The repository also packages tinymind for these build systems; a CI job per system builds a small consumer from [`packaging/`](https://github.com/danmcleran/tinymind/tree/master/packaging) that runs a self-test (the generated streaming model of `examples/tflite_streaming_int8` must reproduce every output TFLite Micro computed), and runs it where the target can run.

- **CMake** (3.14 or later). `add_subdirectory()`, `FetchContent`, or `find_package(tinymind)` after `cmake --install`, then link `tinymind::tinymind`:

  ```cmake
  include(FetchContent)
  FetchContent_Declare(tinymind GIT_REPOSITORY https://github.com/danmcleran/tinymind.git GIT_TAG v19.1)
  FetchContent_MakeAvailable(tinymind)
  target_link_libraries(app PRIVATE tinymind::tinymind)
  target_compile_definitions(app PRIVATE TINYMIND_ENABLE_QUANTIZATION=1)
  ```

  The `TINYMIND_ENABLE_*` CMake options (all `OFF`) put a feature macro on `tinymind::tinymind` itself, so every target that links it gets it. For lookup tables, list them in `TINYMIND_LOOKUP_TABLES` (for example `-DTINYMIND_LOOKUP_TABLES="TINYMIND_USE_TANH_8_8;TINYMIND_USE_SIGMOID_8_8"`) and link `tinymind::lookup_tables`, which compiles `cpp/lookupTables.cpp` with them and passes them on. The top-level `CMakeLists.txt` only packages the library; the tests and examples are built by the Makefiles.
- **Zephyr.** The repository is a Zephyr module (`zephyr/module.yml`). Add it to your west manifest (or to `ZEPHYR_EXTRA_MODULES`) and set, in `prj.conf`, `CONFIG_CPP=y`, `CONFIG_STD_CPP14=y` (or newer) and `CONFIG_TINYMIND=y`. `CONFIG_TINYMIND_QUANTIZATION` (on by default), `CONFIG_TINYMIND_FLOAT`, `CONFIG_TINYMIND_STD` and `CONFIG_TINYMIND_SIMD_ARM_DSP` set the feature macros, and `CONFIG_TINYMIND_LOOKUP_TABLES="TINYMIND_USE_TANH_8_8 ..."` compiles the tables. CI builds the sample for `qemu_cortex_m3` with Zephyr v4.3.2 and runs it under QEMU.
- **ESP-IDF** (5.0 or later). The repository root is an ESP-IDF component (its `CMakeLists.txt` calls `idf_component_register` under ESP-IDF, and `idf_component.yml` describes it). Put the checkout in your project's `components/` directory as `tinymind`, or name its parent in `EXTRA_COMPONENT_DIRS` (the component takes its directory's name), and add `REQUIRES tinymind`. Set the macros for every component in the project's `CMakeLists.txt`, after including `project.cmake` and before `project()`: `idf_build_set_property(COMPILE_DEFINITIONS "TINYMIND_ENABLE_QUANTIZATION=1" APPEND)`. CI builds the sample for the ESP32-S3 with ESP-IDF v5.5.6 and runs it under Espressif's QEMU.
- **PlatformIO.** `library.json` describes the library: `lib_deps = https://github.com/danmcleran/tinymind.git#v19.1`, and the macros in `build_flags`, which reach the library's sources too. CI builds the sample for the host and runs it, and builds it for a NUCLEO-F411RE with STM32Cube.
- **Arduino.** `library.properties` describes the library; install it from a ZIP of the repository or clone it into your `libraries/` folder. Arduino puts only the library's `src/` on the include path, so `src/` holds a forwarding header for every header in `cpp/` and `cpp/include/` (written by `tools/gen_src_forwarders.py`), and `#include <tinymind.h>` followed by `#include "qdense.hpp"` (or a generated `model.hpp`) works. The Arduino IDE has no project-wide flags, so define the feature macros at the top of the sketch, before the first tinymind header; lookup tables need build flags (`arduino-cli compile --build-property compiler.cpp.extra_flags=-DTINYMIND_USE_TANH_8_8=1`). The core must compile C++14 or later: the Arduino Mbed OS cores do, the AVR cores (C++11) do not. CI compiles the sample for a Nano 33 BLE.

The version in `CMakeLists.txt`, `library.json`, `library.properties` and `idf_component.yml` must agree; `tools/check_package_versions.py` checks it, and a release sets all four to the new version before it is tagged.
