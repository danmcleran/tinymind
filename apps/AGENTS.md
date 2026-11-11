# apps

Host-side tools. None of them ships in an embedded build. Repo-wide rules are in
the root `AGENTS.md`.

`activation/` (LUT generator for `lookupTables.cpp`), `import_pytorch/` + `import_onnx/` (Phase 15 int8 importers), `import_trees/` (scikit-learn / XGBoost / LightGBM tree ensembles → C++14 `QStaticDecisionTree` / `QStaticGBDT` pools; `make test` with a venv, compile-verifies every policy bit-exact; not in CI), `anfis_train/` (host ANFIS trainer: hybrid least-squares + normalized premise descent, grid or subtractive-clustering scatter partition, bell/triangular membership policies, rule pruning; emits the frozen arrays `cpp/anfis.hpp` consumes; numpy only, never run in CI).

- `activation/` regenerates the LUT headers and `cpp/lookupTables.cpp`: build it, run `./output/activationTableGenerator ../../cpp/`, then `make check`. The generator itself needs C++17 (`std::filesystem`); it is a host tool, outside the C++14 floor that binds `cpp/`.
- Python dependencies go in a venv, never the system Python.
