# Tests for the ONNX importer. Needs a Python with numpy, torch, onnx and
# onnxruntime, and a host g++; use a venv, never the system Python:
#
#   python3 -m venv .venv && .venv/bin/pip install numpy torch onnx onnxruntime
#   make test PY=.venv/bin/python
#
# Not run in CI: torch and onnxruntime are not CI dependencies.
PY ?= python3

test:
	$(PY) -I test_import_onnx.py -v

.PHONY: test
