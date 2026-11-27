# Tests for the TFLite importer. Needs a Python with numpy, tensorflow and
# tflite-micro (TFLM's Python interpreter), and a host g++; use a venv, never
# the system Python:
#
#   python3 -m venv .venv && .venv/bin/pip install numpy tensorflow tflite-micro
#   make test PY=.venv/bin/python
#
# Not run in CI: tensorflow and tflite-micro are not CI dependencies. The
# importer itself needs only numpy; examples/tflite_digits_int8 runs a model it
# generated in CI.
PY ?= python3

test:
	$(PY) -I test_import_tflite.py -v

.PHONY: test
