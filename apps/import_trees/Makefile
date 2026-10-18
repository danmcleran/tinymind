# Tests for the tree-ensemble importer. Needs a Python with numpy and
# scikit-learn (XGBoost / LightGBM tests run when those are installed) and a
# host C++ compiler; use a venv, never the system Python:
#
#   python3 -m venv .venv && .venv/bin/pip install numpy scikit-learn xgboost lightgbm
#   make test PY=.venv/bin/python
PY ?= python3

test:
	$(PY) -I test_import_trees.py -v

.PHONY: test
