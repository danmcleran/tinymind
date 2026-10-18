"""
Copyright (c) 2026 Dan McLeran

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

"""Tests for tinymind_import_trees. Run with `make test` (needs scikit-learn;
XGBoost / LightGBM tests are skipped when those are not installed, and
compile checks when no host C++ compiler is found)."""

import os
import shutil
import subprocess
import sys
import tempfile
import unittest
import warnings

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tinymind_import_trees as T  # noqa: E402

from sklearn.datasets import load_breast_cancer, load_diabetes, load_digits, load_iris, load_wine  # noqa: E402
from sklearn.ensemble import (ExtraTreesClassifier, GradientBoostingClassifier,  # noqa: E402
                              GradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor)
from sklearn.model_selection import train_test_split  # noqa: E402
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor  # noqa: E402

try:
    import xgboost as xgb
except ImportError:
    xgb = None
try:
    import lightgbm as lgb
except ImportError:
    lgb = None

GXX = shutil.which("g++")
CLANG = shutil.which("clang++")
ARM = shutil.which("arm-none-eabi-g++")

warnings.filterwarnings("ignore")


def split(X, y, seed=0):
    return train_test_split(X, y, test_size=0.3, random_state=seed, stratify=y if y.dtype.kind in "iu" else None)


def int_digits():
    """Digits pixels (0..16) as integer features, a 3-class subset to keep it quick."""
    X, y = load_digits(return_X_y=True)
    keep = y < 3
    return X[keep], y[keep]


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def export(self, model, X, **kw):
        path = os.path.join(self.tmp, "m%d.hpp" % len(os.listdir(self.tmp)))
        return T.export_trees(model, X, path, namespace="m", **kw)

    def check_compiled(self, e, X, **kw):
        if GXX is None:
            self.skipTest("no g++")
        e.verify(X, cxx=GXX, **kw)


class ThresholdRules(unittest.TestCase):
    def test_integer_grid_matches_brute_force(self):
        q = T.FeatureQuant.integer(1, "int8")
        grid = np.arange(-128, 128)
        for t in (-130.0, -128.0, -127.5, -0.5, 0.0, 3.5, 4.0, 4.000001, 126.5, 127.0, 200.0):
            for strict in (False, True):
                kind, thr = q.threshold(0, t, strict)
                want = (grid < t) if strict else (grid <= t)
                got = {"left": np.ones_like(want), "right": np.zeros_like(want)}.get(kind)
                if got is None:
                    got = grid <= thr
                np.testing.assert_array_equal(got, want, err_msg="t=%r strict=%r" % (t, strict))

    def test_quantize_matches_rounding_contract(self):
        q = T.FeatureQuant(np.array([0.5], np.float32), np.array([3]), "int8")
        x = np.array([[-1.25], [-0.25], [0.25], [0.75], [1000.0], [-1000.0]])
        # x / 0.5 = -2.5, -0.5, 0.5, 1.5 -> ties away from zero -> -3, -1, 1, 2; then + 3, clamped.
        np.testing.assert_array_equal(q.quantize(x)[:, 0], [0, 2, 4, 5, 127, -128])

    def test_integer_quant_rejects_non_integers(self):
        with self.assertRaises(ValueError):
            T.FeatureQuant.integer(1).quantize(np.array([[1.5]]))
        with self.assertRaises(ValueError):
            T.FeatureQuant.integer(1).quantize(np.array([[300.0]]))


class IntegerFeaturesAreExact(Base):
    """On integer features every split is reproduced exactly, so the export
    must agree with the original on held-out data, sample for sample."""

    def run_exact(self, model, X, y, **kw):
        Xtr, Xte, ytr, _ = split(X, y)
        model.fit(Xtr, ytr)
        e = self.export(model, Xtr, quant=T.FeatureQuant.integer(X.shape[1], "int8"), **kw)
        self.assertEqual(e.agreement(Xte), 1.0, e.report)
        self.check_compiled(e, Xte[:40])
        return e

    def test_decision_tree(self):
        self.run_exact(DecisionTreeClassifier(max_depth=6, random_state=0), *int_digits())

    def test_random_forest_soft_vote(self):
        X, y = int_digits()
        self.run_exact(RandomForestClassifier(n_estimators=8, max_depth=6, random_state=0), X, y, voting="soft")

    def test_gradient_boosting_multiclass(self):
        self.run_exact(GradientBoostingClassifier(n_estimators=10, max_depth=3, random_state=0), *int_digits())

    def test_gradient_boosting_binary(self):
        X, y = int_digits()
        self.run_exact(GradientBoostingClassifier(n_estimators=10, max_depth=3, random_state=0), X, (y == 1).astype(int))

    @unittest.skipIf(xgb is None, "xgboost not installed")
    def test_xgboost_multiclass(self):
        self.run_exact(xgb.XGBClassifier(n_estimators=10, max_depth=3), *int_digits())

    @unittest.skipIf(xgb is None, "xgboost not installed")
    def test_xgboost_binary(self):
        X, y = int_digits()
        self.run_exact(xgb.XGBClassifier(n_estimators=10, max_depth=3), X, (y == 1).astype(int))

    @unittest.skipIf(lgb is None, "lightgbm not installed")
    def test_lightgbm_multiclass(self):
        self.run_exact(lgb.LGBMClassifier(n_estimators=10, num_leaves=7, verbose=-1), *int_digits())

    @unittest.skipIf(lgb is None, "lightgbm not installed")
    def test_lightgbm_binary(self):
        X, y = int_digits()
        self.run_exact(lgb.LGBMClassifier(n_estimators=10, num_leaves=7, verbose=-1), X, (y == 1).astype(int))

    def test_hard_vote_matches_its_own_definition(self):
        # A hard vote can differ from scikit-learn's probability-averaging
        # predict, so check it against a direct vote over the trees instead.
        X, y = int_digits()
        Xtr, Xte, ytr, _ = split(X, y)
        rf = RandomForestClassifier(n_estimators=8, max_depth=6, random_state=0).fit(Xtr, ytr)
        e = self.export(rf, Xtr, quant=T.FeatureQuant.integer(X.shape[1]))
        votes = np.zeros((len(Xte), 3), np.int64)
        for est in rf.estimators_:
            votes[np.arange(len(Xte)), est.predict(Xte).astype(int)] += 1
        np.testing.assert_array_equal(e.predict(Xte), votes.argmax(1))
        self.check_compiled(e, Xte[:40])


class CalibratedFloatFeatures(Base):
    """Real-valued features on a min/max int8 grid: agreement is not exact
    (a sample within one grid step of a threshold can flip), so it is held to
    a floor on held-out data, and the compiled header to bit-exactness."""

    def run_float(self, model, loader, floor, **kw):
        X, y = loader(return_X_y=True)
        Xtr, Xte, ytr, _ = split(X, y)
        model.fit(Xtr, ytr)
        e = self.export(model, Xtr, **kw)
        self.assertGreaterEqual(e.agreement(Xte), floor, e.report)
        self.check_compiled(e, Xte[:40])
        return e

    def test_random_forest_hard(self):
        self.run_float(RandomForestClassifier(n_estimators=10, max_depth=5, random_state=0), load_wine, 0.95)

    def test_random_forest_soft(self):
        self.run_float(RandomForestClassifier(n_estimators=10, max_depth=5, random_state=0), load_wine, 0.97, voting="soft")

    def test_extra_trees(self):
        self.run_float(ExtraTreesClassifier(n_estimators=10, max_depth=5, random_state=0), load_iris, 0.95)

    def test_gradient_boosting(self):
        self.run_float(GradientBoostingClassifier(n_estimators=20, max_depth=3, random_state=0), load_breast_cancer, 0.98)

    @unittest.skipIf(xgb is None, "xgboost not installed")
    def test_xgboost(self):
        self.run_float(xgb.XGBClassifier(n_estimators=20, max_depth=3), load_wine, 0.97)

    @unittest.skipIf(lgb is None, "lightgbm not installed")
    def test_lightgbm(self):
        self.run_float(lgb.LGBMClassifier(n_estimators=20, num_leaves=7, verbose=-1), load_wine, 0.97)


class Regressors(Base):
    """Regressors return raw / kOutputScale. Error against the original is held
    to a small fraction of the target range on held-out data."""

    def run_reg(self, model, quant_dtype="int16"):
        X, y = load_diabetes(return_X_y=True)
        Xtr, Xte, ytr, _ = train_test_split(X, y, test_size=0.3, random_state=0)
        model.fit(Xtr, ytr)
        e = self.export(model, Xtr, quant=T.FeatureQuant.from_data(Xtr, quant_dtype))
        rng = float(y.max() - y.min())
        err = e.agreement(Xte)
        self.assertLess(err, 0.05 * rng, e.report)
        self.check_compiled(e, Xte[:40])
        return e

    def test_decision_tree(self):
        self.run_reg(DecisionTreeRegressor(max_depth=5, random_state=0))

    def test_random_forest(self):
        self.run_reg(RandomForestRegressor(n_estimators=8, max_depth=5, random_state=0))

    def test_gradient_boosting(self):
        self.run_reg(GradientBoostingRegressor(n_estimators=20, max_depth=3, random_state=0))

    @unittest.skipIf(xgb is None, "xgboost not installed")
    def test_xgboost(self):
        self.run_reg(xgb.XGBRegressor(n_estimators=20, max_depth=3))

    @unittest.skipIf(lgb is None, "lightgbm not installed")
    def test_lightgbm(self):
        self.run_reg(lgb.LGBMRegressor(n_estimators=20, num_leaves=7, verbose=-1))

    def test_leaf_rounding_is_negligible_on_integer_features(self):
        # With exact splits only the int32 leaf rounding remains: tiny.
        X, y = int_digits()
        yr = y.astype(float) * 10.0 + X[:, 20]
        Xtr, Xte, ytr, _ = train_test_split(X, yr, test_size=0.3, random_state=0)
        m = GradientBoostingRegressor(n_estimators=20, max_depth=3, random_state=0).fit(Xtr, ytr)
        e = self.export(m, Xtr, quant=T.FeatureQuant.integer(X.shape[1]))
        self.assertLess(e.agreement(Xte), 1e-6 * float(yr.max() - yr.min()))


class Toolchains(Base):
    """The same header under both compilers and both language levels, other
    input dtypes, and a freestanding cross-compile without float support."""

    def model(self):
        X, y = load_wine(return_X_y=True)
        m = GradientBoostingClassifier(n_estimators=10, max_depth=3, random_state=0).fit(X, y)
        return m, X

    def test_compilers_and_standards(self):
        m, X = self.model()
        e = self.export(m, X)
        for cxx in (GXX, CLANG):
            if cxx is None:
                continue
            for std in ("c++14", "c++17"):
                e.verify(X[:30], cxx=cxx, std=std)

    def test_uint8_and_int16_grids(self):
        m, X = self.model()
        for dtype in ("uint8", "int16"):
            e = self.export(m, X, quant=T.FeatureQuant.from_data(X, dtype))
            self.assertIn("typedef %s InputType" % T._DTYPES[dtype][2], open(e.path).read())
            self.check_compiled(e, X[:30])

    @unittest.skipIf(ARM is None, "no arm-none-eabi-g++")
    def test_freestanding_cortex_m0plus(self):
        # No float, no stdlib: the deployable shape. predict<> under every policy.
        m, X = self.model()
        e = self.export(m, X)
        src = os.path.join(self.tmp, "fs.cpp")
        with open(src, "w") as fh:
            fh.write('#include "%s"\n' % e.path)
            fh.write("int run(const m::InputType* x)\n{\n"
                     "    return static_cast<int>(m::predict<tinymind::QTreeInlineEval>(x)\n"
                     "        + m::predict<tinymind::QTreeCompactEval>(x) + m::predict<tinymind::QTreeWalkEval>(x));\n}\n")
        inc = ["-I" + d for d in T._repo_include_dirs()]
        for std in ("c++14", "c++17"):
            subprocess.run([ARM, "-std=" + std, "-mcpu=cortex-m0plus", "-mthumb", "-mfloat-abi=soft", "-O2",
                            "-Wall", "-Wextra", "-Werror", "-Wpedantic", "-c", src, "-o", os.devnull] + inc,
                           check=True, capture_output=True, text=True)


class Rejections(Base):
    def test_nan_calibration(self):
        X, y = load_iris(return_X_y=True)
        m = DecisionTreeClassifier(max_depth=2).fit(X, y)
        Xn = X.copy()
        Xn[0, 0] = np.nan
        with self.assertRaises(ValueError):
            self.export(m, Xn)

    def test_multi_output_tree(self):
        X, y = load_iris(return_X_y=True)
        m = DecisionTreeRegressor(max_depth=2).fit(X, np.stack([y, y], axis=1))
        with self.assertRaises(ValueError):
            self.export(m, X)

    def test_unsupported_model(self):
        from sklearn.linear_model import LogisticRegression
        X, y = load_iris(return_X_y=True)
        with self.assertRaises(ValueError):
            self.export(LogisticRegression(max_iter=200).fit(X, y), X)

    def test_base_check_catches_a_parsing_error(self):
        # Plant a parsing defect: shift every scikit-learn threshold. The
        # measured base then varies by sample, and export must refuse.
        X, y = load_diabetes(return_X_y=True)
        m = GradientBoostingRegressor(n_estimators=10, max_depth=3, random_state=0).fit(X, y)
        good = T._sklearn_tree

        def shifted(est, value_fn):
            tree = good(est, value_fn)
            for n in tree.nodes:
                if n.feature >= 0:
                    n.threshold += 0.01
            return tree

        T._sklearn_tree = shifted
        try:
            with self.assertRaises(ValueError) as ctx:
                self.export(m, X)
            self.assertIn("does not reproduce", str(ctx.exception))
        finally:
            T._sklearn_tree = good


if __name__ == "__main__":
    unittest.main()
