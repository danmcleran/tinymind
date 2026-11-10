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

Parity test: a GRU trained in PyTorch and exported with gru_export.py must
compute the same function in tinymind's GruNeuralNetwork.

Run with `make parity PYTHON=<python with torch>` (builds gru_parity first),
or directly: python test_gru_parity.py --binary output/gru_parity

Checks:
  1. The trained XOR GRU: tinymind (double) matches PyTorch (evaluated in
     float64 on the same weights) to 1e-9 on the four XOR inputs and on random
     length-5 sequences, and classifies XOR 4/4.
  2. The same file in Q16.16 matches to within the lookup-table activations'
     error.
  3. A randomly initialised GRU with large weights (all three gates far from
     linear) matches too, so the check does not depend on what XOR training
     happened to learn.
  4. The exporter refuses a model whose candidate recurrent bias b_hn is
     nonzero, and GRUNet keeps b_hn at zero through training.
"""

import argparse
import copy
import contextlib
import io
import os
import subprocess
import sys
import tempfile

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gru_export  # noqa: E402

DOUBLE_TOLERANCE = 1e-9
Q16_TOLERANCE = 0.02


def run_tinymind(binary, weights_path, mode, sequences):
    stdin = "".join(
        "%d\n" % len(seq) + "".join("%.17g %.17g\n" % (step[0], step[1]) for step in seq)
        for seq in sequences)
    result = subprocess.run([binary, weights_path, mode], input=stdin,
                            capture_output=True, text=True, check=True)
    return np.array([float(v) for v in result.stdout.split()])


def run_torch(model, sequences):
    reference = copy.deepcopy(model).double()
    with torch.no_grad():
        return np.array([
            reference(torch.tensor(np.asarray(seq, dtype=np.float64)[None]))[0].item()
            for seq in sequences])


def export(model, path, use_q16_16):
    with contextlib.redirect_stdout(io.StringIO()):
        gru_export.export_gru_weights(model, path, use_q16_16=use_q16_16)


def compare(binary, model, sequences, workdir, tag):
    """Return (max |torch - tinymind| for double, the same for Q16.16)."""
    float_path = os.path.join(workdir, tag + "_float.txt")
    q16_path = os.path.join(workdir, tag + "_q16_16.txt")
    export(model, float_path, use_q16_16=False)
    export(model, q16_path, use_q16_16=True)
    expected = run_torch(model, sequences)
    got_double = run_tinymind(binary, float_path, "double", sequences)
    got_q16 = run_tinymind(binary, q16_path, "q16", sequences)
    return expected, got_double, got_q16


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", default=os.path.join("output", "gru_parity"))
    args = parser.parse_args()
    binary = os.path.abspath(args.binary)

    failures = []

    def check(condition, message):
        print(("PASS  " if condition else "FAIL  ") + message)
        if not condition:
            failures.append(message)

    rng = np.random.default_rng(1)
    xor_sequences = [seq for seq in gru_export.XOR_X]
    random_sequences = [rng.uniform(-1.0, 1.0, size=(5, 2)).tolist() for _ in range(32)]
    xor_targets = np.array(gru_export.XOR_Y)[:, 0]

    with tempfile.TemporaryDirectory() as workdir:
        # 1-2. The trained XOR model.
        trained = gru_export.train_xor_gru(seed=0, verbose=False)
        H = trained.hidden_size
        check(bool(torch.all(trained.gru.bias_hh_l0[2 * H:] == 0)),
              "GRUNet keeps b_hn at exactly zero through Adam training")

        expected, got_double, got_q16 = compare(binary, trained, xor_sequences, workdir, "xor")
        err_double = np.max(np.abs(expected - got_double))
        err_q16 = np.max(np.abs(expected - got_q16))
        check(err_double < DOUBLE_TOLERANCE,
              "trained XOR GRU, double: max |torch - tinymind| = %.3g (< %g)" % (err_double, DOUBLE_TOLERANCE))
        check(err_q16 < Q16_TOLERANCE,
              "trained XOR GRU, Q16.16: max |torch - tinymind| = %.3g (< %g)" % (err_q16, Q16_TOLERANCE))
        correct = int(np.sum((got_double > 0.5) == (xor_targets > 0.5)))
        check(correct == 4, "trained XOR GRU in tinymind classifies %d/4" % correct)

        expected, got_double, got_q16 = compare(binary, trained, random_sequences, workdir, "xor_long")
        err_double = np.max(np.abs(expected - got_double))
        err_q16 = np.max(np.abs(expected - got_q16))
        check(err_double < DOUBLE_TOLERANCE,
              "trained XOR GRU, 32 random length-5 sequences, double: max diff %.3g" % err_double)
        check(err_q16 < Q16_TOLERANCE,
              "trained XOR GRU, 32 random length-5 sequences, Q16.16: max diff %.3g" % err_q16)

        # 3. A random GRU with large weights, so every gate is well inside its
        # nonlinear range and a wrong sign or gate swap shows up.
        torch.manual_seed(123)
        wide = gru_export.GRUNet(input_size=2, hidden_size=4, output_size=1)
        with torch.no_grad():
            for name, parameter in wide.named_parameters():
                parameter.mul_(4.0)
            wide.gru.bias_hh_l0[2 * H:] = 0.0
        expected, got_double, got_q16 = compare(binary, wide, random_sequences, workdir, "wide")
        err_double = np.max(np.abs(expected - got_double))
        err_q16 = np.max(np.abs(expected - got_q16))
        spread = np.max(expected) - np.min(expected)
        check(spread > 0.5, "random wide GRU outputs span %.3f, so the comparison is not trivial" % spread)
        check(err_double < DOUBLE_TOLERANCE,
              "random wide GRU, double: max |torch - tinymind| = %.3g" % err_double)
        check(err_q16 < Q16_TOLERANCE,
              "random wide GRU, Q16.16: max |torch - tinymind| = %.3g" % err_q16)

        # 4. A nonzero b_hn cannot be represented, so export must refuse it.
        bad = copy.deepcopy(trained)
        with torch.no_grad():
            bad.gru.bias_hh_l0[2 * H] = 0.25
        try:
            export(bad, os.path.join(workdir, "bad.txt"), use_q16_16=False)
            check(False, "exporter refuses a GRU with nonzero b_hn")
        except ValueError:
            check(True, "exporter refuses a GRU with nonzero b_hn")

    if failures:
        print("\n%d check(s) failed" % len(failures))
        return 1
    print("\nAll GRU parity checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
