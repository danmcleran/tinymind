# PyTorch -> TinyMind int8 Quantization (XOR)

End-to-end post-training int8 quantization demo. The PyTorch script trains
a small XOR MLP in float32, observes per-tensor activation ranges over a
9 x 9 calibration grid, then emits a `weights.hpp` with int8 weights,
int32 biases, and per-tensor `(scale, zero_point)` metadata. The C++ side
(`xor_quant.cpp`) reconstructs the per-layer `Requantizer` and the
`qsigmoid` LUT host-side and runs a pure-integer forward pass over the
four XOR inputs.

This is the affine quantization counterpart to `examples/pytorch/xor/`,
which uses TinyMind's existing single-`ValueType` Q16.16 fixed-point path.

## Pipeline

```
float32 PyTorch                  int8 TinyMind
---------------                  -------------
    nn.Linear(2, 4)      -->     QDense<int8,int8,int32,int8>
    nn.ReLU                -->   qreluBuffer (or fold into requant)
    nn.Linear(4, 1)      -->     QDense<int8,int8,int32,int8>
    nn.Sigmoid             -->   buildQSigmoidLUT + qApplyLUT
```

Calibration follows the TFLite / CMSIS-NN convention:

  * Symmetric per-tensor weights (`zero_point = 0`, `qmax = 127`).
  * Asymmetric activations (`zero_point` in `[-128, 127]`).
  * `int32` biases held at `bias_scale = input_scale * weight_scale`,
    so they add directly into the int32 accumulator.

The `Requantizer` `(multiplier, shift)` pair is rebuilt at startup from
the float scales via `tinymind::buildRequantizer`. For an MCU deployment,
that step happens once on the host and the resulting integer triple is
embedded as a constant — the inference binary needs neither
`<cmath>` nor float math.

## Building and Running

```bash
# (Optional) regenerate weights.hpp from PyTorch training. Requires torch.
python3 xor_quant.py

# Build and run the int8 inference pipeline.
make clean && make && make run
```

The committed `weights.hpp` ships an exact textbook 2-4-1 ReLU+Sigmoid
solver so the example runs without requiring PyTorch. Re-running
`xor_quant.py` overwrites the file with whatever the SGD trainer
converges to.

Expected output:

```
int8 XOR accuracy: 4/4
```

## Weight File Format

`weights.hpp` is a generated C++ header; see the file itself for the
field-by-field layout. Key constants:

| Constant | Meaning |
|----------|---------|
| `kInputScale`, `kInputZeroPoint` | Asymmetric calibration of the input tensor |
| `kHiddenScale`, `kHiddenZeroPoint` | Asymmetric calibration of the post-ReLU hidden tensor |
| `kLogitScale`, `kLogitZeroPoint` | Asymmetric calibration of the pre-sigmoid logit tensor |
| `kSigmoidOutScale`, `kSigmoidOutZeroPoint` | Output grid of the sigmoid LUT (1/256, -128 covers (0, 1)) |
| `kFc1WeightScale`, `kFc2WeightScale` | Symmetric weight scales |
| `kFc1Weights`, `kFc1Biases` | Row-major int8 weights and int32 biases for `nn.Linear(2, hidden)` |
| `kFc2Weights`, `kFc2Biases` | Row-major int8 weights and int32 biases for `nn.Linear(hidden, 1)` |

## Calibration ranges

The Python script uses a 9 x 9 grid over `[0, 1]^2` plus the four exact
XOR corners. For larger / non-toy tasks replace the grid with a
representative sample of the deployment distribution; the rest of the
emit logic stays the same.

## Porting to an MCU

Drop the `qcalibration.hpp` include, replace the `buildRequantizer` and
`buildQSigmoidLUT` calls with the integer constants those functions emit
on the host, and the inference path compiles in the
`TINYMIND_ENABLE_QUANTIZATION=1, TINYMIND_ENABLE_FLOAT=0,
TINYMIND_ENABLE_STD=0` corner exercised by `unit_test/embedded`.
