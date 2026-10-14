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

"""Generate the benchmark models and every framework's C/C++ artifacts.

Two models:
  mlp  64 -> 32 -> 16 -> 10 (ReLU), trained on sklearn digits (real accuracy)
  kws  20x20x1 -> Conv3x3x8 -> MaxPool2 -> DW3x3 -> PW8->16 -> AvgPool7 -> Dense10
       (same shape as tinymind examples/kws_cortex_m_int8, random weights)

One float model per network is the source of truth. TFLite's int8 post-training
quantization of it is then re-expressed as tinymind QDense/QConv layers using the
*same* int8 weights, biases and requantizer multipliers, so TFLM and tinymind run
identical integer math and their int8 outputs must match bit for bit.

Outputs land in gen/out/. Requires: tensorflow-cpu, emlearn, scikit-learn, numpy.
"""

import os
import math
import numpy as np

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import tensorflow as tf  # noqa: E402
import emlearn  # noqa: E402
from sklearn.datasets import load_digits  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402
from sklearn.neural_network import MLPClassifier  # noqa: E402

# An upstream import leaves numpy raising on underflow, which breaks sklearn's Adam.
np.seterr(all="warn", under="ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
os.makedirs(OUT, exist_ok=True)

# ---------------------------------------------------------------- helpers


def c_array(ctype, name, values, per_line=16, align=None, const=True):
    vals = list(np.asarray(values).ravel())
    if ctype == "float":
        fmt = lambda v: repr(float(np.float32(v))) + "f"  # noqa: E731
    else:
        fmt = lambda v: str(int(v))  # noqa: E731
    lines = []
    for i in range(0, len(vals), per_line):
        lines.append("    " + ", ".join(fmt(v) for v in vals[i:i + per_line]) + ",")
    a = f"alignas({align}) " if align else ""
    c = "const " if const else ""
    return f"{a}{c}{ctype} {name}[{len(vals)}] =\n{{\n" + "\n".join(lines) + "\n};\n"


def fnv1a(data_bytes):
    h = 0x811C9DC5
    for b in data_bytes:
        h ^= b
        h = (h * 0x01000193) & 0xFFFFFFFF
    return h


def quantize_multiplier(m):
    """TFLite QuantizeMultiplier: m = q * 2^(shift - 31), shift > 0 = left."""
    if m == 0.0:
        return 0, 0
    frac, exp = math.frexp(m)
    q = int(round(frac * (1 << 31)))
    if q == (1 << 31):
        q //= 2
        exp += 1
    if exp < -31:
        return 0, 0
    return q, exp


def requant(in_scale, w_scale, out_scale, out_zp, relu):
    eff = float(np.float64(np.float32(in_scale)) * np.float64(np.float32(w_scale))
                / np.float64(np.float32(out_scale)))
    mult, shift = quantize_multiplier(eff)
    qmin = max(-128, int(out_zp)) if relu else -128
    # tinymind Requantizer: shift > 0 means right shift (opposite of TFLite).
    return (mult, -shift, int(out_zp), qmin, 127)


def rq_init(r):
    return "{%d, %d, %d, %d, %d}" % r


def dense_decl(name, Wq, Bq, s_in, s_w, s_out, z_in, z_out, relu):
    """QDense for per-tensor weights, 1x1 QPointwiseConv2DPerChannel for per-channel."""
    n_out, n_in = Wq.shape
    out = c_array("int8_t", f"W_{name}", Wq, align=4) + c_array("int32_t", f"B_{name}", Bq)
    if len(s_w) == 1:
        r = requant(s_in, s_w[0], s_out, z_out, relu)
        out += (f"using {name}Type = tinymind::QDense<int8_t, int8_t, int32_t, int8_t, {n_in}, {n_out}>;\n"
                f"const {name}Type {name} = {{W_{name}, B_{name}, {int(z_in)}, {rq_init(r)}}};\n\n")
    else:
        rqs = [requant(s_in, s, s_out, z_out, relu) for s in s_w]
        out += (f"const Rq R_{name}[{n_out}] = {{" + ", ".join(rq_init(r) for r in rqs) + "};\n"
                f"using {name}Type = tinymind::QPointwiseConv2DPerChannel<int8_t, int8_t, int32_t, int8_t, "
                f"1, 1, {n_in}, {n_out}>;\n"
                f"const {name}Type {name} = {{W_{name}, B_{name}, {int(z_in)}, R_{name}}};\n\n")
    return out


def to_tflite(model, rep_data, int8):
    conv = tf.lite.TFLiteConverter.from_keras_model(model)
    if int8:
        conv.optimizations = [tf.lite.Optimize.DEFAULT]

        def rep():
            for x in rep_data:
                yield [x[None, ...].astype(np.float32)]

        conv.representative_dataset = rep
        conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        conv.inference_input_type = tf.int8
        conv.inference_output_type = tf.int8
    return conv.convert()


def ref_interpreter(blob):
    it = tf.lite.Interpreter(
        model_content=blob,
        experimental_op_resolver_type=tf.lite.experimental.OpResolverType.BUILTIN_REF,
        experimental_preserve_all_tensors=True)
    it.allocate_tensors()
    return it


def run_tflite(it, xs):
    inp = it.get_input_details()[0]
    out = it.get_output_details()[0]
    ys = []
    for x in xs:
        it.set_tensor(inp["index"], x[None, ...].astype(inp["dtype"]))
        it.invoke()
        ys.append(it.get_tensor(out["index"])[0].copy())
    return np.array(ys)


def ops(it):
    d = it.get_tensor_details()
    by_idx = {t["index"]: t for t in d}
    return it._get_ops_details(), by_idx


def qp(t):
    q = t["quantization_parameters"]
    return np.asarray(q["scales"], np.float32), np.asarray(q["zero_points"], np.int64)


def write(name, text):
    with open(os.path.join(OUT, name), "w") as f:
        f.write(text)


def flatbuffer_header(var, blob):
    return ("#pragma once\n#include <cstddef>\n\n"
            + c_array("unsigned char", var, list(blob), per_line=20, align=16)
            + f"constexpr std::size_t {var}_len = {len(blob)};\n")


HDR = "// Generated by gen/make_models.py -- do not edit.\n"

# ---------------------------------------------------------------- MLP

digits = load_digits()
X = (digits.data / 16.0).astype(np.float32)
y = digits.target.astype(np.int64)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)

mlp = MLPClassifier(hidden_layer_sizes=(32, 16), activation="relu", max_iter=2000,
                    random_state=0)
mlp.fit(Xtr, ytr)
acc_sklearn = (mlp.predict(Xte) == yte).mean()

keras_mlp = tf.keras.Sequential([
    tf.keras.Input(shape=(64,), batch_size=1),
    tf.keras.layers.Dense(32, activation="relu"),
    tf.keras.layers.Dense(16, activation="relu"),
    tf.keras.layers.Dense(10),
])
for layer, W, b in zip(keras_mlp.layers, mlp.coefs_, mlp.intercepts_):
    layer.set_weights([W.astype(np.float32), b.astype(np.float32)])
acc_keras = (np.argmax(keras_mlp.predict(Xte, batch_size=1, verbose=0), 1) == yte).mean()

mlp_f32 = to_tflite(keras_mlp, Xtr[:300], int8=False)
mlp_i8 = to_tflite(keras_mlp, Xtr[:300], int8=True)

it_f = ref_interpreter(mlp_f32)
yf = run_tflite(it_f, Xte)
acc_tflite_f32 = (np.argmax(yf, 1) == yte).mean()

it_q = ref_interpreter(mlp_i8)
in_scale, in_zp = qp(it_q.get_input_details()[0])
in_scale, in_zp = float(in_scale[0]), int(in_zp[0])
Xte_q = np.clip(np.round(Xte / in_scale) + in_zp, -128, 127).astype(np.int8)
yq = run_tflite(it_q, Xte_q)
acc_tflite_i8 = (np.argmax(yq, 1) == yte).mean()
mlp_i8_hash = fnv1a(yq.astype(np.int8).tobytes())

emlearn_c = emlearn.convert(mlp, method="inline")
emlearn_c.save(file=os.path.join(OUT, "mlp_emlearn.h"), name="mlp")

# Extract the int8 FC chain for tinymind.
op_list, T = ops(it_q)
fc = [o for o in op_list if o["op_name"] == "FULLY_CONNECTED"]
assert len(fc) == 3, [o["op_name"] for o in op_list]
tm = [HDR, "#pragma once\n#include <cstdint>\n#include \"qdense.hpp\"\n#include \"qpointwiseconv2d.hpp\"\n\nnamespace tm_mlp {\n\nusing Rq = tinymind::Requantizer<int32_t, int8_t>;\n\n"]
dims = [64, 32, 16, 10]
relu = [True, True, False]
for i, o in enumerate(fc):
    xin, w, b = (T[k] for k in o["inputs"][:3])
    yout = T[o["outputs"][0]]
    s_in, z_in = qp(xin)
    s_w, _ = qp(w)
    s_out, z_out = qp(yout)
    Wq = it_q.get_tensor(w["index"])  # [out, in]
    Bq = it_q.get_tensor(b["index"])
    assert Wq.shape == (dims[i + 1], dims[i])
    tm.append(dense_decl(f"L{i}", Wq, Bq, s_in[0], s_w, s_out[0], z_in[0], z_out[0], relu[i]))
tm.append("""inline void forward(const int8_t* in, int8_t* out)
{
    static int8_t a0[32];
    static int8_t a1[16];
    L0.forward(in, a0);
    L1.forward(a0, a1);
    L2.forward(a1, out);
}

} // namespace tm_mlp
""")
write("mlp_tinymind.hpp", "".join(tm))

write("mlp_tflite_f32.h", HDR + flatbuffer_header("mlp_f32_tflite", mlp_f32))
write("mlp_tflite_int8.h", HDR + flatbuffer_header("mlp_int8_tflite", mlp_i8))

n = len(Xte)
write("mlp_data.h", HDR + "#pragma once\n#include <stdint.h>\n\n"
      + f"#define MLP_N_TEST {n}\n#define MLP_N_IN 64\n#define MLP_N_OUT 10\n"
      + f"#define MLP_INT8_HASH 0x{mlp_i8_hash:08X}u\n"
      + f"#define MLP_INT8_ACC_X10000 {int(round(acc_tflite_i8 * 10000))}\n\n"
      + "#ifdef MLP_DATA_INT8\n" + c_array("int8_t", "mlp_test_x_i8", Xte_q, align=4) + "#endif\n"
      + "#ifdef MLP_DATA_F32\n" + c_array("float", "mlp_test_x_f32", Xte, per_line=8) + "#endif\n"
      + c_array("uint8_t", "mlp_test_y", yte))

# ---------------------------------------------------------------- KWS

tf.keras.utils.set_random_seed(42)
kws = tf.keras.Sequential([
    tf.keras.Input(shape=(20, 20, 1), batch_size=1),
    tf.keras.layers.Conv2D(8, 3, activation="relu"),
    tf.keras.layers.MaxPooling2D(2),
    tf.keras.layers.DepthwiseConv2D(3, activation="relu"),
    tf.keras.layers.Conv2D(16, 1, activation="relu"),
    tf.keras.layers.AveragePooling2D(7),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(10),
])
# Random but non-degenerate biases so every requant path is exercised.
rng = np.random.default_rng(42)
for layer in kws.layers:
    wts = layer.get_weights()
    if len(wts) == 2:
        wts[1] = rng.uniform(-0.05, 0.05, wts[1].shape).astype(np.float32)
        layer.set_weights(wts)

rep = rng.uniform(-1, 1, (200, 20, 20, 1)).astype(np.float32)
kws_f32 = to_tflite(kws, rep, int8=False)
kws_i8 = to_tflite(kws, rep, int8=True)

itk = ref_interpreter(kws_i8)
ks, kz = qp(itk.get_input_details()[0])
KN = 8
kx = rng.uniform(-1, 1, (KN, 20, 20, 1)).astype(np.float32)
kx_q = np.clip(np.round(kx / ks[0]) + kz[0], -128, 127).astype(np.int8)
kyq = run_tflite(itk, kx_q)
kws_hash = fnv1a(kyq.astype(np.int8).tobytes())

itkf = ref_interpreter(kws_f32)
kyf = run_tflite(itkf, kx)
# float vs dequantized int8 agreement (argmax), just for the record
kws_argmax_agree = (np.argmax(kyf, 1) == np.argmax(kyq, 1)).mean()

op_list, T = ops(itk)
names = [o["op_name"] for o in op_list]
print("kws int8 ops:", names)
print("kws f32 ops:", [o["op_name"] for o in ops(itkf)[0]])


def get(o, k):
    return T[o["inputs"][k]]


conv = [o for o in op_list if o["op_name"] == "CONV_2D"]
dw = [o for o in op_list if o["op_name"] == "DEPTHWISE_CONV_2D"][0]
fcs = [o for o in op_list if o["op_name"] == "FULLY_CONNECTED"][0]
mp = [o for o in op_list if o["op_name"] == "MAX_POOL_2D"][0]
ap = [o for o in op_list if o["op_name"] == "AVERAGE_POOL_2D"][0]
assert len(conv) == 2


def perchannel(o, relu):
    xin, w, b = get(o, 0), get(o, 1), get(o, 2)
    yout = T[o["outputs"][0]]
    s_in, z_in = qp(xin)
    s_w, _ = qp(w)
    s_out, z_out = qp(yout)
    rqs = [requant(s_in[0], s, s_out[0], z_out[0], relu) for s in s_w]
    return itk.get_tensor(w["index"]), itk.get_tensor(b["index"]), int(z_in[0]), rqs


k = [HDR, "#pragma once\n#include <cstdint>\n#include \"qconv2d.hpp\"\n#include \"qdepthwiseconv2d.hpp\"\n"
     "#include \"qpointwiseconv2d.hpp\"\n#include \"qpool2d.hpp\"\n#include \"qdense.hpp\"\n\n"
     "namespace tm_kws {\n\nusing Rq = tinymind::Requantizer<int32_t, int8_t>;\n\n"]

# conv1: TFLite OHWI == tinymind [F][KH][KW][Cin]
W, B, zi, rqs = perchannel(conv[0], True)
assert W.shape == (8, 3, 3, 1)
k.append(c_array("int8_t", "Wc1", W, align=4) + c_array("int32_t", "Bc1", B))
k.append("const Rq Rc1[8] = {" + ", ".join(rq_init(r) for r in rqs) + "};\n")
k.append("using C1Type = tinymind::QConv2DPerChannel<int8_t, int8_t, int32_t, int8_t, 20, 20, 1, 3, 3, 1, 1, 8>;\n")
k.append(f"const C1Type C1 = {{Wc1, Bc1, {zi}, Rc1}};\n\n")

# maxpool: same quant in/out in TFLite, so plain max on int8
k.append("using P1Type = tinymind::QMaxPool2D<int8_t, 18, 18, 8, 2, 2, 2, 2>;\nconst P1Type P1 = {};\n\n")

# depthwise: TFLite [1][KH][KW][C] -> tinymind [C][KH][KW]
W, B, zi, rqs = perchannel(dw, True)
assert W.shape == (1, 3, 3, 8)
Wt = np.transpose(W[0], (2, 0, 1))
k.append(c_array("int8_t", "Wdw", Wt, align=4) + c_array("int32_t", "Bdw", B))
k.append("const Rq Rdw[8] = {" + ", ".join(rq_init(r) for r in rqs) + "};\n")
k.append("using DwType = tinymind::QDepthwiseConv2D<int8_t, int8_t, int32_t, int8_t, 9, 9, 8, 3, 3, 1, 1>;\n")
k.append(f"const DwType DW = {{Wdw, Bdw, {zi}, Rdw}};\n\n")

# pointwise: TFLite [16][1][1][8] -> tinymind [F][Cin]
W, B, zi, rqs = perchannel(conv[1], True)
assert W.shape == (16, 1, 1, 8)
k.append(c_array("int8_t", "Wpw", W, align=4) + c_array("int32_t", "Bpw", B))
k.append("const Rq Rpw[16] = {" + ", ".join(rq_init(r) for r in rqs) + "};\n")
k.append("using PwType = tinymind::QPointwiseConv2DPerChannel<int8_t, int8_t, int32_t, int8_t, 7, 7, 8, 16>;\n")
k.append(f"const PwType PW = {{Wpw, Bpw, {zi}, Rpw}};\n\n")

# avgpool 7x7 over 7x7 == global average; TFLite keeps in/out quant equal
s_a, z_a = qp(get(ap, 0))
s_ao, z_ao = qp(T[ap["outputs"][0]])
assert np.isclose(s_a[0], s_ao[0]) and z_a[0] == z_ao[0]
k.append("using GapType = tinymind::QGlobalAvgPool2D<int8_t, int32_t, 7, 7, 16>;\n"
         "const GapType GAP = {-128, 127};\n\n")

# dense: per-tensor
xin, w, b = get(fcs, 0), get(fcs, 1), get(fcs, 2)
s_in, z_in = qp(xin)
s_w, _ = qp(w)
s_out, z_out = qp(T[fcs["outputs"][0]])
Wd = itk.get_tensor(w["index"])
assert Wd.shape == (10, 16)
k.append(dense_decl("D", Wd, itk.get_tensor(b["index"]), s_in[0], s_w, s_out[0], z_in[0], z_out[0], False))
k.append("""inline void forward(const int8_t* in, int8_t* out)
{
    static int8_t c1[C1Type::OutputSize];
    static int8_t p1[P1Type::OutputSize];
    static int8_t dw[DwType::OutputSize];
    static int8_t pw[PwType::OutputSize];
    static int8_t gap[16];
    C1.forward(in, c1);
    P1.forward(c1, p1);
    DW.forward(p1, dw);
    PW.forward(dw, pw);
    GAP.forward(pw, gap);
    D.forward(gap, out);
}

} // namespace tm_kws
""")
write("kws_tinymind.hpp", "".join(k))
write("kws_tflite_int8.h", HDR + flatbuffer_header("kws_int8_tflite", kws_i8))
write("kws_tflite_f32.h", HDR + flatbuffer_header("kws_f32_tflite", kws_f32))
write("kws_data.h", HDR + "#pragma once\n#include <stdint.h>\n\n"
      + f"#define KWS_N_TEST {KN}\n#define KWS_N_IN 400\n#define KWS_N_OUT 10\n"
      + f"#define KWS_INT8_HASH 0x{kws_hash:08X}u\n\n"
      + "#ifdef KWS_DATA_INT8\n" + c_array("int8_t", "kws_test_x_i8", kx_q, align=4) + "#endif\n"
      + "#ifdef KWS_DATA_F32\n" + c_array("float", "kws_test_x_f32", kx, per_line=8) + "#endif\n")

summary = f"""mlp: sklearn acc {acc_sklearn:.4f} | keras acc {acc_keras:.4f} | tflite f32 acc {acc_tflite_f32:.4f} | tflite int8 acc {acc_tflite_i8:.4f}
mlp: tflite f32 {len(mlp_f32)} B, int8 {len(mlp_i8)} B, int8 hash 0x{mlp_i8_hash:08X}, input q scale {in_scale} zp {in_zp}
kws: tflite f32 {len(kws_f32)} B, int8 {len(kws_i8)} B, int8 hash 0x{kws_hash:08X}, f32/int8 argmax agree {kws_argmax_agree:.3f}
"""
write("summary.txt", summary)
print(summary)
