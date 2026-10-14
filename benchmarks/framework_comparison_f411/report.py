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

"""Turn results/raw_<LEVEL>.txt into results/REPORT.md (F411, the default), or
results/raw_l031_<LEVEL>.txt into results/REPORT_l031.md with `report.py l031`.

flash = image flash - baseline image flash - embedded test-set bytes
ram   = (.data+.bss - baseline) + (peak stack - baseline peak stack)

Images that do not fit the board (`make fit`) get flash and static RAM from
their unbounded link and no cycles; their RAM excludes stack, so it is a floor.
"""
import subprocess
import sys

BOARDS = {
    "f411": {"raw": "results/raw_{lv}.txt", "build": "build/{lv}", "mhz": 100, "out": "results/REPORT.md"},
    "l031": {"raw": "results/raw_l031_{lv}.txt", "build": "build/l031/{lv}", "mhz": 32, "out": "results/REPORT_l031.md"},
}
BOARD = BOARDS[sys.argv[1] if len(sys.argv) > 1 else "f411"]
LEVELS = ["O2", "O3"]
TEST_SYMS = ("mlp_test_x_i8", "mlp_test_x_f32", "mlp_test_y", "kws_test_x_i8", "kws_test_x_f32")


def parse(level):
    recs, sizes, cur = {}, {}, None
    for line in open(BOARD["raw"].format(lv=level)):
        line = line.strip()
        if line.startswith("size "):
            kv = dict(p.split("=") for p in line.split()[1:])
            sizes[kv["image"]] = (int(kv["flash"]), int(kv["ram"]), kv.get("fits", "yes") == "yes")
        elif line.startswith("image="):
            cur = line.split("=", 1)[1]
            recs[cur] = {"layers": []}
        elif cur and "=" in line and not line.startswith("#"):
            k, v = line.split("=", 1)
            if k.startswith("layer."):
                _, idx, tag = k.split(".", 2)
                recs[cur]["layers"].append((tag, int(v)))
            else:
                recs[cur][k] = v
    return recs, sizes


def test_bytes(elf):
    out = subprocess.run(["arm-none-eabi-nm", "-S", "-C", elf], capture_output=True, text=True).stdout
    tot = 0
    for ln in out.splitlines():
        parts = ln.split()
        if len(parts) == 4 and parts[3] in TEST_SYMS:
            tot += int(parts[1], 16)
    return tot


NICE = {
    "tm_mlp": "tinymind int8 (scalar)", "tm_mlp_dsp": "tinymind int8 (+SMLAD)",
    "tm_kws": "tinymind int8 (scalar)", "tm_kws_dsp": "tinymind int8 (+SMLAD)",
    "tflm_cmsis_mlp_int8": "TFLM int8 (CMSIS-NN)", "tflm_ref_mlp_int8": "TFLM int8 (reference)",
    "tflm_cmsis_kws_int8": "TFLM int8 (CMSIS-NN)", "tflm_ref_kws_int8": "TFLM int8 (reference)",
    "tflm_cmsis_mlp_f32": "TFLM float32", "tflm_cmsis_kws_f32": "TFLM float32",
    "eml_mlp": "emlearn float32",
}
ORDER = {
    "mlp": ["tm_mlp", "tm_mlp_dsp", "tflm_cmsis_mlp_int8", "tflm_ref_mlp_int8", "tflm_cmsis_mlp_f32", "eml_mlp"],
    "kws": ["tm_kws", "tm_kws_dsp", "tflm_cmsis_kws_int8", "tflm_ref_kws_int8", "tflm_cmsis_kws_f32"],
}

data = {lv: parse(lv) for lv in LEVELS}
out = []
mhz = BOARD["mhz"]
n_mlp = data["O2"][0]["tm_mlp"]["n_test"]
for model, title in [("mlp", f"MLP 64-32-16-10, sklearn digits ({n_mlp} test samples)"),
                     ("kws", "KWS CNN 20x20x1 -> conv3x3x8 -> maxpool -> dw3x3 -> pw16 -> avgpool -> fc10")]:
    out.append(f"## {title}\n")
    out.append(f"| Framework | O2 cycles | O3 cycles | O3 us @{mhz}MHz | flash O2 / O3 (B) | RAM (B) | "
               + ("acc | " if model == "mlp" else "") + "int8 bit-exact |")
    out.append("|---|---:|---:|---:|---:|---:|" + ("---:|" if model == "mlp" else "") + "---|")
    for img in ORDER[model]:
        if img not in data["O2"][1]:
            continue
        if not data["O2"][1][img][2]:
            fl, ram = [], 0
            for lv in LEVELS:
                sizes = data[lv][1]
                elf = BOARD["build"].format(lv=lv) + f"/fit/{img}.elf"
                fl.append(sizes[img][0] - sizes["baseline"][0] - test_bytes(elf))
                ram = max(ram, sizes[img][1] - sizes["baseline"][1])
            acc = "- | " if model == "mlp" else ""
            out.append(f"| {NICE[img]} | does not fit | does not fit | - | {fl[0]:,} / {fl[1]:,} | {ram:,}+ | {acc}- |")
            continue
        fl, ram = [], None
        cyc = []
        for lv in LEVELS:
            recs, sizes = data[lv]
            r, b = recs[img], recs["baseline"]
            cyc.append(int(r["infer_med"]))
            elf = BOARD["build"].format(lv=lv) + f"/{img}.elf"
            fl.append(sizes[img][0] - sizes["baseline"][0] - test_bytes(elf))
            ram_lv = (int(r["static_ram"]) - int(b["static_ram"])) + (int(r["stack_bytes"]) - int(b["stack_bytes"]))
            ram = ram_lv if ram is None else max(ram, ram_lv)
        r3 = data["O3"][0][img]
        acc = f"{int(r3['correct']) / int(r3['n_test']) * 100:.2f}% | " if model == "mlp" else ""
        exact = "yes" if r3.get("hash_match") == "1" else ("n/a (float)" if "hash_match" not in r3 else "NO")
        out.append(f"| {NICE[img]} | {cyc[0]:,} | {cyc[1]:,} | {cyc[1] / mhz:,.0f} | {fl[0]:,} / {fl[1]:,} | {ram:,} | {acc}{exact} |")
    out.append("")
    # per-layer
    out.append(f"### Per-layer cycles ({model}, each layer timed separately)\n")
    imgs, heads = zip(*[(i, h) for i, h in [(f"tm_{model}_prof", "tinymind scalar"),
                                            (f"tm_{model}_dsp_prof", "tinymind +SMLAD"),
                                            (f"tflm_cmsis_{model}_int8_prof", "TFLM CMSIS-NN")]
                        if i in data["O2"][0]])
    out.append("| Layer | " + " | ".join(f"{h} O2 | {h} O3" for h in heads) + " |")
    out.append("|---|" + "---:|" * (2 * len(heads)))
    n = len(data["O2"][0][imgs[0]]["layers"])
    for i in range(n):
        cells = []
        for img in imgs:
            for lv in LEVELS:
                L = data[lv][0][img]["layers"]
                cells.append(f"{L[i][1]:,}" if i < len(L) else "-")
        tag = data["O2"][0][imgs[0]]["layers"][i][0]
        out.append(f"| {tag} | " + " | ".join(cells) + " |")
    out.append("")

open(BOARD["out"], "w").write("\n".join(out) + "\n")
print("\n".join(out))
