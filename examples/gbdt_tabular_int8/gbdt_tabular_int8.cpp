/**
* Copyright (c) 2026 Dan McLeran
*
* Permission is hereby granted, free of charge, to any person obtaining a copy
* of this software and associated documentation files (the "Software"), to deal
* in the Software without restriction, including without limitation the rights
* to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
* copies of the Software, and to permit persons to whom the Software is
* furnished to do so, subject to the following conditions:
*
* The above copyright notice and this permission notice shall be included in all
* copies or substantial portions of the Software.
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
* IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
* FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
* AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
* LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
* OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
* SOFTWARE.
*/

// Int8 gradient-boosted decision tree (GBDT) ensemble on a 2-feature, 3-class
// tabular problem -- the cheapest model family on a microcontroller: each
// prediction is a handful of integer compares and a branch, no MACs.
//
// The ensemble is a small additive set of shallow trees (as exported from an
// XGBoost / LightGBM-style trainer): each tree contributes its leaf value to
// its target class's logit, and argmax over the per-class logits is the
// prediction. Thresholds are quantized onto the int8 feature grid; because an
// affine map is monotone, x_q <= t_q matches x_real <= t_real, so the int8
// tree reproduces the float tree's split decisions exactly (away from a sample
// landing within one grid step of a threshold).
//
// The driver evaluates the GBDT over a dense 2D grid in both float (real
// thresholds) and int8 (quantized thresholds), reports how often they agree,
// and writes the int8 decision-region map to a CSV for plotting.

#include "qaffine.hpp"
#include "qtree.hpp"
#include "include/qcalibration.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>

namespace {

constexpr std::size_t NF = 2;   // features
constexpr std::size_t NC = 3;   // classes
constexpr std::size_t G  = 48;  // grid resolution per axis

// A shallow tree authored in the float domain (as a trainer would export it):
// internal node branches left if x[feature] <= threshold. feature < 0 is a
// leaf with an int32 logit contribution.
struct FNode { int feature; float threshold; int left; int right; int32_t leaf; };

// 5-tree additive ensemble over 2 features. Vertical bands by f0 (class 0 left,
// class 1 middle, class 2 right) modulated by f1.
//   t0 (class0): f0 <= -0.15 ? +12 : -4
//   t1 (class1): f0 <= -0.15 ? -8 : (f0 <= 0.25 ? +12 : -8)
//   t2 (class2): f0 <=  0.25 ? -4 : +12
//   t3 (class0): f1 <= -0.10 ? +5 : -3
//   t4 (class2): f1 <=  0.10 ? -3 : +5
const FNode pool[] = {
    // t0  (root 0)
    { 0, -0.15f, 1, 2, 0 }, { -1, 0, -1, -1, 12 }, { -1, 0, -1, -1, -4 },
    // t1  (root 3)
    { 0, -0.15f, 4, 5, 0 }, { -1, 0, -1, -1, -8 },
    { 0,  0.25f, 6, 7, 0 }, { -1, 0, -1, -1, 12 }, { -1, 0, -1, -1, -8 },
    // t2  (root 8)
    { 0,  0.25f, 9, 10, 0 }, { -1, 0, -1, -1, -4 }, { -1, 0, -1, -1, 12 },
    // t3  (root 11)
    { 1, -0.10f, 12, 13, 0 }, { -1, 0, -1, -1, 5 }, { -1, 0, -1, -1, -3 },
    // t4  (root 14)
    { 1,  0.10f, 15, 16, 0 }, { -1, 0, -1, -1, -3 }, { -1, 0, -1, -1, 5 },
};
constexpr std::size_t POOL = sizeof(pool) / sizeof(pool[0]);
const int16_t roots[]   = {0, 3, 8, 11, 14};
const int16_t classes[] = {0, 1, 2, 0, 2};
constexpr std::size_t NT = sizeof(roots) / sizeof(roots[0]);

int floatWalk(int16_t root, const float* x)
{
    int n = root;
    while (pool[n].feature >= 0)
        n = (x[pool[n].feature] <= pool[n].threshold) ? pool[n].left : pool[n].right;
    return static_cast<int>(pool[n].leaf);
}
std::size_t floatPredict(const float* x)
{
    int32_t logits[NC] = {0, 0, 0};
    for (std::size_t t = 0; t < NT; ++t) logits[classes[t]] += floatWalk(roots[t], x);
    std::size_t best = 0;
    for (std::size_t c = 1; c < NC; ++c) if (logits[c] > logits[best]) best = c;
    return best;
}

} // namespace

int main(int argc, char** argv)
{
    const bool golden_mode = (argc >= 2) && std::strcmp(argv[1], "--golden") == 0;

    using tinymind::computeAffineParamsAsymmetric;
    using tinymind::quantize;

    // Shared feature grid over [-1, 1].
    const auto pg = computeAffineParamsAsymmetric(-1.0f, 1.0f, -128, 127);

    // Quantize the tree thresholds onto the feature grid.
    tinymind::QTreeNode<int8_t> qnodes[POOL];
    for (std::size_t i = 0; i < POOL; ++i)
    {
        qnodes[i].feature = static_cast<int16_t>(pool[i].feature);
        qnodes[i].left    = static_cast<int16_t>(pool[i].left);
        qnodes[i].right   = static_cast<int16_t>(pool[i].right);
        qnodes[i].leaf_value = pool[i].leaf;
        qnodes[i].threshold = (pool[i].feature < 0) ? 0
            : quantize<int8_t>(pool[i].threshold, pg.scale, pg.zero_point, -128, 127);
    }
    tinymind::QGBDT<int8_t, int8_t, NT, NC, POOL, NF> gbdt;
    gbdt.nodes = qnodes;
    gbdt.tree_root = roots;
    gbdt.tree_class = classes;
    gbdt.base_score = nullptr;

    // Sweep the 2D grid in float and int8; record the int8 region map.
    int8_t region[G * G];
    std::size_t agree = 0, total = 0;
    std::size_t hist[NC] = {0, 0, 0};
    for (std::size_t iy = 0; iy < G; ++iy)
        for (std::size_t ix = 0; ix < G; ++ix)
        {
            const float f0 = -1.0f + 2.0f * static_cast<float>(ix) / static_cast<float>(G - 1);
            const float f1 = -1.0f + 2.0f * static_cast<float>(iy) / static_cast<float>(G - 1);
            const float xf[NF] = {f0, f1};
            int8_t xq[NF] = {
                quantize<int8_t>(f0, pg.scale, pg.zero_point, -128, 127),
                quantize<int8_t>(f1, pg.scale, pg.zero_point, -128, 127),
            };
            const std::size_t c_q = gbdt.predict(xq);
            const std::size_t c_f = floatPredict(xf);
            region[iy * G + ix] = static_cast<int8_t>(c_q);
            ++hist[c_q];
            if (c_q == c_f) ++agree;
            ++total;
        }

    // CSV: f0, f1, int8 predicted class (for the region-map plot).
    {
        std::FILE* csv = std::fopen("gbdt_tabular_int8.csv", "w");
        std::fprintf(csv, "f0,f1,class\n");
        for (std::size_t iy = 0; iy < G; ++iy)
            for (std::size_t ix = 0; ix < G; ++ix)
            {
                const float f0 = -1.0f + 2.0f * static_cast<float>(ix) / static_cast<float>(G - 1);
                const float f1 = -1.0f + 2.0f * static_cast<float>(iy) / static_cast<float>(G - 1);
                std::fprintf(csv, "%.5f,%.5f,%d\n", f0, f1, static_cast<int>(region[iy * G + ix]));
            }
        std::fclose(csv);
    }

    // Fixed probe points for a compact, deterministic golden.
    const float probes[8][NF] = {
        {-0.8f, -0.8f}, {-0.8f, 0.8f}, {0.0f, -0.8f}, {0.0f, 0.8f},
        { 0.8f, -0.8f}, { 0.8f, 0.8f}, {-0.3f, 0.0f}, {0.5f, 0.0f},
    };

    if (golden_mode)
    {
        std::printf("# gbdt_tabular_int8 golden output\n");
        std::printf("# trees=%zu classes=%zu grid=%zu agree=%zu/%zu\n",
                    static_cast<size_t>(NT), static_cast<size_t>(NC),
                    static_cast<size_t>(G), agree, total);
        std::printf("probes:");
        for (std::size_t p = 0; p < 8; ++p)
        {
            int8_t xq[NF] = {
                quantize<int8_t>(probes[p][0], pg.scale, pg.zero_point, -128, 127),
                quantize<int8_t>(probes[p][1], pg.scale, pg.zero_point, -128, 127),
            };
            std::printf(" %zu", gbdt.predict(xq));
        }
        std::printf("\n");
        return 0;
    }

    std::printf("Int8 GBDT ensemble (tabular, %zu trees, %zu classes)\n",
                static_cast<size_t>(NT), static_cast<size_t>(NC));
    std::printf("  feature grid: scale=%.5f zp=%d   pool=%zu nodes\n",
                pg.scale, pg.zero_point, static_cast<size_t>(POOL));
    std::printf("  int8 vs float GBDT agreement over %zux%zu grid: %zu / %zu (%.2f%%)\n",
                static_cast<size_t>(G), static_cast<size_t>(G), agree, total,
                100.0 * static_cast<double>(agree) / static_cast<double>(total));
    std::printf("  predicted-class histogram: c0=%zu c1=%zu c2=%zu\n",
                hist[0], hist[1], hist[2]);
    std::printf("  probe predictions:");
    for (std::size_t p = 0; p < 8; ++p)
    {
        int8_t xq[NF] = {
            quantize<int8_t>(probes[p][0], pg.scale, pg.zero_point, -128, 127),
            quantize<int8_t>(probes[p][1], pg.scale, pg.zero_point, -128, 127),
        };
        std::printf(" (%+.1f,%+.1f)->%zu", probes[p][0], probes[p][1], gbdt.predict(xq));
    }
    std::printf("\n");

    // The int8 tree only disagrees with the float tree at a grid cell whose
    // feature lands within one int8 step (~0.008) of a split threshold; on a
    // dense sweep that boundary band is a couple percent of cells.
    const double frac = static_cast<double>(agree) / static_cast<double>(total);
    if (frac < 0.97)
    {
        std::printf("FAIL: int8/float agreement %.2f%% below 97%%\n", 100.0 * frac);
        return 1;
    }
    std::printf("PASS (int8 reproduces the float GBDT's regions; flips only at boundary cells)\n");
    return 0;
}
