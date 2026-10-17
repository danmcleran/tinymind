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

// Compile-time model checks for QStaticDecisionTree / QStaticGBDT
// (cpp/qtree.hpp), driven by the qtree_static_regressions make target.
//
// Built clean, this is a freestanding program that runs both evaluation
// policies over every int8 input pair and exits non-zero on any disagreement
// with the run-time QDecisionTree / QGBDT. The target builds it at -std=c++14
// (the README's floor, where QTreeWalkEval needs the out-of-class pool
// definitions below) and at -std=c++17.
//
// Built with one QTREE_BAD_* macro, it plants one model defect that must be a
// compile error, and the target checks the diagnostic names it:
//
//   QTREE_BAD_FEATURE  split feature >= NumFeatures (out-of-bounds input read)
//   QTREE_BAD_CHILD    child index past the end of the pool
//   QTREE_BAD_CYCLE    a node that is its own child (inline expansion only)
//   QTREE_BAD_ROOT     GBDT tree_root past the end of the pool
//   QTREE_BAD_CLASS    GBDT tree_class >= NumClasses

#include "include/tinymind_platform.hpp"

#include <cstddef>
#include <cstdint>

#include "qtree.hpp"

namespace
{

typedef tinymind::QTreeNode<int8_t> Node;

// Depth-2 tree over 2 features, laid out with the children *before* their
// parent so the static types are not relying on a pre-order pool.
//   root (node 6): f0 <= 3 ? node4 : node5
//   node4: f1 <= -10 ? leaf0(10) : leaf1(20)
//   node5: f1 <=  40 ? leaf2(30) : leaf3(40)
struct TreePool
{
    static constexpr Node nodes[7] =
    {
        { -1, 0, -1, -1, 10 },
        { -1, 0, -1, -1, 20 },
        { -1, 0, -1, -1, 30 },
        { -1, 0, -1, -1, 40 },
#if defined(QTREE_BAD_FEATURE)
        {  2, -10, 0, 1, 0 },
#elif defined(QTREE_BAD_CHILD)
        {  1, -10, 0, 7, 0 },
#elif defined(QTREE_BAD_CYCLE)
        {  1, -10, 0, 4, 0 },
#else
        {  1, -10, 0, 1, 0 },
#endif
        {  1,  40, 2, 3, 0 },
        {  0,   3, 4, 5, 0 },
    };
};

// The same tree in pre-order with the root at node 0, the layout
// QStaticDecisionTree (root 0) and the run-time QDecisionTree expect.
struct RootFirstPool
{
    static constexpr Node nodes[7] =
    {
        {  0,   3, 1, 2, 0 },
        {  1, -10, 3, 4, 0 },
        {  1,  40, 5, 6, 0 },
        { -1, 0, -1, -1, 10 },
        { -1, 0, -1, -1, 20 },
        { -1, 0, -1, -1, 30 },
        { -1, 0, -1, -1, 40 },
    };
};

// Three stumps over a shared pool, two classes, base score {100, -7}.
struct GbdtPool
{
    static constexpr Node nodes[9] =
    {
        { 0,  0, 1, 2, 0 }, { -1, 0, -1, -1,  5 }, { -1, 0, -1, -1, -5 },
        { 1,  0, 4, 5, 0 }, { -1, 0, -1, -1,  8 }, { -1, 0, -1, -1, -8 },
        { 0, 50, 7, 8, 0 }, { -1, 0, -1, -1,  3 }, { -1, 0, -1, -1, -3 },
    };
#if defined(QTREE_BAD_ROOT)
    static constexpr int16_t tree_root[3]  = { 0, 3, 9 };
#else
    static constexpr int16_t tree_root[3]  = { 0, 3, 6 };
#endif
#if defined(QTREE_BAD_CLASS)
    static constexpr int16_t tree_class[3] = { 0, 2, 0 };
#else
    static constexpr int16_t tree_class[3] = { 0, 1, 0 };
#endif
    static constexpr int32_t base_score[2] = { 100, -7 };
};

} // namespace

#if TINYMIND_CPLUSPLUS < 201703L
// C++14: QTreeWalkEval reads the pools at run time, so they need definitions.
constexpr Node TreePool::nodes[];
constexpr Node RootFirstPool::nodes[];
constexpr Node GbdtPool::nodes[];
constexpr int16_t GbdtPool::tree_root[];
constexpr int16_t GbdtPool::tree_class[];
constexpr int32_t GbdtPool::base_score[];
#endif

namespace
{

// TreePool rooted at node 6, through the policies directly.
template<typename EvalPolicy>
int32_t treePoolLeaf(const int8_t* x)
{
    return EvalPolicy::template eval<TreePool, 6>(x);
}

} // namespace

int main()
{
    typedef tinymind::QStaticDecisionTree<int8_t, RootFirstPool, 2, tinymind::QTreeInlineEval> InlineTree;
    typedef tinymind::QStaticDecisionTree<int8_t, RootFirstPool, 2, tinymind::QTreeWalkEval>   WalkTree;
    typedef tinymind::QStaticGBDT<int8_t, GbdtPool, 2, 2, tinymind::QTreeInlineEval> InlineGbdt;
    typedef tinymind::QStaticGBDT<int8_t, GbdtPool, 2, 2, tinymind::QTreeWalkEval>   WalkGbdt;

    // Instantiating the type runs the pool validation on TreePool.
    typedef tinymind::QStaticDecisionTree<int8_t, TreePool, 2, tinymind::QTreeInlineEval> TreePoolCheck;
    static_assert(TreePoolCheck::NumNodes == 7, "TreePool size");

    tinymind::QDecisionTree<int8_t, int8_t, 7, 2> runtimeTree;
    runtimeTree.nodes = RootFirstPool::nodes;

    tinymind::QGBDT<int8_t, int8_t, 3, 2, 9, 2> runtimeGbdt;
    runtimeGbdt.nodes      = GbdtPool::nodes;
    runtimeGbdt.tree_root  = GbdtPool::tree_root;
    runtimeGbdt.tree_class = GbdtPool::tree_class;
    runtimeGbdt.base_score = GbdtPool::base_score;

    int mismatches = 0;
    for (int a = -128; a <= 127; ++a)
    {
        for (int b = -128; b <= 127; ++b)
        {
            const int8_t x[2] = { static_cast<int8_t>(a), static_cast<int8_t>(b) };

            const int32_t leaf = runtimeTree.predict(x);
            mismatches += (InlineTree::predict(x) != leaf) ? 1 : 0;
            mismatches += (WalkTree::predict(x) != leaf) ? 1 : 0;
            mismatches += (treePoolLeaf<tinymind::QTreeInlineEval>(x) != leaf) ? 1 : 0;
            mismatches += (treePoolLeaf<tinymind::QTreeWalkEval>(x) != leaf) ? 1 : 0;

            int32_t ref[2];
            int32_t got[2];
            runtimeGbdt.forward(x, ref);
            InlineGbdt::forward(x, got);
            mismatches += (got[0] != ref[0] || got[1] != ref[1]) ? 1 : 0;
            WalkGbdt::forward(x, got);
            mismatches += (got[0] != ref[0] || got[1] != ref[1]) ? 1 : 0;
            mismatches += (InlineGbdt::predict(x) != runtimeGbdt.predict(x)) ? 1 : 0;
            mismatches += (WalkGbdt::predict(x) != runtimeGbdt.predict(x)) ? 1 : 0;
        }
    }

    return (mismatches == 0) ? 0 : 1;
}
