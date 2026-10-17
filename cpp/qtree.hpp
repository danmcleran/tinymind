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

#pragma once

#include "include/tinymind_platform.hpp"

#include <cstddef>
#include <cstdint>

/*
 * Quantized decision trees + gradient-boosted ensemble (int8 inference).
 *
 * Tree inference is the cheapest model family on a microcontroller: a walk
 * from root to leaf is a handful of integer compares and a branch -- no MACs,
 * no matmul, no activation tables. The split feature is compared against an
 * int8 threshold on the same affine grid as the input feature; because an
 * affine map is monotone, x_q <= t_q is equivalent to x_real <= t_real, so
 * quantizing the threshold preserves the split exactly (no accuracy loss in
 * the comparison itself). Leaf values are int32 so a boosted ensemble can
 * accumulate them into per-class logits.
 *
 *   QDecisionTree  -- one tree: walk to a leaf, return its int32 value.
 *   QGBDT          -- additive ensemble: each tree contributes its leaf value
 *                     to its target class's logit; argmax (or the raw logits)
 *                     is the prediction. Single-tree / single-output regression
 *                     is the degenerate case (NumClasses == 1).
 *
 * Complements the neural-network family: trees win on tabular sensor data
 * where a dense net is overkill, and they share the deployment story (train
 * host-side, quantize thresholds onto the feature grid, ship inference-only).
 *
 * Nodes are a caller-owned flat array (a node pool shared across trees for the
 * ensemble). Pure integer at runtime; freestanding-safe (no float, no LUT).
 *
 * Compile-time models. When the trees are known at build time, the same node
 * pool can instead be a constexpr array inside a type (the "Pool"):
 *
 *   QStaticDecisionTree -- one tree from Pool::nodes, root at node 0.
 *   QStaticGBDT         -- the additive ensemble from Pool::nodes,
 *                          Pool::tree_root, Pool::tree_class and
 *                          Pool::base_score.
 *
 * An evaluation policy picks how the pool is run:
 *
 *   QTreeInlineEval -- every node is expanded at compile time into a compare
 *                      of x[feature] against an immediate threshold and a
 *                      branch, the nested if/else a code generator would
 *                      emit. Nothing of the pool is read at run time, so it is
 *                      not stored. Fastest; code size grows with node count.
 *   QTreeWalkEval   -- walkQTree over Pool::nodes at run time: one shared loop
 *                      and the pool as data. Smallest for large ensembles.
 *
 * Both give the same result for every input. The pool is validated at compile
 * time for either policy: every split feature must be < NumFeatures and every
 * child, root and class index must be in range, so a bad model is a build
 * error instead of an out-of-bounds read.
 *
 * Under C++14, QTreeWalkEval reads Pool::nodes at run time and so needs the
 * usual namespace-scope definition of the static member
 * (`constexpr QTreeNode<int8_t> MyPool::nodes[];`); C++17 makes it implicit.
 * QTreeInlineEval only reads the pool in constant expressions and needs none.
 */

namespace tinymind
{
    /**
     * One node of a quantized decision tree.
     *
     * feature >= 0 marks an internal node: branch left if
     * x[feature] <= threshold, else right (left/right index into the same node
     * array). feature < 0 marks a leaf, whose prediction is leaf_value.
     */
    template<typename ThreshStorage_>
    struct QTreeNode
    {
        typedef ThreshStorage_ ThreshType;

        int16_t     feature;     // < 0 => leaf
        ThreshType  threshold;
        int16_t     left;
        int16_t     right;
        int32_t     leaf_value;
    };

    /**
     * Walk a tree in a flat node pool from `root` to a leaf and return the
     * leaf value. The iteration is bounded by max_nodes so a malformed
     * (cyclic) tree terminates instead of spinning.
     */
    template<typename InStorage_, typename ThreshStorage_>
    inline int32_t walkQTree(const QTreeNode<ThreshStorage_>* nodes,
                             int16_t root, const InStorage_* x,
                             std::size_t max_nodes)
    {
        int16_t n = root;
        for (std::size_t guard = 0; guard < max_nodes; ++guard)
        {
            const QTreeNode<ThreshStorage_>& node = nodes[n];
            if (node.feature < 0)
            {
                return node.leaf_value;
            }
            const int32_t xv =
                static_cast<int32_t>(x[static_cast<std::size_t>(node.feature)]);
            const int32_t tv = static_cast<int32_t>(node.threshold);
            n = (xv <= tv) ? node.left : node.right;
        }
        return nodes[n].leaf_value;
    }

    /**
     * Single quantized decision tree. nodes[0] is the root.
     */
    template<typename InStorage_, typename ThreshStorage_,
             std::size_t NumNodes_, std::size_t NumFeatures_>
    struct QDecisionTree
    {
        typedef InStorage_     InputType;
        typedef ThreshStorage_ ThreshType;

        static constexpr std::size_t NumNodes    = NumNodes_;
        static constexpr std::size_t NumFeatures = NumFeatures_;

        const QTreeNode<ThreshStorage_>* nodes;

        int32_t predict(const InputType* x) const
        {
            return walkQTree<InputType, ThreshStorage_>(nodes, 0, x, NumNodes_);
        }

        static_assert(NumNodes_    > 0, "NumNodes must be > 0.");
        static_assert(NumFeatures_ > 0, "NumFeatures must be > 0.");
    };

    /**
     * Gradient-boosted decision-tree ensemble.
     *
     * NumTrees trees share one node pool. tree_root[t] is tree t's root index
     * into the pool; tree_class[t] is the class whose logit tree t contributes
     * to (0 for single-output regression). forward() accumulates every tree's
     * leaf value into its class logit, seeded by the optional per-class
     * base_score; predict() returns the argmax class.
     */
    template<typename InStorage_, typename ThreshStorage_,
             std::size_t NumTrees_, std::size_t NumClasses_,
             std::size_t NumNodes_, std::size_t NumFeatures_>
    struct QGBDT
    {
        typedef InStorage_     InputType;
        typedef ThreshStorage_ ThreshType;

        static constexpr std::size_t NumTrees    = NumTrees_;
        static constexpr std::size_t NumClasses  = NumClasses_;
        static constexpr std::size_t NumNodes    = NumNodes_;
        static constexpr std::size_t NumFeatures = NumFeatures_;

        const QTreeNode<ThreshStorage_>* nodes;   // shared pool
        const int16_t* tree_root;                 // [NumTrees]
        const int16_t* tree_class;                // [NumTrees], in [0, NumClasses)
        const int32_t* base_score;                // [NumClasses] or nullptr

        void forward(const InputType* x, int32_t* logits) const
        {
            for (std::size_t c = 0; c < NumClasses_; ++c)
            {
                logits[c] = (base_score != nullptr) ? base_score[c] : 0;
            }
            for (std::size_t t = 0; t < NumTrees_; ++t)
            {
                const int32_t leaf = walkQTree<InputType, ThreshStorage_>(
                    nodes, tree_root[t], x, NumNodes_);
                logits[static_cast<std::size_t>(tree_class[t])] += leaf;
            }
        }

        std::size_t predict(const InputType* x) const
        {
            int32_t logits[NumClasses_];
            forward(x, logits);
            std::size_t best = 0;
            for (std::size_t c = 1; c < NumClasses_; ++c)
            {
                if (logits[c] > logits[best]) best = c;
            }
            return best;
        }

        static_assert(NumTrees_    > 0, "NumTrees must be > 0.");
        static_assert(NumClasses_  > 0, "NumClasses must be > 0.");
        static_assert(NumNodes_    > 0, "NumNodes must be > 0.");
        static_assert(NumFeatures_ > 0, "NumFeatures must be > 0.");
    };

    /**
     * Deepest root-to-leaf path QTreeInlineEval expands. A cyclic pool would
     * otherwise recurse until the compiler's instantiation limit; this turns
     * it (or an absurdly deep tree) into a readable static_assert.
     */
    static constexpr std::size_t QTreeInlineMaxDepth = 64;

    namespace detail
    {
        template<typename Pool>
        constexpr std::size_t qtreePoolSize()
        {
            return sizeof(Pool::nodes) / sizeof(Pool::nodes[0]);
        }

        // Every internal node of the pool splits on a feature < NumFeatures
        // and points at children inside the pool.
        template<typename Pool, std::size_t NumFeatures>
        constexpr bool qtreePoolIsValid()
        {
            for (std::size_t i = 0; i < qtreePoolSize<Pool>(); ++i)
            {
                if (Pool::nodes[i].feature < 0)
                {
                    continue;
                }
                if (static_cast<std::size_t>(Pool::nodes[i].feature) >= NumFeatures
                    || Pool::nodes[i].left < 0
                    || static_cast<std::size_t>(Pool::nodes[i].left) >= qtreePoolSize<Pool>()
                    || Pool::nodes[i].right < 0
                    || static_cast<std::size_t>(Pool::nodes[i].right) >= qtreePoolSize<Pool>())
                {
                    return false;
                }
            }
            return true;
        }

        template<typename Pool, int16_t N>
        constexpr bool qtreeNodeIsLeaf()
        {
            return Pool::nodes[N].feature < 0;
        }

        // Node N of Pool, expanded at compile time. The node's fields are
        // copied into constants so the pool itself is never odr-used.
        template<typename Pool, int16_t N, std::size_t Depth,
                 bool IsLeaf = qtreeNodeIsLeaf<Pool, N>()>
        struct QTreeInlineNode;

        template<typename Pool, int16_t N, std::size_t Depth>
        struct QTreeInlineNode<Pool, N, Depth, true>
        {
            static constexpr int32_t leaf_value = Pool::nodes[N].leaf_value;

            template<typename InStorage_>
            static inline int32_t eval(const InStorage_*)
            {
                return leaf_value;
            }
        };

        template<typename Pool, int16_t N, std::size_t Depth>
        struct QTreeInlineNode<Pool, N, Depth, false>
        {
            static_assert(Depth < QTreeInlineMaxDepth,
                          "QTreeInlineEval: tree deeper than QTreeInlineMaxDepth (or cyclic).");

            static constexpr std::size_t feature   = static_cast<std::size_t>(Pool::nodes[N].feature);
            static constexpr int32_t     threshold = static_cast<int32_t>(Pool::nodes[N].threshold);
            static constexpr int16_t     left      = Pool::nodes[N].left;
            static constexpr int16_t     right     = Pool::nodes[N].right;

            template<typename InStorage_>
            static inline int32_t eval(const InStorage_* x)
            {
                if (static_cast<int32_t>(x[feature]) <= threshold)
                {
                    return QTreeInlineNode<Pool, left, Depth + 1>::eval(x);
                }
                return QTreeInlineNode<Pool, right, Depth + 1>::eval(x);
            }
        };

        // logits[c] = Pool::base_score[c] for c in [Lo, Lo + Count), split in
        // halves so the instantiation depth is logarithmic in the count.
        template<typename Pool, std::size_t Lo, std::size_t Count>
        struct QGBDTSeed
        {
            static inline void run(int32_t* logits)
            {
                QGBDTSeed<Pool, Lo, Count / 2>::run(logits);
                QGBDTSeed<Pool, Lo + Count / 2, Count - Count / 2>::run(logits);
            }
        };

        template<typename Pool, std::size_t Lo>
        struct QGBDTSeed<Pool, Lo, 1>
        {
            static constexpr int32_t base = Pool::base_score[Lo];

            static inline void run(int32_t* logits)
            {
                logits[Lo] = base;
            }
        };

        // Trees [Lo, Lo + Count) each add their leaf to their class's logit.
        template<typename Pool, typename EvalPolicy, std::size_t NumClasses,
                 std::size_t Lo, std::size_t Count>
        struct QGBDTAccumulate
        {
            template<typename InStorage_>
            static inline void run(const InStorage_* x, int32_t* logits)
            {
                QGBDTAccumulate<Pool, EvalPolicy, NumClasses, Lo, Count / 2>::run(x, logits);
                QGBDTAccumulate<Pool, EvalPolicy, NumClasses, Lo + Count / 2, Count - Count / 2>::run(x, logits);
            }
        };

        template<typename Pool, typename EvalPolicy, std::size_t NumClasses, std::size_t Lo>
        struct QGBDTAccumulate<Pool, EvalPolicy, NumClasses, Lo, 1>
        {
            static constexpr int16_t root = Pool::tree_root[Lo];
            static constexpr int16_t cls  = Pool::tree_class[Lo];

            static_assert(root >= 0 && static_cast<std::size_t>(root) < qtreePoolSize<Pool>(),
                          "QStaticGBDT: tree_root index out of range.");
            static_assert(cls >= 0 && static_cast<std::size_t>(cls) < NumClasses,
                          "QStaticGBDT: tree_class out of range.");

            template<typename InStorage_>
            static inline void run(const InStorage_* x, int32_t* logits)
            {
                logits[static_cast<std::size_t>(cls)] += EvalPolicy::template eval<Pool, root>(x);
            }
        };
    } // namespace detail

    /**
     * Evaluation policy: expand the pool into compare-and-branch code at
     * compile time.
     */
    struct QTreeInlineEval
    {
        template<typename Pool, int16_t Root, typename InStorage_>
        static inline int32_t eval(const InStorage_* x)
        {
            return detail::QTreeInlineNode<Pool, Root, 0>::eval(x);
        }
    };

    /**
     * Evaluation policy: walk the pool at run time with walkQTree.
     */
    struct QTreeWalkEval
    {
        template<typename Pool, int16_t Root, typename InStorage_>
        static inline int32_t eval(const InStorage_* x)
        {
            return walkQTree(Pool::nodes, Root, x, detail::qtreePoolSize<Pool>());
        }
    };

    /**
     * Single decision tree known at compile time. Pool::nodes is a constexpr
     * QTreeNode array; nodes[0] is the root.
     */
    template<typename InStorage_, typename Pool_, std::size_t NumFeatures_,
             typename EvalPolicy_ = QTreeInlineEval>
    struct QStaticDecisionTree
    {
        typedef InStorage_ InputType;
        typedef Pool_      PoolType;
        typedef EvalPolicy_ EvalPolicy;

        static constexpr std::size_t NumNodes    = detail::qtreePoolSize<Pool_>();
        static constexpr std::size_t NumFeatures = NumFeatures_;

        static int32_t predict(const InputType* x)
        {
            return EvalPolicy_::template eval<Pool_, 0>(x);
        }

        static_assert(NumFeatures_ > 0, "NumFeatures must be > 0.");
        static_assert(detail::qtreePoolIsValid<Pool_, NumFeatures_>(),
                      "QStaticDecisionTree: a split feature is >= NumFeatures or a child index is out of range.");
    };

    /**
     * Gradient-boosted ensemble known at compile time. Pool_ provides constexpr
     * arrays: nodes (the shared node pool), tree_root and tree_class (one
     * entry per tree) and base_score (NumClasses entries; zeros for none).
     * Same semantics as QGBDT.
     */
    template<typename InStorage_, typename Pool_, std::size_t NumClasses_,
             std::size_t NumFeatures_, typename EvalPolicy_ = QTreeInlineEval>
    struct QStaticGBDT
    {
        typedef InStorage_ InputType;
        typedef Pool_      PoolType;
        typedef EvalPolicy_ EvalPolicy;

        static constexpr std::size_t NumTrees    = sizeof(Pool_::tree_root) / sizeof(Pool_::tree_root[0]);
        static constexpr std::size_t NumClasses  = NumClasses_;
        static constexpr std::size_t NumNodes    = detail::qtreePoolSize<Pool_>();
        static constexpr std::size_t NumFeatures = NumFeatures_;

        static void forward(const InputType* x, int32_t* logits)
        {
            detail::QGBDTSeed<Pool_, 0, NumClasses_>::run(logits);
            detail::QGBDTAccumulate<Pool_, EvalPolicy_, NumClasses_, 0, NumTrees>::run(x, logits);
        }

        static std::size_t predict(const InputType* x)
        {
            int32_t logits[NumClasses_];
            forward(x, logits);
            std::size_t best = 0;
            for (std::size_t c = 1; c < NumClasses_; ++c)
            {
                if (logits[c] > logits[best]) best = c;
            }
            return best;
        }

        static_assert(NumTrees > 0, "QStaticGBDT: Pool::tree_root must name at least one tree.");
        static_assert(sizeof(Pool_::tree_class) / sizeof(Pool_::tree_class[0]) == NumTrees,
                      "QStaticGBDT: Pool::tree_class must have one entry per tree.");
        static_assert(sizeof(Pool_::base_score) / sizeof(Pool_::base_score[0]) == NumClasses_,
                      "QStaticGBDT: Pool::base_score must have NumClasses entries.");
        static_assert(NumClasses_  > 0, "NumClasses must be > 0.");
        static_assert(NumFeatures_ > 0, "NumFeatures must be > 0.");
        static_assert(detail::qtreePoolIsValid<Pool_, NumFeatures_>(),
                      "QStaticGBDT: a split feature is >= NumFeatures or a child index is out of range.");
    };

#if TINYMIND_CPLUSPLUS < 201703L
    // Before C++17 a static constexpr member used by reference (std::max, a
    // test macro, binding to const&) needs a definition; C++17 makes them
    // implicitly inline.
    template<typename InStorage_, typename ThreshStorage_, std::size_t NumNodes_, std::size_t NumFeatures_>
    constexpr std::size_t QDecisionTree<InStorage_, ThreshStorage_, NumNodes_, NumFeatures_>::NumNodes;
    template<typename InStorage_, typename ThreshStorage_, std::size_t NumNodes_, std::size_t NumFeatures_>
    constexpr std::size_t QDecisionTree<InStorage_, ThreshStorage_, NumNodes_, NumFeatures_>::NumFeatures;

    template<typename InStorage_, typename ThreshStorage_, std::size_t NumTrees_, std::size_t NumClasses_,
             std::size_t NumNodes_, std::size_t NumFeatures_>
    constexpr std::size_t QGBDT<InStorage_, ThreshStorage_, NumTrees_, NumClasses_, NumNodes_, NumFeatures_>::NumTrees;
    template<typename InStorage_, typename ThreshStorage_, std::size_t NumTrees_, std::size_t NumClasses_,
             std::size_t NumNodes_, std::size_t NumFeatures_>
    constexpr std::size_t QGBDT<InStorage_, ThreshStorage_, NumTrees_, NumClasses_, NumNodes_, NumFeatures_>::NumClasses;
    template<typename InStorage_, typename ThreshStorage_, std::size_t NumTrees_, std::size_t NumClasses_,
             std::size_t NumNodes_, std::size_t NumFeatures_>
    constexpr std::size_t QGBDT<InStorage_, ThreshStorage_, NumTrees_, NumClasses_, NumNodes_, NumFeatures_>::NumNodes;
    template<typename InStorage_, typename ThreshStorage_, std::size_t NumTrees_, std::size_t NumClasses_,
             std::size_t NumNodes_, std::size_t NumFeatures_>
    constexpr std::size_t QGBDT<InStorage_, ThreshStorage_, NumTrees_, NumClasses_, NumNodes_, NumFeatures_>::NumFeatures;

    template<typename InStorage_, typename Pool_, std::size_t NumFeatures_, typename EvalPolicy_>
    constexpr std::size_t QStaticDecisionTree<InStorage_, Pool_, NumFeatures_, EvalPolicy_>::NumNodes;
    template<typename InStorage_, typename Pool_, std::size_t NumFeatures_, typename EvalPolicy_>
    constexpr std::size_t QStaticDecisionTree<InStorage_, Pool_, NumFeatures_, EvalPolicy_>::NumFeatures;

    template<typename InStorage_, typename Pool_, std::size_t NumClasses_, std::size_t NumFeatures_, typename EvalPolicy_>
    constexpr std::size_t QStaticGBDT<InStorage_, Pool_, NumClasses_, NumFeatures_, EvalPolicy_>::NumTrees;
    template<typename InStorage_, typename Pool_, std::size_t NumClasses_, std::size_t NumFeatures_, typename EvalPolicy_>
    constexpr std::size_t QStaticGBDT<InStorage_, Pool_, NumClasses_, NumFeatures_, EvalPolicy_>::NumClasses;
    template<typename InStorage_, typename Pool_, std::size_t NumClasses_, std::size_t NumFeatures_, typename EvalPolicy_>
    constexpr std::size_t QStaticGBDT<InStorage_, Pool_, NumClasses_, NumFeatures_, EvalPolicy_>::NumNodes;
    template<typename InStorage_, typename Pool_, std::size_t NumClasses_, std::size_t NumFeatures_, typename EvalPolicy_>
    constexpr std::size_t QStaticGBDT<InStorage_, Pool_, NumClasses_, NumFeatures_, EvalPolicy_>::NumFeatures;
#endif

} // namespace tinymind
