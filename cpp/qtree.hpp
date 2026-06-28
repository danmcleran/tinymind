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

} // namespace tinymind
