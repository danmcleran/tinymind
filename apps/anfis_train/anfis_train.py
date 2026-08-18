#
# Copyright (c) 2026 Dan McLeran
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#

"""Host-side ANFIS trainer for TinyMind's cpp/anfis.hpp (Phase C).

The device side is inference only: cpp/anfis.hpp takes frozen premise
parameters, a rule table, and consequent parameters. This module produces
those three arrays.

Training follows Jang's hybrid scheme, because the two parameter sets are
not alike:

  * The consequents enter the output *linearly* given fixed premises, so
    they are solved exactly by least squares -- one pseudo-inverse, no
    iteration, no learning rate.
  * The premises do not, so they descend on the gradient of the same
    squared error.

Alternating the two converges far faster than descending on everything,
which is the whole point of the hybrid rule. It is also why training does
not belong on a microcontroller: a pseudo-inverse over a
(samples x rules*(inputs+1)) design matrix is not an MCU workload.

Membership function shape is a policy, mirroring cpp/anfis.hpp:

  * BellMembership       -> GeneralizedBellMembershipFunction<ValueType, 1>
  * TriangularMembership -> TriangularMembershipFunction<ValueType>

Neither needs a transcendental function, so a model trained here deploys
into the freestanding (TINYMIND_ENABLE_FLOAT=0 / TINYMIND_ENABLE_STD=0)
corner unchanged.

Two ways to lay out the rule base:

  * build_grid_anfis     -- the Cartesian grid, M^N rules
  * build_scatter_anfis  -- one rule per subtractive-clustering center, so
                            the rule count follows the data rather than the
                            input count

numpy only -- no torch, no scipy.
"""

import numpy as np


# ---------------------------------------------------------------------------
# Membership function policies
# ---------------------------------------------------------------------------
#
# Each policy provides:
#   NumberOfParameters  parameters per membership function
#   parameter_names     names, in stored order (documents the emitted header)
#   cpp_type            the matching cpp/anfis.hpp policy
#   evaluate(p, x)      -> mu, broadcasting p against x
#   gradients(p, x)     -> d(mu)/d(parameter), stacked on a trailing axis
#   initialize(X, M)    -> [inputs, M, NumberOfParameters] grid partition
#   project(p, eps)     -> in-place repair of any constraint descent broke


# Mirrors the clamp in cpp/anfis.hpp: the device reports mu = 0 once
# u^2 reaches this, which both keeps the rule base sparse and stops a
# narrow Q format from overflowing. Applied here so the host model and the
# device model agree on the tails.
BELL_CLAMP = 64.0


class BellMembership:
    """Generalized bell with the exponent pinned at 1.

        u     = (x - c) / a
        mu(x) = 1 / (1 + u^2)

    Smooth everywhere and non-zero everywhere inside the clamp, which makes
    it the shape to train with: the premise gradient never vanishes just
    because a membership function drifted away from the data.
    """

    NumberOfParameters = 2
    parameter_names = ("a", "c")
    cpp_type = "GeneralizedBellMembershipFunction<ValueType, 1>"
    name = "bell"

    @staticmethod
    def evaluate(p, x):
        a = p[..., 0]
        c = p[..., 1]
        u = (x - c) / a
        t = u * u
        return np.where(t >= BELL_CLAMP, 0.0, 1.0 / (1.0 + t))

    @staticmethod
    def gradients(p, x):
        """d(mu)/da and d(mu)/dc.

            d(mu)/du = -2u / (1 + u^2)^2
            du/dc    = -1/a   ->  d(mu)/dc = 2u   / (a (1 + u^2)^2)
            du/da    = -u/a   ->  d(mu)/da = 2u^2 / (a (1 + u^2)^2)

        Both are zero past the clamp, matching `evaluate`.
        """
        a = p[..., 0]
        c = p[..., 1]
        u = (x - c) / a
        t = u * u
        denom = (1.0 + t) ** 2
        live = t < BELL_CLAMP
        d_da = np.where(live, 2.0 * t / (a * denom), 0.0)
        d_dc = np.where(live, 2.0 * u / (a * denom), 0.0)
        return np.stack([d_da, d_dc], axis=-1)

    @staticmethod
    def initialize(X, n_mfs):
        lo = X.min(axis=0)
        hi = X.max(axis=0)
        span = np.where(hi > lo, hi - lo, 1.0)
        n_inputs = X.shape[1]

        p = np.zeros((n_inputs, n_mfs, 2))
        step = span / max(n_mfs - 1, 1)
        for i in range(n_inputs):
            p[i, :, 0] = step[i] * 0.5                       # width
            p[i, :, 1] = np.linspace(lo[i], hi[i], n_mfs)    # centers
        return p

    @staticmethod
    def initialize_at(centers, widths):
        """Place one bell per cluster center. centers is [clusters, inputs]."""
        n_clusters, n_inputs = centers.shape
        p = np.zeros((n_inputs, n_clusters, 2))
        for i in range(n_inputs):
            p[i, :, 0] = max(widths[i] * 0.5, 1e-3)   # width
            p[i, :, 1] = centers[:, i]                # center
        return p

    @staticmethod
    def project(p, eps):
        # A non-positive width is not a bell.
        np.maximum(p[..., 0], eps, out=p[..., 0])


class TriangularMembership:
    """Triangle with feet at a and c and its peak at b.

        mu(x) = 0                  x <= a or x >= c
                (x - a) / (b - a)  a <  x <  b
                1                  x == b
                (c - x) / (c - b)  b <  x <  c

    Compact support, so most rules contribute nothing for most inputs --
    cheap on the device (cpp/anfis.hpp bails out of the product t-norm on
    the first zero grade) but harder to train: a triangle that drifts off
    the data has an exactly zero gradient and can never come back. Prefer
    BellMembership for the descent and switch to triangles only if the
    deployment target wants the compact support.
    """

    NumberOfParameters = 3
    parameter_names = ("a", "b", "c")
    cpp_type = "TriangularMembershipFunction<ValueType>"
    name = "triangular"

    @staticmethod
    def evaluate(p, x):
        a = p[..., 0]
        b = p[..., 1]
        c = p[..., 2]

        rising = np.divide(x - a, b - a, out=np.zeros_like(x * a),
                           where=(b - a) > 0.0)
        falling = np.divide(c - x, c - b, out=np.zeros_like(x * a),
                            where=(c - b) > 0.0)

        mu = np.where(x < b, rising, np.where(x > b, falling, 1.0))
        mu = np.where((x <= a) | (x >= c), 0.0, mu)
        return np.clip(mu, 0.0, 1.0)

    @staticmethod
    def gradients(p, x):
        """d(mu)/da, d(mu)/db, d(mu)/dc.

        On the rising flank, mu = (x - a) / (b - a):
            d(mu)/da = (x - b) / (b - a)^2      (negative: the ramp shifts right)
            d(mu)/db = -(x - a) / (b - a)^2     (negative: the ramp widens)
            d(mu)/dc = 0

        On the falling flank, mu = (c - x) / (c - b):
            d(mu)/db = (c - x) / (c - b)^2
            d(mu)/dc = (x - b) / (c - b)^2
            d(mu)/da = 0

        Outside the support every derivative is zero, and the two kinks (at
        the peak and at each foot) are measure-zero, so they are reported as
        zero rather than as a one-sided derivative.
        """
        a = p[..., 0]
        b = p[..., 1]
        c = p[..., 2]

        zero = np.zeros_like(x * a)
        inside = (x > a) & (x < c)
        rising = inside & (x < b)
        falling = inside & (x > b)

        left = np.divide(1.0, (b - a) ** 2, out=np.zeros_like(zero),
                         where=(b - a) > 0.0)
        right = np.divide(1.0, (c - b) ** 2, out=np.zeros_like(zero),
                          where=(c - b) > 0.0)

        d_da = np.where(rising, (x - b) * left, 0.0)
        d_db = np.where(rising, -(x - a) * left, 0.0) + \
               np.where(falling, (c - x) * right, 0.0)
        d_dc = np.where(falling, (x - b) * right, 0.0)
        return np.stack([d_da, d_db, d_dc], axis=-1)

    @staticmethod
    def initialize(X, n_mfs):
        """Partition of unity: peaks evenly spaced, feet at the neighbours.

        Adjacent triangles then cross at mu = 0.5 and sum to 1 across the
        observed range, which starts training from a sensible partition.
        """
        lo = X.min(axis=0)
        hi = X.max(axis=0)
        span = np.where(hi > lo, hi - lo, 1.0)
        n_inputs = X.shape[1]

        p = np.zeros((n_inputs, n_mfs, 3))
        step = span / max(n_mfs - 1, 1)
        for i in range(n_inputs):
            centers = np.linspace(lo[i], hi[i], n_mfs)
            p[i, :, 0] = centers - step[i]
            p[i, :, 1] = centers
            p[i, :, 2] = centers + step[i]
        return p

    @staticmethod
    def initialize_at(centers, widths):
        """Place one triangle per cluster center, feet at +/- the width."""
        n_clusters, n_inputs = centers.shape
        p = np.zeros((n_inputs, n_clusters, 3))
        for i in range(n_inputs):
            w = max(widths[i], 1e-3)
            p[i, :, 0] = centers[:, i] - w
            p[i, :, 1] = centers[:, i]
            p[i, :, 2] = centers[:, i] + w
        return p

    @staticmethod
    def project(p, eps):
        # Descent can push the three knots out of order, which would invert
        # a flank or divide by zero. Restore a <= b <= c with a minimum
        # separation, sweeping left to right.
        p[..., 1] = np.maximum(p[..., 1], p[..., 0] + eps)
        p[..., 2] = np.maximum(p[..., 2], p[..., 1] + eps)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

DONT_CARE = 255


class Anfis:
    """Takagi-Sugeno ANFIS with first-order consequents.

    Parameter layout matches cpp/anfis.hpp exactly, so `premise_flat()`,
    `rule_table_flat()`, and `consequent_flat()` can be emitted verbatim:

      premise     [n_inputs][n_mfs][membership function parameters]
      rule table  [n_rules][n_inputs]  -> membership function index
      consequent  [n_rules][n_outputs][p_0 .. p_{n-1}, q]

    Only n_outputs == 1 is trained here; the layout carries the output axis
    so the emitted header matches the C++ indexing without a special case.
    """

    def __init__(self, n_inputs, n_mfs, rules, params, mf=BellMembership):
        self.n_inputs = int(n_inputs)
        self.n_mfs = int(n_mfs)
        self.mf = mf
        self.rules = np.asarray(rules, dtype=np.int32)     # [R, n_inputs]
        self.params = np.asarray(params, dtype=np.float64)  # [I, M, P]
        self.theta = np.zeros(self.n_rules * (self.n_inputs + 1))

        if self.params.shape != (self.n_inputs, self.n_mfs, mf.NumberOfParameters):
            raise ValueError(
                "premise parameters must be [%d, %d, %d] for %s, got %s"
                % (self.n_inputs, self.n_mfs, mf.NumberOfParameters,
                   mf.name, self.params.shape))

    @property
    def n_rules(self):
        return self.rules.shape[0]

    # -- forward ------------------------------------------------------------

    def memberships(self, X):
        """mu[sample, input, mf] for a batch X of shape [N, n_inputs]."""
        return self.mf.evaluate(self.params[None, :, :, :], X[:, :, None])

    def firing_strengths(self, X):
        """Raw product-t-norm firing strengths, shape [N, n_rules].

        A DONT_CARE antecedent contributes a grade of 1, matching the
        device: cpp/anfis.hpp skips any index at or above the membership
        function count.
        """
        mu = self.memberships(X)                            # [N, I, M]
        w = np.ones((X.shape[0], self.n_rules))
        for i in range(self.n_inputs):
            idx = self.rules[:, i]
            live = idx < self.n_mfs
            if not np.any(live):
                continue
            grades = mu[:, i, :][:, idx[live]]              # [N, live]
            w[:, live] *= grades
        return w

    def design_matrix(self, X):
        """Rows of the least-squares system for the consequent parameters.

        Column block r holds [wbar_r * x_0 ... wbar_r * x_{n-1}, wbar_r],
        so `A @ theta` reproduces the defuzzified output exactly.
        """
        w = self.firing_strengths(X)
        total = w.sum(axis=1, keepdims=True)
        wbar = np.divide(w, total, out=np.zeros_like(w), where=total > 0.0)

        ones = np.ones((X.shape[0], 1))
        Xa = np.hstack([X, ones])                           # [N, n_inputs+1]
        # [N, R, n_inputs+1] -> [N, R*(n_inputs+1)]
        A = (wbar[:, :, None] * Xa[:, None, :])
        return A.reshape(X.shape[0], -1), w, wbar

    def predict(self, X):
        A, _, _ = self.design_matrix(X)
        return A @ self.theta

    # -- training -----------------------------------------------------------

    def solve_consequents(self, X, y, ridge=0.0):
        """Least-squares step for the consequent parameters.

        With ridge == 0 this is the exact ordinary least-squares solution --
        Jang's original hybrid rule.

        A non-zero `ridge` adds an L2 penalty, and it is worth understanding
        why that matters beyond the usual overfitting argument: it is what
        makes a model deployable in int8.

        The unregularized solve is free to produce very large consequent
        coefficients that nearly cancel each other -- on the bundled
        Mackey-Glass benchmark the largest is 169 while the model's output
        spans about 0.94. In float that is harmless. In int8 it is not: the
        input carries a quantization error of half a grid step, and a
        coefficient of 169 amplifies it into an error far larger than the
        output quantum, breaking the cancellation the fit depends on.

        Measured on that benchmark, ridge = 1e-6 drops the largest coefficient
        from 169 to 16, costs about 20% float accuracy, and improves int8
        accuracy 4.7x. If the model is headed for cpp/qanfis.hpp, use it.
        """
        A, _, _ = self.design_matrix(X)

        if ridge > 0.0:
            n = A.shape[1]
            self.theta = np.linalg.solve(
                A.T @ A + ridge * np.eye(n), A.T @ y)
        else:
            self.theta, *_ = np.linalg.lstsq(A, y, rcond=None)

        return self.theta

    def premise_gradients(self, X, y):
        """dE/d(premise parameters) for E = mean((y_hat - y)^2), shape [I, M, P].

        With W = sum_r w_r and y_hat = sum_r w_r f_r / W:

            dy_hat/dw_r      = (f_r - y_hat) / W
            dw_r/dmu_{i,j}   = w_r / mu_{i,j}   for the antecedent (i, j)

        The division by mu is guarded: a rule whose grade underflowed to
        zero contributes no gradient through that antecedent.
        """
        n = X.shape[0]
        A, w, _ = self.design_matrix(X)
        y_hat = A @ self.theta
        resid = y_hat - y                                   # [N]

        total = w.sum(axis=1)                               # [N]
        safe_total = np.where(total > 0.0, total, 1.0)

        # f_r per sample: the rule's own linear consequent.
        Xa = np.hstack([X, np.ones((n, 1))])                # [N, I+1]
        P = self.theta.reshape(self.n_rules, self.n_inputs + 1)
        f = Xa @ P.T                                        # [N, R]

        # dE/dw_r
        dE_dy = (2.0 / n) * resid                           # [N]
        dE_dw = dE_dy[:, None] * (f - y_hat[:, None]) / safe_total[:, None]

        mu = self.memberships(X)                            # [N, I, M]
        d_mu = self.mf.gradients(self.params[None, :, :, :],
                                 X[:, :, None])             # [N, I, M, P]

        grad = np.zeros_like(self.params)                   # [I, M, P]

        for i in range(self.n_inputs):
            idx = self.rules[:, i]
            for j in range(self.n_mfs):
                sel = idx == j
                if not np.any(sel):
                    continue
                grade = mu[:, i, j]                         # [N]
                # dw_r/dmu = w_r / mu, guarded against an underflowed grade.
                ratio = np.divide(w[:, sel], grade[:, None],
                                  out=np.zeros_like(w[:, sel]),
                                  where=grade[:, None] > 1e-12)
                dE_dmu = np.sum(dE_dw[:, sel] * ratio, axis=1)   # [N]
                grad[i, j, :] = dE_dmu @ d_mu[:, i, j, :]

        return grad

    def fit(self, X, y, epochs=200, step=0.005, min_width=1e-3, ridge=0.0,
            X_val=None, y_val=None, verbose=False):
        """Hybrid training: exact consequents, descended premises.

        The premise step is *normalized* -- the gradient is scaled to unit
        L2 norm before stepping, so `step` is the distance the premise
        parameters travel per epoch in their own units rather than a raw
        learning rate. This matters because the least-squares step drives
        the residual down first, which leaves the premise gradient several
        orders of magnitude smaller than the parameters it acts on; a plain
        learning rate large enough to move anything on one problem diverges
        on the next. Normalizing makes `step` mean the same thing regardless
        of how well the consequents already fit.

        Returns a history list of (epoch, train_rmse, val_rmse) with
        val_rmse == None when no validation set is supplied.
        """
        history = []

        def record(epoch):
            # Measured only ever right after a solve, so the reported error
            # belongs to a consistent (premise, consequent) pair. Measuring
            # after the premise step but before the re-solve would report a
            # model that never exists -- stepped premises still carrying the
            # previous premises' consequents.
            tr = rmse(self.predict(X), y)
            va = rmse(self.predict(X_val), y_val) if X_val is not None else None
            history.append((epoch, tr, va))
            if verbose and (epoch % 20 == 0 or epoch == epochs):
                if va is None:
                    print("  epoch %4d  train RMSE %.6f" % (epoch, tr))
                else:
                    print("  epoch %4d  train RMSE %.6f  val RMSE %.6f"
                          % (epoch, tr, va))

        for epoch in range(epochs):
            self.solve_consequents(X, y, ridge=ridge)
            record(epoch)

            grad = self.premise_gradients(X, y)
            norm = np.sqrt(np.sum(grad * grad))
            if norm > 0.0:
                self.params -= step * grad / norm
            # Repair whatever constraint the step may have broken (a
            # non-positive bell width, out-of-order triangle knots).
            self.mf.project(self.params, min_width)

        # The last descent step moved the premises, so re-solve to leave the
        # consequents exact for the premises actually being shipped, and
        # record that final pair.
        self.solve_consequents(X, y, ridge=ridge)
        record(epochs)
        return history

    # -- rule pruning -------------------------------------------------------

    def rule_importance(self, X):
        """Mean normalized firing strength per rule -- the pruning score."""
        w = self.firing_strengths(X)
        total = w.sum(axis=1, keepdims=True)
        wbar = np.divide(w, total, out=np.zeros_like(w), where=total > 0.0)
        return wbar.mean(axis=0)

    def prune(self, X, y, threshold):
        """Drop rules whose mean normalized firing strength is below
        `threshold`, then re-solve the consequents over the survivors.

        Returns the indices kept. Never drops every rule.
        """
        score = self.rule_importance(X)
        keep = np.where(score >= threshold)[0]
        if keep.size == 0:
            keep = np.array([int(np.argmax(score))])
        if keep.size == self.n_rules:
            return keep

        self.rules = self.rules[keep]
        self.theta = self.theta.reshape(-1, self.n_inputs + 1)[keep].reshape(-1)
        self.solve_consequents(X, y)
        return keep

    # -- flat parameter views for the header emitter ------------------------

    def premise_flat(self):
        """[n_inputs][n_mfs][parameters] flattened, matching cpp/anfis.hpp."""
        return self.params.reshape(-1)

    def rule_table_flat(self):
        """[n_rules][n_inputs] flattened, as uint8 antecedent indices."""
        return np.clip(self.rules, 0, 255).astype(np.uint8).reshape(-1)

    def consequent_flat(self):
        """[n_rules][n_outputs=1][p_0..p_{n-1}, q] flattened."""
        return self.theta.reshape(-1)


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------

def full_grid_rules(n_inputs, n_mfs):
    """Every combination of one membership function per input.

    Mirrors AnfisFullGridRuleTable in cpp/anfis.hpp, including the odometer
    order (the last input varies fastest), so a table generated here and one
    generated on the device are identical.
    """
    n_rules = n_mfs ** n_inputs
    rules = np.zeros((n_rules, n_inputs), dtype=np.int32)
    for r in range(n_rules):
        rem = r
        for i in range(n_inputs - 1, -1, -1):
            rules[r, i] = rem % n_mfs
            rem //= n_mfs
    return rules


def build_grid_anfis(X, n_mfs, mf=BellMembership):
    """Grid-partitioned ANFIS over the full rule grid."""
    return Anfis(X.shape[1], n_mfs, full_grid_rules(X.shape[1], n_mfs),
                 mf.initialize(X, n_mfs), mf=mf)


def subtractive_clustering(X, radius=0.5, squash=1.25, accept_ratio=0.5,
                           reject_ratio=0.15, max_clusters=64):
    """Chiu's subtractive clustering (1994). Returns (centers, widths).

    Every data point is a candidate cluster center. A point's potential is
    the sum of its neighbours' proximity, so points in dense regions score
    highest:

        P_i = sum_j exp(-alpha * ||z_i - z_j||^2),   alpha = 4 / radius^2

    The highest-potential point becomes a center, then that center's
    influence is *subtracted* from every remaining point, which stops the
    next pick from landing on its immediate neighbour:

        P_i <- P_i - P_k * exp(-beta * ||z_i - z_k||^2),
                                     beta = 4 / (squash * radius)^2

    Accept / reject thresholds are relative to the first center's potential.
    A candidate between the two is taken only if it is far enough from every
    existing center to be worth a rule of its own; otherwise it is discarded
    and the search continues.

    Distances are computed on the unit hypercube (each input min-max scaled),
    so `radius` means the same thing regardless of the physical units of the
    inputs. It is the one knob that matters: smaller radius, more clusters.

    Returns centers in the ORIGINAL input units, and the per-input width that
    `radius` corresponds to, ready for a membership function policy's
    `initialize_at`.
    """
    lo = X.min(axis=0)
    hi = X.max(axis=0)
    span = np.where(hi > lo, hi - lo, 1.0)
    Z = (X - lo) / span

    alpha = 4.0 / (radius * radius)
    beta = 4.0 / ((squash * radius) ** 2)

    squared = ((Z[:, None, :] - Z[None, :, :]) ** 2).sum(-1)
    potential = np.exp(-alpha * squared).sum(axis=1)

    centers = []
    first = float(potential.max())

    while len(centers) < max_clusters:
        k = int(np.argmax(potential))
        peak = float(potential[k])

        if peak <= 0.0:
            break

        if peak <= reject_ratio * first:
            break

        if peak < accept_ratio * first:
            # Gray zone: keep it only if it buys coverage somewhere new.
            nearest = min(float(np.sqrt(((Z[k] - c) ** 2).sum()))
                          for c in centers) if centers else float("inf")
            if (nearest / radius) + (peak / first) < 1.0:
                potential[k] = 0.0
                continue

        centers.append(Z[k].copy())
        potential = potential - peak * np.exp(
            -beta * ((Z - Z[k]) ** 2).sum(-1))
        np.maximum(potential, 0.0, out=potential)

    if not centers:
        centers = [Z[int(np.argmax(potential))].copy()]

    return np.array(centers) * span + lo, radius * span


def build_scatter_anfis(X, radius=0.5, mf=BellMembership, **kwargs):
    """Scatter-partitioned ANFIS: one rule per subtractive-clustering center.

    This is the structural alternative to `build_grid_anfis`, and the reason
    to reach for it is rule count. A grid takes the Cartesian product of every
    input's membership functions, so rules grow as M^N -- 3 membership
    functions over 8 inputs is 6561 rules, which is unusable. A scatter
    partition puts one membership function per input *per cluster* and pairs
    them diagonally, so the rule count is the number of clusters the data
    actually has, independent of the input count.

    On the bundled Mackey-Glass benchmark (4 inputs), a grid at 3 membership
    functions is 81 rules while subtractive clustering finds 8; at 8 inputs it
    is 6561 against 14.

    The rule table is the diagonal -- rule r reads membership function r of
    every input -- which `cpp/anfis.hpp` consumes unchanged, since it takes an
    explicit table rather than assuming a grid.
    """
    centers, widths = subtractive_clustering(X, radius=radius, **kwargs)
    n_clusters, n_inputs = centers.shape
    rules = np.tile(np.arange(n_clusters, dtype=np.int32)[:, None],
                    (1, n_inputs))
    return Anfis(n_inputs, n_clusters, rules,
                 mf.initialize_at(centers, widths), mf=mf)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def rmse(pred, target):
    return float(np.sqrt(np.mean((np.asarray(pred) - np.asarray(target)) ** 2)))


# ---------------------------------------------------------------------------
# Header emission
# ---------------------------------------------------------------------------

def _format_array(cpp_type, name, values, per_line, fmt):
    body = []
    vals = list(values)
    for start in range(0, len(vals), per_line):
        chunk = vals[start:start + per_line]
        body.append("        " + ", ".join(fmt % v for v in chunk))
    joined = ",\n".join(body)
    return ("    constexpr %s %s[%d] = {\n%s\n    };\n"
            % (cpp_type, name, len(vals), joined))


def emit_model_header(path, model, namespace, value_type="double",
                      meta=None, comment=None):
    """Write a header holding the three frozen arrays cpp/anfis.hpp needs.

    The arrays are emitted at full double precision. The C++ side chooses
    its own ValueType; for a Q-format build the values are converted at the
    call site, matching how the rest of TinyMind treats host-produced
    constants.
    """
    n_inputs = model.n_inputs
    n_mfs = model.n_mfs
    n_rules = model.n_rules
    mf = model.mf

    parts = []
    parts.append("// Generated by apps/anfis_train/anfis_train.py -- do not edit by hand.\n")
    if comment:
        for line in comment.strip().splitlines():
            parts.append("// %s\n" % line.rstrip())
    parts.append("\n#pragma once\n\n#include <cstddef>\n#include <cstdint>\n\n")
    parts.append("namespace %s {\n" % namespace)

    parts.append("    constexpr std::size_t NumberOfInputs = %d;\n" % n_inputs)
    parts.append("    constexpr std::size_t NumberOfMembershipFunctionsPerInput = %d;\n" % n_mfs)
    parts.append("    constexpr std::size_t NumberOfRules = %d;\n" % n_rules)
    parts.append("    constexpr std::size_t NumberOfOutputs = 1;\n")
    parts.append("    constexpr std::size_t NumberOfParametersPerMembershipFunction = %d;\n"
                 % mf.NumberOfParameters)
    for key, value in sorted((meta or {}).items()):
        parts.append("    constexpr std::size_t %s = %d;\n" % (key, int(value)))
    parts.append("\n")

    parts.append("    // Membership function: tinymind::%s\n" % mf.cpp_type)
    parts.append("    // [input][membershipFunction]{%s}\n"
                 % ", ".join(mf.parameter_names))
    parts.append(_format_array(value_type, "Premise",
                               model.premise_flat(), 6, "%.9g"))
    parts.append("\n    // [rule][input] -- antecedent membership function index.\n")
    parts.append(_format_array("uint8_t", "RuleTable",
                               model.rule_table_flat(), 16, "%d"))
    parts.append("\n    // [rule][output]{p_0 .. p_{n-1}, q} -- first-order consequent.\n")
    parts.append(_format_array(value_type, "Consequent",
                               model.consequent_flat(), 5, "%.9g"))

    parts.append("}\n")

    with open(path, "w") as handle:
        handle.write("".join(parts))
    return path
