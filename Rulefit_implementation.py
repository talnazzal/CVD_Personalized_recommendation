"""
Rulefit_implementation.py
=================
Fully reproducible RuleFit Classifier using only scikit-learn.
No dependency on other public library.
The implementation is not new, it follows RuleFit developed by Jerome H. Friedman and Bogdan E. Popescu.
We just added some enhancement and options like warm-starting, improving the rule extraction by considering removing the canonical rules, and option to use Elastic-net to control the rules selection and reproducibility.
The implementation focused on classification implementation only.

Algorithm
---------
Phase 1 : GBM warm-start tree generation  (GradientBoostingRegressor)
Phase 2 : Rule extraction at ALL nodes     + canonical deduplication
Phase 3 : Binary rule matrix               (pandas.query vectorized evaluation)
Phase 4 : L1 or elastic-net LogisticRegression (saga / liblinear) with CV alpha search

Blueprint bugs fixed
--------------------
1. Rule extraction: original code only appended rules at LEAF nodes (else branch).
   Fixed → emit a rule for every non-root node (internal + leaf) → ~2x candidates.
2. predict_proba: blueprint used softmax(vstack((1-score, score))) which is
   mathematically incorrect. Fixed → delegate to LogisticRegression.predict_proba.
3. Alpha search: used `alphas[i-1]` which raises IndexError when loop exhausts
   all alphas without exceeding budget. Fixed → explicit valid_alphas accumulator.
4. coef_zero_threshold = 1e-6 / mean(|y|) is unstable for binary labels.
   Fixed → constant threshold 1e-6.
5. Feature names with spaces/special chars break split(" ") in factorize_rule.
   Fixed → sanitize feature names (non-alphanumeric → underscores) in fit().
6. Tree sizes not clamped. Fixed → max(2, 2 + floor(s)) guarantees min 2 leaves.
7. Global RNG pollution: blueprint used np.random.seed(). Fixed → private
   np.random.RandomState(random_state) — zero impact on global state.

Usage
-----
    from Rulefit_implementation import RuleFitClassifierModel

    clf = RuleFitClassifierModel(random_state=0)
    clf.fit(X_train, y_train, feature_names=feature_names)

    print(clf.get_rules(top_n=20))
    clf.summary()

    y_pred  = clf.predict(X_test)
    proba   = clf.predict_proba(X_test)
    acc     = clf.score(X_test, y_test)

Dependencies
------------
    pip install numpy pandas scikit-learn
"""

from __future__ import annotations

import re
import random
import multiprocessing
import warnings
import contextlib
import numpy as np
import pandas as pd

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.tree import _tree
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels

try:
    from joblib import Parallel, delayed
    _HAS_JOBLIB = True
except ImportError:
    _HAS_JOBLIB = False

try:
    from threadpoolctl import threadpool_limits as _threadpool_limits
except ImportError:
    _threadpool_limits = None

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


# =============================================================================
# Module-level worker functions (must be picklable for joblib loky backend)
# =============================================================================

def _extract_rules_from_tree(tree, feature_names: list[str]) -> list[str]:
    """Worker: extract rules from a single sklearn DecisionTree."""
    return RuleFitClassifierModel._tree_to_rules(tree, feature_names)


def _eval_rule_chunk(
    X_values: np.ndarray,
    columns: list[str],
    rules: list[str],
    start_idx: int,
    n_samples: int,
) -> list[tuple[int, np.ndarray]]:
    """
    Worker: evaluate a chunk of rules against the data.

    Returns a list of (column_index, matching_row_indices) tuples.
    """
    df = pd.DataFrame(X_values, columns=columns)
    results: list[tuple[int, np.ndarray]] = []
    for j, rule in enumerate(rules):
        try:
            idx = df.query(rule).index.values
            results.append((start_idx + j, idx))
        except Exception:
            pass  # malformed rule — column stays 0
    return results


def _fit_one_cv_fold(
    estimator: LogisticRegression,
    X_rules: np.ndarray,
    y: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    blas_threads: int | None,
) -> float:
    """
    Worker: fit one CV fold with BLAS threads pinned.

    Process-isolated (loky backend) so BLAS state cannot leak between folds.
    """
    ctx = (
        _threadpool_limits(limits=blas_threads, user_api="blas")
        if (blas_threads is not None and _threadpool_limits is not None)
        else contextlib.nullcontext()
    )
    with ctx:
        est = clone(estimator)
        est.fit(X_rules[train_idx], y[train_idx])
        pred = est.predict(X_rules[test_idx])
        return float(np.mean(pred == y[test_idx]))


def _run_stability_iteration(
    iteration_idx: int,
    X_all: np.ndarray,
    y_all: np.ndarray,
    n_sub: int,
    base_random_state: int,
    estimator_builder_params: dict,
    best_alpha: float,
    coef_zero_threshold: float,
    offset: int,
) -> list[float]:
    """
    Worker function for a single stability selection iteration.
    Returns the coefficients for the rules in this iteration.
    """
    rng = np.random.RandomState(base_random_state + iteration_idx)
    n_samples = X_all.shape[0]
    indices = rng.choice(n_samples, size=n_sub, replace=False)
    X_sub = X_all[indices]
    y_sub = y_all[indices]
    
    # Reconstruct simple LogisticRegression for this iteration
    # We use a simplified version because we can't easily pass the full class method
    C = 1.0 / float(best_alpha)
    pen = estimator_builder_params['penalty']
    if pen == "elasticnet":
        model = LogisticRegression(
            penalty="elasticnet",
            l1_ratio=estimator_builder_params['l1_ratio'],
            C=C,
            solver="saga",
            random_state=base_random_state + iteration_idx,
            max_iter=estimator_builder_params['max_iter'],
            tol=estimator_builder_params['tol'],
            n_jobs=1,
        )
    else:
        model = LogisticRegression(
            penalty="l1",
            C=C,
            solver=estimator_builder_params['solver'],
            random_state=base_random_state + iteration_idx,
            max_iter=estimator_builder_params['max_iter'],
            tol=estimator_builder_params['tol'],
        )
    
    model.fit(X_sub, y_sub)
    coefs = model.coef_.flatten()
    rule_coefs = coefs[offset:]
    intercept = float(model.intercept_[0])
    return rule_coefs.tolist(), intercept


# =============================================================================
# RuleFit Classifier
# =============================================================================

class RuleFitClassifierModel(BaseEstimator, ClassifierMixin):
    """
    RuleFit Classifier — fully reproducible, sklearn-compatible.

    Parameters
    ----------
    n_estimators : int, default=300
        Number of GBM boosting iterations (one tree per iteration).
        Increase (e.g. 1500) for stronger tree ensembles at higher runtime cost.

    tree_size : int, default=4
        Mean leaf-node count per tree.  When exp_rand_tree_size=True (default),
        this is the scale parameter of the Exponential distribution used to draw
        per-tree leaf counts: actual_leaves ~ 2 + Exp(tree_size - 2).

    max_depth : int or None, default=None
        Maximum depth of each tree in the GBM ensemble.  When None, depth is
        effectively unconstrained (internal ceiling of 100).  When set, each
        tree is simultaneously constrained by **both** max_leaf_nodes (from
        tree_size) and max_depth — growth stops when either limit is reached.
        This lets you control rule complexity (shorter rules) independently
        of tree leaf diversity.

    sample_fract : float, default=0.5
        Row-subsampling fraction for each GBM tree (maps to `subsample` in
        GradientBoostingRegressor).  Note: imodels silently ignores this
        parameter and uses its own formula; this implementation honours it.

    max_rules : int, default=115
        Maximum number of non-zero rule coefficients allowed in the final model.
        The alpha search stops as soon as any alpha would exceed this budget.

    memory_par : float, default=0.1
        GBM learning rate (shrinkage per tree).  Maps to `learning_rate` in
        GradientBoostingRegressor.  Smaller values give slower learning per tree;
        default is tuned for quicker iteration with elastic-net Phase 4.

    exp_rand_tree_size : bool, default=True
        If True, draw each tree\'s leaf count from Exp(tree_size-2) for diversity.
        If False, every tree has exactly tree_size leaves.

    cv : bool, default=True
        Use stratified cross-validation (see ``cv_n_splits``) to select alpha.
        If False, selects the last valid alpha (weakest regularisation within budget).

    logistic_penalty : str, default='elasticnet'
        ``'l1'`` or ``'elasticnet'``. Elastic net uses L1+L2 and requires ``solver='saga'``.

    elasticnet_l1_ratio : float, default=0.5
        L1 share of the elastic-net penalty (only when ``logistic_penalty='elasticnet'``).

    refit_on_selected_rules : bool, default=False
        If True, after Phase 4 selection, fit a *second* logistic model (see
        ``refit_penalty``, ``refit_solver``) using **only** the selected rule
        columns. ``predict``/``predict_proba`` then use this refit model. The
        **rule list** is still entirely determined by the first stage; the refit
        only re-estimates weights (e.g. L2) on that submatrix, often smoothing
        probabilities but adding one extra fit cost.

    random_state : int, default=0
        Master random seed.  Controls:
          - Private RandomState for tree-size sampling (no global pollution)
          - GBM warm-start seed, advanced by i per iteration
          - LogisticRegression solver seed

    max_iter : int, default=8000
        Maximum iterations for the logistic solver in Phase 4 (mainly ``saga``).

    include_linear : bool, default=False
        If True, append the raw feature columns to the binary rule matrix
        before fitting the logistic regression (equivalent to RuleFit\'s
        include_linear=True).  The linear feature coefficients are NOT included
        in get_rules() output.

    Attributes (set after fit)
    --------------------------
    classes_ : ndarray of shape (n_classes,)
    feature_names_ : list[str]   — sanitized feature names used internally
    gbm_ : GradientBoostingRegressor — fitted GBM (phases 1-2)
    rules_ : list[str]           — all unique candidate rule strings
    lasso_model_ : LogisticRegression — fitted Phase-4 model (or refit model if enabled)
    best_alpha_ : float          — selected L1 penalty
    selected_rules_ : list[str]  — non-zero rules after L1 selection
    selected_coefs_ : ndarray    — corresponding coefficients
    n_rules_total_ : int         — candidate rules after deduplication
    n_rules_selected_ : int      — rules with non-zero coefficient
    """

    def __init__(
        self,
        n_estimators: int   = 300,
        tree_size: int      = 4,
        max_depth: int | None = None,
        sample_fract: float = 0.5,
        max_rules: int      = 115,
        memory_par: float   = 0.1,
        exp_rand_tree_size: bool = True,
        cv: bool            = True,
        random_state: int   = 0,
        max_iter: int       = 8000,
        include_linear: bool = False,
        coef_zero_threshold: float = 5e-4,
        solver_tol: float = 5e-5,
        saga_tol_floor: float = 5e-5,
        n_alphas: int       = 40,
        cv_n_splits: int    = 3,
        logistic_penalty: str = "elasticnet",
        elasticnet_l1_ratio: float = 0.5,
        logistic_solver: str = "saga",
        blas_threads: int | None = 1,
        n_jobs: int             = 1,
        refit_on_selected_rules: bool = False,
        refit_penalty: str = "l2",
        refit_C: float = 1.0,
        refit_solver: str = "lbfgs",
        refit_max_iter: int | None = None,
        l1_mask_round_decimals: int | None = 4,
        l1_rank_round_decimals: int | None = None,
        l1_selection_policy: str = "top_k_lexsort",
        winsor_fract: float = 0.025,
    ) -> None:
        self.n_estimators       = n_estimators
        self.tree_size          = tree_size
        self.max_depth          = max_depth
        self.sample_fract       = sample_fract
        self.max_rules          = max_rules
        self.memory_par         = memory_par
        self.exp_rand_tree_size = exp_rand_tree_size
        self.cv                 = cv
        self.random_state       = random_state
        self.max_iter           = max_iter
        self.include_linear     = include_linear
        self.coef_zero_threshold = coef_zero_threshold
        self.solver_tol = solver_tol
        self.saga_tol_floor = saga_tol_floor
        self.n_alphas = int(n_alphas)
        self.cv_n_splits = int(cv_n_splits)
        self.logistic_penalty = logistic_penalty
        self.elasticnet_l1_ratio = float(elasticnet_l1_ratio)
        self.logistic_solver = logistic_solver
        self.blas_threads = blas_threads
        self.n_jobs = n_jobs
        self.refit_on_selected_rules = refit_on_selected_rules
        self.refit_penalty = refit_penalty
        self.refit_C = refit_C
        self.refit_solver = refit_solver
        self.refit_max_iter = refit_max_iter
        self.l1_mask_round_decimals = l1_mask_round_decimals
        self.l1_rank_round_decimals = l1_rank_round_decimals
        self.l1_selection_policy = l1_selection_policy
        self.winsor_fract = winsor_fract

    def _effective_logistic_solver(self) -> str:
        """``elasticnet`` is only implemented with ``solver='saga'`` in scikit-learn."""
        if (self.logistic_penalty or "l1").lower() == "elasticnet":
            return "saga"
        return self.logistic_solver

    def _resolve_n_jobs(self) -> int:
        """
        Return effective n_jobs, automatically falling back to 1 when nested.

        Nested loky process pools (e.g. model.fit() called inside an outer
        joblib.Parallel CV loop) crash on Windows with
        ``TerminatedWorkerError``.  This method detects that situation via
        ``multiprocessing.parent_process()`` (non-None ⇒ we are in a child)
        and silently returns 1 so the model runs sequentially inside the
        worker, while the outer parallelism still provides speedup.
        """
        if self.n_jobs == 1 or not _HAS_JOBLIB:
            return 1
        if multiprocessing.parent_process() is not None:
            return 1  # already inside a worker — do not nest
        return self.n_jobs

    def _make_logistic_regression(self, alpha: float) -> LogisticRegression:
        """
        Build regularized logistic regression for a given alpha (``C = 1/alpha``).

        - ``logistic_penalty='l1'``: use ``logistic_solver`` (``liblinear`` or ``saga``).
        - ``logistic_penalty='elasticnet'``: mix of L1 + L2; sklearn **requires**
          ``solver='saga'`` (``liblinear`` cannot fit this objective). The L2 term
          often smooths coefficient paths vs pure L1, which can help stability,
          at the cost of typically **longer** runs than ``liblinear``.

        ``liblinear`` is fast but has shown run-to-run drift in this pipeline;
        ``saga`` is preferred when reproducibility of the selected rule set matters.
        """
        pen = (self.logistic_penalty or "l1").lower()
        C = 1.0 / float(alpha)
        if pen == "elasticnet":
            lratio = float(self.elasticnet_l1_ratio)
            if not (0.0 <= lratio <= 1.0):
                raise ValueError(
                    f"elasticnet_l1_ratio must be in [0, 1], got {lratio}"
                )
            tol = max(float(self.solver_tol), float(self.saga_tol_floor))
            return LogisticRegression(
                penalty="elasticnet",
                l1_ratio=lratio,
                C=C,
                solver="saga",
                random_state=self.random_state,
                max_iter=self.max_iter,
                tol=tol,
                n_jobs=1,
            )
        if pen != "l1":
            raise ValueError(
                "logistic_penalty must be 'l1' or 'elasticnet', "
                f"got {self.logistic_penalty!r}"
            )
        solver = self.logistic_solver
        tol = float(self.solver_tol)
        if solver == "saga":
            tol = max(tol, float(self.saga_tol_floor))
        kwargs: dict = dict(
            penalty="l1",
            C=C,
            solver=solver,
            random_state=self.random_state,
            max_iter=self.max_iter,
            tol=tol,
        )
        if solver == "saga":
            kwargs["n_jobs"] = 1
        return LogisticRegression(**kwargs)

    @staticmethod
    def _winsorize(X: np.ndarray, q_low: np.ndarray, q_high: np.ndarray) -> np.ndarray:
        """Apply winsorization using pre-computed quantiles per column."""
        return np.clip(X, q_low, q_high)

    # =========================================================================
    # Utility
    # =========================================================================

    @staticmethod
    def _sanitize_names(names: list[str]) -> list[str]:
        """
        Replace characters that are illegal in pandas.query column names
        (spaces, dots, parentheses, hyphens …) with underscores.
        This is required before building rule strings so that
        _factorize_rule can safely call term.split(" ") expecting exactly
        3 tokens: <feature> <operator> <value>.
        """
        return [re.sub(r"[^a-zA-Z0-9_]", "_", n) for n in names]

    # =========================================================================
    # Phase 1 — GBM Warm-Start Tree Generation
    # =========================================================================

    def _compute_tree_sizes(self) -> np.ndarray:
        """
        Sample per-tree leaf counts.

        Uses a *private* np.random.RandomState so that this method never
        modifies the global numpy random state — unlike the imodels reference
        which calls np.random.seed() and corrupts any concurrent code.

        Returns
        -------
        tree_sizes : ndarray[int], shape (n_estimators,)
            Each value is the max_leaf_nodes for that GBM iteration.
            Minimum is 2 (single split).
        """
        rng = np.random.RandomState(self.random_state)
        if self.exp_rand_tree_size:
            scale = max(self.tree_size - 2, 1e-9)     # guard against tree_size <= 2
            raw   = rng.exponential(scale=scale, size=self.n_estimators)
            return np.asarray(
                [max(2, 2 + int(np.floor(s))) for s in raw], dtype=int
            )
        return np.full(self.n_estimators, max(2, self.tree_size), dtype=int)

    def _generate_trees(
        self, X: np.ndarray, y: np.ndarray
    ) -> GradientBoostingRegressor:
        """
        Build the GBM via a warm-start loop — one tree added per iteration.

        Why warm-start?
        ---------------
        We need each tree to have a *different* max_leaf_nodes, which sklearn's
        GradientBoostingRegressor does not support natively.  By setting
        warm_start=True and calling fit() n times (each time with
        n_estimators = i+1), only the (i+1)-th tree is grown at each step.

        Why advance random_state each step?
        ------------------------------------
        With warm_start=True, sklearn resets its internal RNG on every fit()
        call using the current random_state.  If random_state is fixed, every
        new tree starts from an identical RNG state, making successive trees
        highly correlated.  Advancing by 1 per step (i + base_seed) breaks this
        correlation without sacrificing reproducibility.

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features) — training features
        y : ndarray, shape (n_samples,)             — float targets (0.0 / 1.0)

        Returns
        -------
        GradientBoostingRegressor with n_estimators fitted trees.
        """
        tree_sizes = self._compute_tree_sizes()

        effective_depth = self.max_depth if self.max_depth is not None else 100

        gbm = GradientBoostingRegressor(
            n_estimators  = 1,
            max_leaf_nodes= int(tree_sizes[0]),
            learning_rate = self.memory_par,
            subsample     = self.sample_fract,
            random_state  = self.random_state,
            max_depth     = effective_depth,
            warm_start    = True,
        )

        for i in range(self.n_estimators):
            gbm.set_params(
                n_estimators  = i + 1,
                max_leaf_nodes= int(tree_sizes[i]),
                random_state  = i + self.random_state,   # advances for diversity
                max_depth     = effective_depth,          # explicit per-iteration
            )
            gbm.fit(X.copy(), y.copy())    # .copy() avoids any accidental mutation

        gbm.set_params(warm_start=False)
        return gbm

    # =========================================================================
    # Phase 2 — Rule Extraction + Deduplication
    # =========================================================================

    @staticmethod
    def _tree_to_rules(tree, feature_names: list[str]) -> list[str]:
        """
        Extract rule strings for EVERY non-root node (root-to-node paths).

        Blueprint bug
        -------------
        The original recurse() only appended rule strings inside the `else`
        branch (leaf nodes).  This means only N rules are generated for a tree
        with N leaves, missing all 2N-2 internal-node paths.  The RuleFit paper
        explicitly requires rules from every node.

        Fix
        ---
        Emit `" and ".join(conditions)` at the *start* of every recursive call
        (before branching), skipping only the root (when conditions == []).
        A tree with N leaves has 2N-1 total nodes → yields 2N-2 candidate rules.

        Parameters
        ----------
        tree         : sklearn DecisionTree estimator (gbm.estimators_[i][0])
        feature_names: list[str] — sanitized names matching X columns

        Returns
        -------
        rules : list[str] — one rule string per non-root node
        """
        tree_   = tree.tree_
        feat    = [
            feature_names[i] if i != _tree.TREE_UNDEFINED else "__undef__"
            for i in tree_.feature
        ]
        rules: list[str] = []

        def recurse(node: int, conditions: list[str]) -> None:
            # Emit rule for every non-root node
            if conditions:
                rules.append(" and ".join(conditions))
            # Recurse into children (internal nodes only)
            if tree_.feature[node] != _tree.TREE_UNDEFINED:
                name   = feat[node]
                thresh = tree_.threshold[node]
                recurse(tree_.children_left[node],
                        conditions + [f"{name} <= {thresh}"])
                recurse(tree_.children_right[node],
                        conditions + [f"{name} > {thresh}"])

        recurse(0, [])
        return rules

    @staticmethod
    def _factorize_rule(rule_str: str) -> tuple:
        """
        Convert a rule string into a canonical hashable form for deduplication.

        Two rules are duplicates if, after resolving redundant conditions on the
        same feature, they constrain each variable to identical intervals.

        Redundancy resolution
        ---------------------
        A tree can split on the same feature twice (e.g., `age > 30` at depth 1
        and `age > 40` at depth 2).  We keep only the tighter bound:
          - x <= A and x <= B  →  x <= min(A, B)   (tighter upper bound)
          - x >  A and x >  B  →  x >  max(A, B)   (tighter lower bound)

        Requires
        --------
        Feature names must be pre-sanitized (no spaces) so that each term
        can be split by " " into exactly 3 tokens: <feature> <op> <value>.

        Returns
        -------
        canonical : tuple — sorted tuple of ((feature, op), value) items
        """
        agg: dict = {}
        for term in rule_str.split(" and "):
            feat, op, val = term.strip().split(" ")
            val            = float(val)
            key            = (feat, op)
            if key not in agg:
                agg[key] = val
            elif op.startswith("<"):
                agg[key] = min(agg[key], val)   # keep tighter upper bound
            elif op.startswith(">"):
                agg[key] = max(agg[key], val)   # keep tighter lower bound
        return tuple(sorted(agg.items()))

    def _extract_unique_rules(
        self,
        gbm: GradientBoostingRegressor,
        feature_names: list[str],
    ) -> list[str]:
        """
        Traverse all trees, extract rules at every node, deduplicate.

        When ``n_jobs != 1``, tree traversals run in parallel (one worker per
        tree).  Deduplication is always sequential in tree order to guarantee
        deterministic rule lists regardless of worker completion order.

        Returns
        -------
        unique_rules : list[str]
            First-seen rule string for each unique canonical form.
            Order is deterministic (traversal order of gbm.estimators_).
        """
        trees = [stage[0] for stage in gbm.estimators_]

        # ── Extract rules (parallel-safe: pure traversal, no shared state) ──
        eff_jobs = self._resolve_n_jobs()
        if eff_jobs != 1 and _HAS_JOBLIB and len(trees) > 1:
            all_rule_lists: list[list[str]] = Parallel(
                n_jobs=eff_jobs, backend="loky"
            )(
                delayed(_extract_rules_from_tree)(t, feature_names)
                for t in trees
            )
        else:
            all_rule_lists = [
                self._tree_to_rules(t, feature_names) for t in trees
            ]

        # ── Sequential dedup in deterministic tree order ─────────────────
        seen:         set       = set()
        unique_rules: list[str] = []
        for rule_list in all_rule_lists:
            for rule_str in rule_list:
                canonical = self._factorize_rule(rule_str)
                if canonical not in seen:
                    unique_rules.append(rule_str)
                    seen.add(canonical)

        return unique_rules

    # =========================================================================
    # Phase 3 — Binary Rule Matrix
    # =========================================================================

    @staticmethod
    def _rules_to_binary_matrix(
        X: np.ndarray,
        rules: list[str],
        feature_names: list[str],
        n_jobs: int = 1,
    ) -> np.ndarray:
        """
        Convert rule strings to a float32 binary indicator matrix.

        For each rule r_j and sample x_i:
            X_rules[i, j] = 1.0  if x_i satisfies all conditions in r_j
                          = 0.0  otherwise

        When ``n_jobs != 1``, rules are split into balanced chunks and each
        chunk is evaluated in a separate worker process.  Results are written
        into the pre-allocated matrix by column index, so the output is
        identical regardless of worker count or completion order.

        Parameters
        ----------
        X            : ndarray, shape (n_samples, n_features)
        rules        : list[str], length n_rules
        feature_names: list[str], length n_features (sanitized)
        n_jobs       : int, default=1 — number of parallel workers

        Returns
        -------
        X_rules : ndarray[float32], shape (n_samples, n_rules)
        """
        n_samples = X.shape[0]
        n_rules   = len(rules)
        X_rules   = np.zeros((n_samples, n_rules), dtype=np.float32)

        if n_jobs != 1 and _HAS_JOBLIB and n_rules > 1:
            # Resolve effective worker count for chunking
            import joblib
            eff_jobs = joblib.effective_n_jobs(n_jobs)
            chunk_size = max(1, (n_rules + eff_jobs - 1) // eff_jobs)
            chunks = [
                (rules[i : i + chunk_size], i)
                for i in range(0, n_rules, chunk_size)
            ]
            chunk_results = Parallel(n_jobs=n_jobs, backend="loky")(
                delayed(_eval_rule_chunk)(
                    X, feature_names, chunk_rules, start_idx, n_samples
                )
                for chunk_rules, start_idx in chunks
            )
            # Assemble results by column index (deterministic)
            for result_list in chunk_results:
                for col_idx, row_indices in result_list:
                    X_rules[row_indices, col_idx] = 1.0
        else:
            df = pd.DataFrame(X, columns=feature_names)
            for i, rule in enumerate(rules):
                try:
                    idx = df.query(rule).index.values
                    X_rules[idx, i] = 1.0
                except Exception:
                    pass  # malformed rule — column stays 0

        return X_rules

    # =========================================================================
    # Phase 4 — L1 Alpha Search + Final Logistic Regression
    # =========================================================================

    def _mean_stratified_cv_accuracy(
        self,
        base_estimator: LogisticRegression,
        X_rules: np.ndarray,
        y: np.ndarray,
        cv_splitter: StratifiedKFold,
    ) -> float:
        """
        Mean accuracy over StratifiedKFold splits.

        When ``n_jobs != 1``, folds are fitted in parallel with process
        isolation (loky backend) and per-worker BLAS thread pinning to
        prevent floating-point accumulation-order drift.  Fold scores are
        collected in fold order (deterministic), not completion order.
        """
        y_arr  = np.asarray(y)
        splits = list(cv_splitter.split(X_rules, y_arr))  # materialize for determinism

        eff_jobs = self._resolve_n_jobs()
        if eff_jobs != 1 and _HAS_JOBLIB and len(splits) > 1:
            fold_scores = Parallel(n_jobs=eff_jobs, backend="loky")(
                delayed(_fit_one_cv_fold)(
                    base_estimator, X_rules, y_arr,
                    train_idx, test_idx, self.blas_threads,
                )
                for train_idx, test_idx in splits
            )
        else:
            fold_scores = []
            for train_idx, test_idx in splits:
                est = clone(base_estimator)
                est.fit(X_rules[train_idx], y_arr[train_idx])
                pred = est.predict(X_rules[test_idx])
                fold_scores.append(float(np.mean(pred == y_arr[test_idx])))

        return float(np.mean(fold_scores))

    def _find_best_alpha(
        self, X_rules: np.ndarray, y: np.ndarray
    ) -> tuple[float, LogisticRegression]:
        """
        Sweep ``n_alphas`` log-spaced L1 penalty values from high → low regularisation.

        At each alpha:
          1. Fit LogisticRegression(C = 1/alpha).
          2. Count non-zero coefficients (|coef| > 1e-6).
          3. If count > max_rules: stop (exceeded budget).
          4. Otherwise: record alpha as valid, compute stratified CV accuracy.

        Return the alpha with the highest CV accuracy among valid candidates.

        Blueprint bugs fixed
        --------------------
        Original code: `best_alpha = alphas[i - 1]` using the loop variable i.
        If the loop completes without breaking (all alphas stay within budget),
        i points to the last element — but if it does break, alphas[i-1] may
        not correspond to the last CV-evaluated alpha.  This causes subtle
        wrong-alpha bugs even when no IndexError occurs.

        Fix: accumulate valid_alphas explicitly.  Use valid_alphas[-1] as
        fallback (weakest regularisation still within budget) when CV is off
        or only one candidate exists.

        Parameters
        ----------
        X_rules : ndarray, shape (n_samples, n_rules)
        y       : ndarray, shape (n_samples,)

        Returns
        -------
        best_alpha : float
        model_at_best : LogisticRegression
            Full-data model from the sweep at the chosen alpha (same object used for
            CV ``clone``); avoids a second SAGA fit that can diverge from the sweep.
        """
        alphas      = np.flip(np.logspace(-4, 4, num=max(2, self.n_alphas), base=10))
        cv_splitter = StratifiedKFold(n_splits=max(2, self.cv_n_splits), shuffle=False)
        coef_thr = self.coef_zero_threshold

        valid_alphas: list[float] = []
        valid_models: list[LogisticRegression] = []
        cv_scores:   list[float]  = []

        for i, alpha in enumerate(alphas):
            model = self._make_logistic_regression(alpha)
            model.fit(X_rules, y)

            n_nonzero = int(np.sum(np.abs(model.coef_.flatten()) > coef_thr))
            if n_nonzero > self.max_rules:
                break                            # budget exceeded — stop sweep

            valid_alphas.append(float(alpha))
            valid_models.append(model)

            if self.cv:
                score = self._mean_stratified_cv_accuracy(
                    model, X_rules, y, cv_splitter
                )
                cv_scores.append(score)

        if not valid_alphas:
            # Every alpha (even the most regularised) exceeds the budget.
            # Fallback: return the strongest regularisation available.
            m0 = self._make_logistic_regression(float(alphas[0]))
            m0.fit(X_rules, y)
            return float(alphas[0]), m0

        if self.cv and len(cv_scores) > 1:
            s = np.asarray(cv_scores, dtype=np.float64)
            # Stabilise CV winner: raw floats can disagree across runs at ~1e-9;
            # round then take max; ties → smallest index (stronger reg in sweep order).
            s_r = np.round(s, 8)
            m = float(np.max(s_r))
            tied = np.flatnonzero(s_r == m)
            best_idx = int(np.min(tied))
            best_alpha = float(valid_alphas[best_idx])
            return best_alpha, valid_models[best_idx]

        # No CV, or only one candidate — return the weakest valid regularisation
        # (most rules within budget), which maximises model expressiveness.
        return float(valid_alphas[-1]), valid_models[-1]

    def _fit_final_model(
        self, X_rules: np.ndarray, y: np.ndarray, alpha: float
    ) -> LogisticRegression:
        """Fit the final L1 logistic regression with the selected alpha."""
        model = self._make_logistic_regression(alpha)
        model.fit(X_rules, y)
        
        return model

    # =========================================================================
    # Public API
    # =========================================================================

    def fit(
        self,
        X,
        y,
        feature_names: list[str] | None = None,
    ) -> "RuleFitClassifierModel":
        """
        Fit the RuleFit model.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
        y : array-like, shape (n_samples,) — binary labels {0, 1}
        feature_names : list[str], optional
            Human-readable column names.  Sanitized internally (spaces →
            underscores) before use in rule strings.  If None, uses
            ["x0", "x1", ...].

        Returns
        -------
        self
        """
        X, y          = check_X_y(X, y)
        self.classes_ = unique_labels(y)

        # Sanitize feature names (fix blueprint Bug 5)
        raw_names              = (list(feature_names) if feature_names is not None
                                  else [f"x{i}" for i in range(X.shape[1])])
        self._raw_feature_names = raw_names
        self.feature_names_     = self._sanitize_names(raw_names)

        _blas_ctx = (
            _threadpool_limits(limits=self.blas_threads, user_api="blas")
            if (self.blas_threads is not None and _threadpool_limits is not None)
            else contextlib.nullcontext()
        )

        # GBM + rule matrix + Phase 4 all use BLAS; limit threads for whole fit so
        # tree growth order matches across runs (previously only Phase 4 was limited).
        _rs_np = np.random.get_state()
        _rs_py = random.getstate()
        try:
            np.random.seed(int(self.random_state))
            random.seed(int(self.random_state))
            # Store data for analysis methods (stability_test)
            self._X_fit = X.copy()
            self._y_fit = y.copy()
            
            with _blas_ctx:
                # ── Phase 1: GBM Trees ──────────────────────────────────────
                self.gbm_ = self._generate_trees(X, y.astype(float))

                # ── Phase 2: Rules ──────────────────────────────────────────
                self.rules_ = self._extract_unique_rules(self.gbm_, self.feature_names_)

                # ── Phase 3: Binary Matrix ──────────────────────────────────
                X_rules = self._rules_to_binary_matrix(
                    X, self.rules_, self.feature_names_, n_jobs=self._resolve_n_jobs(),
                )
                if self.include_linear:
                    # Winsorize linear features to mitigate outliers (Friedman, 2008)
                    wf = float(self.winsor_fract)
                    if wf > 0:
                        self.linear_winsor_q_low_ = np.percentile(X, 100 * wf, axis=0)
                        self.linear_winsor_q_high_ = np.percentile(X, 100 * (1 - wf), axis=0)
                    else:
                        self.linear_winsor_q_low_ = np.min(X, axis=0)
                        self.linear_winsor_q_high_ = np.max(X, axis=0)
                    
                    X_lin = self._winsorize(X, self.linear_winsor_q_low_, self.linear_winsor_q_high_)
                    
                    # Scale linear features to match rule indicator scales (std ~ 0.4)
                    # This stabilizes the L1 coefficients and speeds up convergence.
                    stds = np.std(X_lin, axis=0)
                    self.linear_scales_ = np.where(stds > 1e-12, 0.4 / stds, 1.0)
                    X_lin_scaled = X_lin * self.linear_scales_
                    
                    X_rules = np.hstack([X_lin_scaled, X_rules])
                
                self._n_rule_cols = len(self.rules_)   # remember offset for include_linear
                # Dense contiguous float64 reduces cross-run AMO/BLAS ordering drift in SAGA.
                X_rules_fit = np.ascontiguousarray(X_rules, dtype=np.float64)
                y_fit = np.asarray(y).astype(np.float64, copy=False)

                # ── Phase 4: Alpha Search + Logistic Fit ────────────────────
                if (self.logistic_penalty or "l1").lower() == "elasticnet" and self.logistic_solver != "saga":
                    warnings.warn(
                        "logistic_penalty='elasticnet' requires solver='saga' in scikit-learn. "
                        f"logistic_solver={self.logistic_solver!r} is ignored for Phase 4 (elastic net).",
                        UserWarning,
                        stacklevel=2,
                    )
                if self.refit_on_selected_rules and self._effective_logistic_solver() == "liblinear":
                    warnings.warn(
                        "refit_on_selected_rules with logistic_solver='liblinear' (L1 only): the "
                        "selected rule identity (H4 selected_rules_sig) can differ across identical reruns. "
                        "Stage-2 L2 refit only reweights the rules L1 kept. "
                        "Prefer logistic_solver='saga' or logistic_penalty='elasticnet' (also uses saga) "
                        "when you need the same rules between runs.",
                        UserWarning,
                        stacklevel=2,
                    )
                self.best_alpha_, self.lasso_model_ = self._find_best_alpha(
                    X_rules_fit, y_fit
                )

                # L1 selection: which rule columns survive thresholding
                coefs_l1 = self.lasso_model_.coef_.flatten()
                rule_coefs = coefs_l1[: self._n_rule_cols]
                if self.l1_mask_round_decimals is not None:
                    coef_for_mask = np.round(
                        rule_coefs, int(self.l1_mask_round_decimals)
                    ).astype(np.float64, copy=False)
                else:
                    coef_for_mask = rule_coefs
                coef_abs = np.abs(coef_for_mask)
                thr = float(self.coef_zero_threshold)
                pol = (self.l1_selection_policy or "threshold").lower()
                rank_rd: int | None = None
                if pol == "top_k_lexsort":
                    if self.l1_rank_round_decimals is None:
                        _base = (
                            4
                            if self.l1_mask_round_decimals is None
                            else int(self.l1_mask_round_decimals)
                        )
                        rank_rd = max(0, _base - 1)
                    else:
                        rank_rd = int(self.l1_rank_round_decimals)
                    scores = np.round(
                        coef_abs.astype(np.float64, copy=False), rank_rd
                    )
                    nfeat = int(scores.size)
                    cap = min(int(self.max_rules), nfeat)
                    idx_axis = np.arange(nfeat, dtype=np.intp)
                    # Primary key -scores ascending => largest |coef| first (order[0] is best).
                    order = np.lexsort((idx_axis, -scores))
                    mask = np.zeros(nfeat, dtype=bool)
                    for rank in range(cap):
                        i = int(order[rank])
                        if float(scores[i]) <= thr:
                            break
                        mask[i] = True
                elif pol == "threshold":
                    mask = coef_abs > thr
                else:
                    raise ValueError(
                        "l1_selection_policy must be 'threshold' or 'top_k_lexsort', "
                        f"got {self.l1_selection_policy!r}"
                    )
                selected_rules_list = [r for r, m in zip(self.rules_, mask) if m]

                self.selected_rules_ = selected_rules_list
                self._refit_subset_only = False
                self.selected_coefs_ = rule_coefs[mask]
                self.n_rules_total_ = len(self.rules_)
                self.n_rules_selected_ = len(self.selected_rules_)

                # ── Optional stage 2: plain logistic on L1-selected columns only ─
                if self.refit_on_selected_rules:
                    if self.include_linear:
                        warnings.warn(
                            "refit_on_selected_rules is ignored when include_linear=True.",
                            RuntimeWarning,
                            stacklevel=2,
                        )
                    else:
                        idx = np.flatnonzero(np.asarray(mask, dtype=bool))
                        if idx.size > 0:
                            X_sub = X_rules_fit[:, idx]
                            rmi = self.refit_max_iter if self.refit_max_iter is not None else self.max_iter
                            if self.refit_penalty == "none":
                                l2 = LogisticRegression(
                                    penalty="none",
                                    solver=self.refit_solver,
                                    random_state=self.random_state,
                                    max_iter=int(rmi),
                                    tol=1e-4,
                                )
                            else:
                                l2 = LogisticRegression(
                                    penalty=self.refit_penalty,
                                    C=float(self.refit_C),
                                    solver=self.refit_solver,
                                    random_state=self.random_state,
                                    max_iter=int(rmi),
                                    tol=1e-4,
                                )
                            l2.fit(X_sub, y_fit)
                            self.lasso_model_ = l2
                            self.selected_rule_indices_ = idx
                            self._refit_subset_only = True
                            self.selected_coefs_ = l2.coef_.flatten()
        finally:
            np.random.set_state(_rs_np)
            random.setstate(_rs_py)

        return self

    def _build_feature_matrix(self, X: np.ndarray) -> np.ndarray:
        """Internal helper: validate + transform X into the rule feature matrix."""
        check_is_fitted(self, ["rules_", "lasso_model_"])
        X       = check_array(X)
        X_rules = self._rules_to_binary_matrix(
            X, self.rules_, self.feature_names_, n_jobs=self._resolve_n_jobs(),
        )
        if self.include_linear:
            # Apply same winsorization and scaling as during fit
            X_lin = self._winsorize(X, self.linear_winsor_q_low_, self.linear_winsor_q_high_)
            X_lin_scaled = X_lin * self.linear_scales_
            X_rules = np.hstack([X_lin_scaled, X_rules])
        X_rules = np.ascontiguousarray(X_rules, dtype=np.float64)
        if getattr(self, "_refit_subset_only", False):
            X_rules = X_rules[:, self.selected_rule_indices_]
        return X_rules

    def predict_proba(self, X) -> np.ndarray:
        """
        Predict class probabilities.

        Blueprint bug
        -------------
        Original implementation:
            raw_score = X_rules @ coefs + intercept
            logits    = np.vstack((1 - raw_score, raw_score)).T
            return softmax(logits, axis=1)

        This is *wrong*.  Logistic regression maps a scalar logit z through
        the sigmoid function: p(y=1) = 1 / (1 + exp(-z)).  Constructing a
        2-vector (1-z, z) and applying softmax over it does NOT equal sigmoid(z)
        unless z happens to equal the log-odds, which it doesn\'t here because
        raw_score is already a linear combination, not a logit.

        Fix: delegate to sklearn\'s LogisticRegression.predict_proba(), which
        computes sigmoid(X @ w + b) correctly and returns calibrated class
        probabilities.

        Returns
        -------
        proba : ndarray, shape (n_samples, 2)
            Columns are P(y=0) and P(y=1).
        """
        return self.lasso_model_.predict_proba(self._build_feature_matrix(X))

    def predict(self, X) -> np.ndarray:
        """
        Predict class labels.

        Returns
        -------
        y_pred : ndarray, shape (n_samples,)
        """
        return self.lasso_model_.predict(self._build_feature_matrix(X))

    def score(self, X, y) -> float:
        """Return mean accuracy on (X, y)."""
        return float(np.mean(self.predict(X) == y))

    def get_rules(self, top_n: int | None = None) -> pd.DataFrame:
        """
        Return selected rules as a DataFrame sorted by |coefficient|.

        Columns
        -------
        rule        : str   — human-readable rule condition string
        coefficient : float — L1 logistic regression weight (positive → class 1)
        abs_coef    : float — |coefficient|, used for sorting

        Parameters
        ----------
        top_n : int, optional
            Return only the top_n rules by |coefficient|.  None → all.

        Returns
        -------
        pd.DataFrame, shape (n_selected_rules, 3) or (top_n, 3)
        """
        check_is_fitted(self)
        df = pd.DataFrame({
            "rule"       : self.selected_rules_,
            "coefficient": self.selected_coefs_,
            "abs_coef"   : np.abs(self.selected_coefs_),
        }).sort_values("abs_coef", ascending=False).reset_index(drop=True)
        return df if top_n is None else df.head(top_n)

    def stability_test(
        self,
        subsample_fract: float = 0.75,
        n_iterations: int = 100,
        stability_threshold: float | None = None,
        n_threshold: int = 10,
        random_state: int = 0,
        n_jobs: int = 1,
        verbose: bool = False,
        round_thresholds: int | None = 4,
        min_consistency: float = 0.9
    ) -> pd.DataFrame:
        """
        Perform stability selection test by repeatedly fitting the model on subsamples.

        Parameters
        ----------
        subsample_fract : float, default=0.75
        n_iterations : int, default=100
        stability_threshold : float, optional
            Minimum selection frequency to keep a rule.
        n_threshold : int, default=10
            Minimum number of iterations to keep a rule (acts as a count-based threshold).
        random_state : int, default=0
        n_jobs : int, default=1
            Number of parallel jobs.
        verbose : bool, default=False
            Whether to show progress.
        round_thresholds : int, optional, default=4
            Round rule thresholds to this many decimal places.
        min_consistency : float, default=0.9
            Minimum sign consistency to be considered 'confirmed'.

        Returns
        -------
        report : pd.DataFrame
            Stability metrics for all candidate rules.
        """
        check_is_fitted(self, ["rules_", "best_alpha_", "_X_fit", "_y_fit"])
        X, y = self._X_fit, self._y_fit
        n_samples = X.shape[0]
        n_sub = int(n_samples * subsample_fract)
        
        # Round thresholds in rules if requested
        rules_to_use = self.rules_
        if round_thresholds is not None:
            def round_val_match(m):
                # group(1) is the operator part, group(2) is the value part
                v = float(m.group(2))
                rounded = format(v, f'.{round_thresholds}f').rstrip('0').rstrip('.')
                return m.group(1) + (rounded if rounded != "" else "0")
            
            rules_to_use = [
                re.sub(r"([=<>]\s*)([-+]?\d*\.?\d+([eE][-+]?\d+)?)", round_val_match, r) 
                for r in self.rules_
            ]

        # All predictors (linear features + rules)
        predictor_names = []
        if self.include_linear:
            predictor_names.extend(self.feature_names_)
        predictor_names.extend(rules_to_use)

        # Track coefficients for each predictor across iterations
        self.stability_coefs_ = {name: [] for name in predictor_names}
        self.stability_intercepts_ = []
        
        X_rules_all = self._rules_to_binary_matrix(X, rules_to_use, self.feature_names_, n_jobs=self._resolve_n_jobs())
        if self.include_linear:
            X_lin = self._winsorize(X, self.linear_winsor_q_low_, self.linear_winsor_q_high_)
            X_lin_scaled = X_lin * self.linear_scales_
            X_all = np.hstack([X_lin_scaled, X_rules_all])
        else:
            X_all = X_rules_all

        # Prepare parameters for workers
        builder_params = {
            'penalty': (self.logistic_penalty or "l1").lower(),
            'l1_ratio': float(self.elasticnet_l1_ratio),
            'max_iter': int(self.max_iter),
            'tol': max(float(self.solver_tol), float(self.saga_tol_floor)) if (self.logistic_penalty or "l1").lower() == "elasticnet" else float(self.solver_tol),
            'solver': self.logistic_solver
        }
        # offset is 0 if we including all predictors in the tracker
        # All coefs corresponds to predictor_names in order

        if n_jobs != 1 and _HAS_JOBLIB:
            results = Parallel(n_jobs=n_jobs, verbose=10 if verbose else 0)(
                delayed(_run_stability_iteration)(
                    i, X_all, y, n_sub, random_state, builder_params, self.best_alpha_, self.coef_zero_threshold, 0 # offset 0 to get all coefs
                )
                for i in range(n_iterations)
            )
        else:
            results = []
            for i in range(n_iterations):
                if verbose:
                    print(f"Iteration {i+1}/{n_iterations}...")
                res = _run_stability_iteration(
                    i, X_all, y, n_sub, random_state, builder_params, self.best_alpha_, self.coef_zero_threshold, 0
                )
                results.append(res)

        # Aggregate results
        for iteration_coefs, iteration_intercept in results:
            self.stability_intercepts_.append(iteration_intercept)
            for j, c in enumerate(iteration_coefs):
                if np.abs(c) > self.coef_zero_threshold:
                    name = predictor_names[j]
                    self.stability_coefs_[name].append(float(c))

        self.average_stability_intercept_ = float(np.mean(self.stability_intercepts_))
        if verbose:
            print(f"Average stability intercept: {self.average_stability_intercept_:.4f}")

        # Calculate metrics
        results = []
        optimal_rules_set = set(self.selected_rules_)
        
        # We need to correctly identify which ones are linear vs rules for the status/type
        for j, name in enumerate(predictor_names):
            is_linear = j < (X.shape[1] if self.include_linear else 0)
            orig_name = self.feature_names_[j] if is_linear else self.rules_[j - (X.shape[1] if self.include_linear else 0)]
            
            coefs_list = self.stability_coefs_[name]
            sel_count = len(coefs_list)
            freq = sel_count / n_iterations
            
            if sel_count > 0:
                mean_c = np.mean(coefs_list)
                std_c = np.std(coefs_list)
                signs = np.sign(coefs_list)
                mean_sign = np.sign(mean_c)
                consistency = np.mean(signs == mean_sign) if mean_sign != 0 else 0.0
            else:
                mean_c = 0.0
                std_c = 0.0
                consistency = 0.0
            
            # For in_optimal_model, we check if it was selected in main fit
            # Lasso model coefs include linear features at start if include_linear
            in_opt = False
            if hasattr(self, "lasso_model_"):
                main_coefs = self.lasso_model_.coef_.flatten()
                if np.abs(main_coefs[j]) > self.coef_zero_threshold:
                    in_opt = True
            
            # Determine status
            if freq >= (stability_threshold or 0.0) and sel_count >= n_threshold:
                if consistency >= min_consistency:
                    status = 'confirmed'
                else:
                    status = 'unstable_selected' if in_opt else 'unstable_not_selected'
            else:
                status = 'stable_not_selected' if freq < 0.1 else 'unstable_not_selected'
                if in_opt:
                    status = 'unstable_selected'

            results.append({
                "rule": name,
                "type": 'linear' if is_linear else 'rule',
                "selection_frequency": freq,
                "mean_coef": mean_c,
                "std_coef": std_c,
                "sign_consistency": consistency,
                "in_optimal_model": in_opt,
                "status": status,
                "selection_count": sel_count
            })
            
        self.stability_report_full_ = pd.DataFrame(results)
        
        # Filtering logic
        mask = np.ones(len(self.stability_report_full_), dtype=bool)
        if stability_threshold is not None:
            mask &= (self.stability_report_full_["selection_frequency"] >= stability_threshold)
        if n_threshold is not None:
            mask &= (self.stability_report_full_["selection_count"] >= n_threshold)
            
        self.stability_report_ = self.stability_report_full_[mask].sort_values("selection_frequency", ascending=False).reset_index(drop=True)
        return self.stability_report_

    def plot_stability(self, stability_report: pd.DataFrame | None = None, stability_threshold: float | None = None, top_n: int = 40):
        """
        Plot selection probabilities from the stability test.

        Parameters
        ----------
        stability_report : pd.DataFrame, optional
            A report returned from stability_test. If None, uses self.stability_report_.
        stability_threshold : float, optional
            Add a vertical line for reference.
        top_n : int, default=40
            Show only top n rules/features.
        """
        df = stability_report if stability_report is not None else getattr(self, "stability_report_", None)
        if df is None:
            raise ValueError("No stability report found. Run stability_test first.")
        
        if df.empty:
            print("Stability report is empty. Nothing to plot.")
            return

        # Prepare data
        df = df.sort_values("selection_frequency", ascending=True) # Ascending for horizontal bar or just top-n?
        # Let's do a vertical bar plot if it's not too many rules, or just top rules.
        if len(df) > 40:
            df = df.tail(40) # Show top 40 most frequent

        fig, ax = plt.subplots(figsize=(12, 8))
        
        y_pos = np.arange(len(df))
        heights = df["selection_frequency"]
        
        # Bar color based on sign consistency
        # We don't have min_consistency here, so we default to 0.9 or try to infer?
        # Let's just use 0.9 as the visual benchmark for green.
        colors = ['#2ecc71' if c >= 0.9 else '#e67e22' for c in df["sign_consistency"]]
        
        # Hatching for optimal model rules
        hatches = ['///' if opt else '' for opt in df["in_optimal_model"]]
        
        # Labels: prefix linear features with 'L: '
        labels = [
            (f"L: {row['rule']}" if row['type'] == 'linear' else row['rule'])
            for _, row in df.iterrows()
        ]
        
        bars = ax.barh(y_pos, heights, color=colors, edgecolor='black', linewidth=0.5)
        
        for bar, hatch in zip(bars, hatches):
            bar.set_hatch(hatch)

        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_xlabel("Selection Frequency")
        ax.set_title("Stability Selection Results")
        
        if stability_threshold is not None:
            ax.axvline(x=stability_threshold, color='red', linestyle='--', label=f'Threshold ({stability_threshold})')

        # Legend
        green_patch = mpatches.Patch(color='#2ecc71', label='Consistent Sign (>= 90%)')
        orange_patch = mpatches.Patch(color='#e67e22', label='Mixed Sign (< 90%)')
        hatch_patch = mpatches.Patch(facecolor='white', edgecolor='black', hatch='///', label='In Optimal Model')
        
        handles = [green_patch, orange_patch, hatch_patch]
        if stability_threshold is not None:
            handles.append(plt.Line2D([0], [0], color='red', linestyle='--', label=f'Threshold ({stability_threshold})'))
            
        ax.legend(handles=handles, loc='lower right')
        
        plt.tight_layout()
        plt.show()

    def summary(self) -> None:
        """Print a compact fit summary."""
        check_is_fitted(self)
        print(f"  Candidate rules (post-dedup)  : {self.n_rules_total_:>6,}")
        print(f"  Selected rules (L1 non-zero)  : {self.n_rules_selected_:>6,}")
        print(f"  Best alpha (L1 strength)      : {self.best_alpha_:.6f}")
        print(f"  Best C  (= 1 / alpha)         : {1.0 / self.best_alpha_:.4f}")
        print(f"  Intercept                     : {self.lasso_model_.intercept_[0]:.4f}")


# =============================================================================
# Demo — run this file directly to verify the implementation
# =============================================================================

# import os
# os.makedirs("output", exist_ok=True)
# with open("output/custom_rulefit.py", "w") as f:
#     f.write(complete_code)

# lines = complete_code.count("\n")
# print(f"Saved output/custom_rulefit.py  ({lines} lines)")