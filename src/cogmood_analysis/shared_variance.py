"""Estimate the generalizable shared-variance ceiling between two views.

The two views are:

* **View A (cognitive)** - the 30 fitted cognitive-model parameters
  (BART / RDM / CAB / FLKR), excluding fit diagnostics
  (``sub_score`` / ``map_dif`` / ``abs_map_dif`` / ``max_rhat``).
* **View B (symptom)** - the baseline survey subscale and summary scores
  (BAARS / GAD-7 / PHQ-8 / HiTOP), derived from
  :data:`cogmood_analysis.survey_helpers.SCALES`.

The estimation strategy is "FM-embedding CCA": each view is embedded
independently with a frozen tabular foundation model (TabPFN v2), and a
regularized CCA is fit on the embeddings. Everything is fit on a training
split and evaluated on a held-out split, repeated over many splits, with a
permutation null. We report *held-out* canonical correlations only.

Leakage discipline (important):

* Each view is embedded in complete isolation - the embedder for one view
  never sees the other view.
* TabPFN embeddings are target-conditioned and there is no working
  unsupervised mode, so we use a **within-view pseudo-target**: the first
  principal component of that view's *training* columns. This keeps the
  embedding self-supervised within-view and free of cross-view information.
* Scalers, pseudo-target PCA, embedding context, post-embedding PCA, and the
  CCA are all fit on TRAIN only and applied to TEST.

The resulting number is a *representation/model-conditional* estimate of
achievable shared variance, not an absolute maximum.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence
import os

import numpy as np
import polars as pl
from numpy.typing import NDArray
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler

from . import survey_helpers as sh

# --- View definitions -------------------------------------------------------

#: Fitted cognitive-model parameters, by task (fit diagnostics excluded).
COGNITIVE_PARAMS: dict[str, tuple[str, ...]] = {
    "bart": ("alpha", "gamma_neg", "gamma_pos", "tau", "theta"),
    "rdm": ("v0", "w_d", "w_s", "t0", "sigma", "sigma_timer", "v_timer"),
    "cab": ("a", "alpha", "gamma", "kappa", "lam", "nu", "rho", "t0", "tau", "w"),
    "flkr": ("r", "p", "sd0", "K", "L", "thresh", "alpha", "t0"),
}

#: Column names for View A (cognitive), e.g. ``bart__alpha``.
VIEW_A_COLUMNS: list[str] = [
    f"{task}__{param}" for task, params in COGNITIVE_PARAMS.items() for param in params
]


def _scale_columns_from_scales() -> list[str]:
    """Build the baseline symptom score column names from ``sh.SCALES``.

    Each ``(scale, None)`` entry contributes the summary column ``scale`` and
    each ``(scale, subscale)`` entry contributes ``scale_subscale``. The
    attention-check pseudo-scale ``attnbin`` is dropped. ``baars_inattentive``
    is appended because it is a genuine symptom subscale present in the data
    even though it is absent from ``SCALES``.
    """
    cols: list[str] = []
    for scale, subscale in sh.SCALES:
        if scale == "attnbin":
            continue
        cols.append(scale if subscale is None else f"{scale}_{subscale}")
    if "baars_inattentive" not in cols:
        cols.append("baars_inattentive")
    # de-duplicate while preserving order
    seen: set[str] = set()
    return [c for c in cols if not (c in seen or seen.add(c))]


#: Candidate column names for View B (symptom). Intersected with the CSV
#: columns at load time (some HiTOP subscales lack a summary column).
VIEW_B_COLUMNS: list[str] = _scale_columns_from_scales()


# --- Data loading -----------------------------------------------------------


#: Column used to stratify the train/test splits.
STRATA_COLUMN = "backfilled_prolific_screen_group"


@dataclass
class Views:
    """Row-aligned, complete-case views ready for analysis."""

    A: NDArray[np.float64]  # cognitive, shape (n, p_A)
    B: NDArray[np.float64]  # symptom, shape (n, p_B)
    sub_ids: NDArray[np.str_]
    strata: NDArray[np.str_]  # stratification labels, shape (n,)
    a_columns: list[str]
    b_columns: list[str]
    n_total: int  # rows before complete-case filtering


#: Tasks whose ``{task}__max_rhat`` columns gate model-fit convergence.
RHAT_TASKS = ("bart", "rdm", "cab", "flkr")


def load_views(
    csv_path: str | os.PathLike,
    a_columns: Sequence[str] | None = None,
    b_columns: Sequence[str] | None = None,
    strata_column: str = STRATA_COLUMN,
    exclude_rhat_above: float | None = None,
    rhat_tasks: Sequence[str] = RHAT_TASKS,
) -> Views:
    """Load and complete-case the two views from ``training_data.csv``.

    Parameters
    ----------
    csv_path : path to ``training_data.csv``.
    a_columns, b_columns : optional column overrides. Defaults select the
        cognitive parameters (View A) and baseline symptom scores (View B).
        Requested columns missing from the CSV are dropped with the remaining
        ones used (so e.g. HiTOP subscales without a summary column are
        skipped silently rather than erroring).
    strata_column : column whose values stratify the train/test splits
        (default ``backfilled_prolific_screen_group``). Missing values are
        labelled ``"missing"`` so they form their own stratum.
    exclude_rhat_above : if set (e.g. ``1.1``), drop subjects whose
        ``{task}__max_rhat`` exceeds this threshold on *any* task in
        ``rhat_tasks`` (model-fit convergence exclusion). ``None`` (default)
        keeps every complete-case subject, matching the original behavioral-only
        exclusion.
    rhat_tasks : tasks whose ``max_rhat`` columns are checked when
        ``exclude_rhat_above`` is set.

    Returns
    -------
    Views with ``A``, ``B`` as float arrays, aligned ``sub_ids`` and ``strata``,
    the resolved column lists, and the pre-filter row count.
    """
    csv_path = Path(csv_path)
    df = pl.read_csv(csv_path, infer_schema_length=10000)
    available = set(df.columns)

    a_cols = [c for c in (a_columns or VIEW_A_COLUMNS) if c in available]
    b_cols = [c for c in (b_columns or VIEW_B_COLUMNS) if c in available]
    if not a_cols:
        raise ValueError("No View A (cognitive) columns found in the CSV.")
    if not b_cols:
        raise ValueError("No View B (symptom) columns found in the CSV.")
    if strata_column not in available:
        raise ValueError(f"Strata column {strata_column!r} not found in the CSV.")

    n_total = df.height

    # optional model-fit convergence exclusion (any task max_rhat over threshold)
    if exclude_rhat_above is not None:
        rhat_cols = [f"{t}__max_rhat" for t in rhat_tasks if f"{t}__max_rhat" in available]
        if not rhat_cols:
            raise ValueError("exclude_rhat_above set but no {task}__max_rhat columns found.")
        keep = pl.all_horizontal(
            [pl.col(c) <= exclude_rhat_above for c in rhat_cols]
        )
        df = df.filter(keep)

    # cast the analysis columns to float, then complete-case on both views
    sub = df.select(["sub_id", strata_column, *a_cols, *b_cols]).with_columns(
        [pl.col(c).cast(pl.Float64, strict=False) for c in (*a_cols, *b_cols)]
    )
    sub = sub.drop_nulls(subset=[*a_cols, *b_cols])

    A = sub.select(a_cols).to_numpy().astype(np.float64)
    B = sub.select(b_cols).to_numpy().astype(np.float64)
    sub_ids = sub.select("sub_id").to_numpy().ravel().astype(str)
    strata = (
        sub.select(pl.col(strata_column).fill_null("missing"))
        .to_numpy()
        .ravel()
        .astype(str)
    )
    return Views(
        A=A, B=B, sub_ids=sub_ids, strata=strata,
        a_columns=a_cols, b_columns=b_cols, n_total=n_total,
    )


# --- Embedding --------------------------------------------------------------


def _standardize(
    train: NDArray[np.float64], test: NDArray[np.float64]
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Standardize ``test`` using statistics fit on ``train`` only."""
    scaler = StandardScaler().fit(train)
    return scaler.transform(train), scaler.transform(test)


def embed_view_raw(
    X_train: NDArray[np.float64], X_test: NDArray[np.float64]
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Control embedder: just standardize (no foundation model)."""
    return _standardize(X_train, X_test)


def embed_view_fm(
    X_train: NDArray[np.float64],
    X_test: NDArray[np.float64],
    reduce: str = "mean",
    pseudo_target: str = "pca1",
    train_data_source: str = "test",
    device: str = "auto",
    ignore_pretraining_limits: bool = True,
    seed: int = 0,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Embed a single view with TabPFN, fit on TRAIN and applied to TEST.

    Uses the native ``TabPFNRegressor.get_embeddings`` API (tabpfn 2.x), which
    needs no license token. A within-view pseudo-target is used so the
    embedding carries no cross-view information (see module docstring): the
    model is fit on the training view with that pseudo-target, train
    embeddings are read with ``data_source="train"`` and test embeddings with
    ``data_source="test"`` (the training set is the frozen in-context
    reference; test labels are never seen). ``tabpfn`` is imported lazily so
    the rest of the module (and its tests) work without it installed.

    Parameters
    ----------
    X_train, X_test : the view's raw feature matrices.
    reduce : how to collapse the ``(n_estimators, n, d)`` output - ``"mean"``
        or ``"first"``. The same rule is applied to train and test.
    pseudo_target : ``"pca1"`` (first PC of the training view) or ``"col0"``
        (first standardized column) - used only as the supervised target the
        embedder requires; report sensitivity to this choice.
    train_data_source : ``data_source`` passed when embedding the TRAIN rows.
        ``"test"`` (default) embeds train rows in the same query regime as the
        held-out rows so the two embedding distributions match - this transfers
        markedly better than ``"train"``.

        CAVEAT (support/query asymmetry): the in-context reference is the fitted
        ``X_train`` (with its within-view pseudo-target). TRAIN rows are therefore
        queried against a context that *contains labeled copies of themselves*,
        whereas TEST rows are absent from their context. This can leak the
        within-view pseudo-target into the train representations and optimistically
        bias the CCA fit for the FM arms. It is a within-view effect only (the
        pseudo-target carries no cross-view information), and because the FM arms
        do not beat the linear ``raw`` baseline the direction is conservative for
        the study's conclusion. A leakage-free fix would require per-row
        leave-one-out contexts (not supported by the TabPFN embedding API) and is
        not implemented; the caveat is reported instead.
    device : ``"auto"`` (use CUDA if available, else CPU), ``"cpu"`` or
        ``"cuda"``.
    ignore_pretraining_limits : pass through to ``TabPFNRegressor``. Required to
        embed >1000 rows on CPU (TabPFN otherwise refuses for performance).
    seed : random seed for the embedder.

    Returns
    -------
    (E_train, E_test) as 2D float arrays.
    """
    from tabpfn import TabPFNRegressor  # lazy import

    if device == "auto":
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"

    Xtr, Xte = _standardize(X_train, X_test)

    if pseudo_target == "pca1":
        y = PCA(n_components=1, random_state=seed).fit_transform(Xtr).ravel()
    elif pseudo_target == "col0":
        y = Xtr[:, 0].copy()
    else:
        raise ValueError(f"unknown pseudo_target {pseudo_target!r}")

    reg = TabPFNRegressor(
        device=device,
        random_state=seed,
        ignore_pretraining_limits=ignore_pretraining_limits,
    )
    reg.fit(Xtr, y)
    E_tr = np.asarray(reg.get_embeddings(Xtr, data_source=train_data_source))
    E_te = np.asarray(reg.get_embeddings(Xte, data_source="test"))
    return _reduce_embedding(E_tr, reduce), _reduce_embedding(E_te, reduce)


# --- Google TabFM embeddings ------------------------------------------------

#: TabFM per-row representation dimensionality (the ``reps`` fed to the ICL head).
TABFM_EMBED_DIM = 2048
_TABFM_REPO = "google/tabfm-1.0.0-pytorch"
_TABFM_CACHE: dict[tuple[str, str], Any] = {}


def _load_tabfm(device: str, model_type: str = "regression"):
    """Load Google TabFM (`google/tabfm-1.0.0-pytorch`), cached per device.

    The HF repo ships ``model.safetensors`` but the packaged ``load`` expects a
    ``pytorch_model.bin``, so we construct the model from its config and load the
    safetensors state dict directly.
    """
    key = (device, model_type)
    if key in _TABFM_CACHE:
        return _TABFM_CACHE[key]
    import os

    from huggingface_hub import snapshot_download
    from safetensors.torch import load_file
    from tabfm.src.pytorch.tabfm_v1_0_0 import (
        ClassificationConfig,
        RegressionConfig,
        TabFM,
    )

    cfg = RegressionConfig() if model_type == "regression" else ClassificationConfig()
    model = TabFM(**cfg.to_dict())
    base = snapshot_download(repo_id=_TABFM_REPO)
    model.load_state_dict(
        load_file(os.path.join(base, model_type, "model.safetensors")), strict=True
    )
    model.to(device).eval()
    _TABFM_CACHE[key] = model
    return model


def tabfm_prefix(model, x, y, train_size):
    """TabFM forward up to (but excluding) the final ``row_interactor_2`` block.

    Returns the tensor fed into ``row_interactor_2``. Splitting the forward here
    lets a fine-tuner run this frozen bulk under ``no_grad`` and train only the
    last block + head - essential given TabFM is ~1.6B params. ``x`` is
    ``(B, T, H)``, ``y`` is ``(B, T)`` with query rows padded by ``-100.0``,
    ``train_size`` is a ``(B,)`` tensor.
    """
    import torch

    emb = model.cell_embedder(x, y, train_size, None, d=None)
    emb = model.col_embedder(emb, train_size)
    b, t, _, _ = emb.shape
    cls = model.cls_tokens.expand(b, t, -1, -1)
    emb = torch.cat([cls, emb], dim=2)
    emb = model.row_interactor(emb, d=None)
    return model.col_embedder_2(emb, train_size)


def tabfm_reps(model, x, y, train_size):
    """Per-row TabFM representations (the ``reps`` fed to the ICL head).

    Full forward up to ``row_interactor_2`` (prefix + final block). See
    :func:`tabfm_prefix` for the arg conventions.
    """
    return model.row_interactor_2(tabfm_prefix(model, x, y, train_size), d=None)


def tabfm_query_embeddings(model, context_X, context_y, query_X, device, grad=False):
    """TabFM row embeddings for ``query_X`` against the ``context_X`` in-context set.

    ``context_X``/``query_X`` are standardized float arrays; ``context_y`` is the
    context pseudo-target. Returns a ``(n_query, TABFM_EMBED_DIM)`` tensor (kept on
    ``device``; caller detaches/moves as needed). No query labels are used.
    """
    import torch

    n_ctx = context_X.shape[0]
    Xcat = np.concatenate([context_X, query_X], axis=0).astype(np.float32)
    y = np.concatenate(
        [context_y.astype(np.float32), np.full(len(query_X), -100.0, np.float32)]
    )
    x_t = torch.tensor(Xcat, device=device).unsqueeze(0)
    y_t = torch.tensor(y, device=device).unsqueeze(0)
    ts = torch.tensor([n_ctx], device=device)
    with (torch.enable_grad() if grad else torch.no_grad()):
        reps = tabfm_reps(model, x_t, y_t, ts)
    return reps[0, n_ctx:]  # (n_query, TABFM_EMBED_DIM)


def embed_view_tabfm(
    X_train: NDArray[np.float64],
    X_test: NDArray[np.float64],
    pseudo_target: str = "pca1",
    device: str = "auto",
    seed: int = 0,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Embed one view with Google TabFM (frozen), fit on TRAIN, applied to TEST.

    Mirrors :func:`embed_view_fm` but uses TabFM's per-row representation
    (dim ``TABFM_EMBED_DIM``). TabFM is target-conditioned (its cell embedder
    takes ``y``), so a within-view PC1 pseudo-target is used - each view is
    embedded in isolation, no cross-view leakage. Train rows are embedded as
    queries against the train context, the same regime as the test rows.
    """
    import torch

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    Xtr, Xte = _standardize(X_train, X_test)
    Xtr = Xtr.astype(np.float32)
    Xte = Xte.astype(np.float32)
    if pseudo_target == "pca1":
        y = PCA(1, random_state=seed).fit_transform(Xtr).ravel().astype(np.float32)
    elif pseudo_target == "col0":
        y = Xtr[:, 0].astype(np.float32).copy()
    else:
        raise ValueError(f"unknown pseudo_target {pseudo_target!r}")
    model = _load_tabfm(device)
    E_tr = tabfm_query_embeddings(model, Xtr, y, Xtr, device, grad=False)
    E_te = tabfm_query_embeddings(model, Xtr, y, Xte, device, grad=False)
    return (
        E_tr.detach().cpu().numpy().astype(np.float64),
        E_te.detach().cpu().numpy().astype(np.float64),
    )


def _reduce_embedding(E: NDArray[np.float64], reduce: str) -> NDArray[np.float64]:
    """Collapse a TabPFN embedding to 2D ``(n_samples, d)``."""
    E = np.asarray(E, dtype=np.float64)
    if E.ndim == 2:
        return E
    if E.ndim == 3:  # (n_estimators, n_samples, d)
        if reduce == "mean":
            return E.mean(axis=0)
        if reduce == "first":
            return E[0]
        raise ValueError(f"unknown reduce {reduce!r}")
    raise ValueError(f"unexpected embedding ndim {E.ndim}")


# --- CCA on embeddings ------------------------------------------------------


def _pca_reduce(
    E_train: NDArray[np.float64], E_test: NDArray[np.float64], n_pca: int, seed: int
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """PCA-reduce embeddings, fit on TRAIN only. ``n_pca`` is capped."""
    n_comp = min(n_pca, E_train.shape[1], E_train.shape[0] - 1)
    pca = PCA(n_components=n_comp, random_state=seed).fit(E_train)
    return pca.transform(E_train), pca.transform(E_test)


def _heldout_corr(
    scores_a: NDArray[np.float64], scores_b: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Per-dimension Pearson correlation between paired canonical scores."""
    k = scores_a.shape[1]
    out = np.empty(k, dtype=np.float64)
    for i in range(k):
        a = scores_a[:, i]
        b = scores_b[:, i]
        if a.std() < 1e-12 or b.std() < 1e-12:
            out[i] = 0.0
        else:
            out[i] = float(np.corrcoef(a, b)[0, 1])
    return out


def fit_score_cca(
    EA_tr: NDArray[np.float64],
    EB_tr: NDArray[np.float64],
    EA_te: NDArray[np.float64],
    EB_te: NDArray[np.float64],
    n_pca: int,
    k: int,
    c: float,
    seed: int = 0,
) -> NDArray[np.float64]:
    """Fit regularized CCA on TRAIN embeddings, score on TEST.

    PCA-reduces each view (fit on TRAIN), fits ``cca_zoo.linear.rCCA`` with
    regularization ``c`` on both views, transforms TEST, and returns the
    per-dimension held-out canonical correlations (length ``min(k, ...)``).
    The absolute value is taken because canonical-variate sign is arbitrary.
    """
    from cca_zoo.linear import rCCA  # lazy import

    ZA_tr, ZA_te = _pca_reduce(EA_tr, EA_te, n_pca, seed)
    ZB_tr, ZB_te = _pca_reduce(EB_tr, EB_te, n_pca, seed)
    k_eff = int(min(k, ZA_tr.shape[1], ZB_tr.shape[1]))

    model = rCCA(latent_dimensions=k_eff, c=[c, c])
    model.fit((ZA_tr, ZB_tr))
    sa, sb = model.transform((ZA_te, ZB_te))
    # Signed held-out canonical correlations: a joint sign-flip of both variates
    # preserves r, so a negative held-out r means the fitted direction did not
    # generalize (reversed) -- evidence against association, not sign ambiguity.
    # Do NOT take the absolute value (that folds the null and inflates the ceiling).
    return _heldout_corr(np.asarray(sa), np.asarray(sb))


# --- Hyperparameter tuning (inner CV on TRAIN only) -------------------------


@dataclass
class Grid:
    """Hyperparameter grid for the inner CV."""

    n_pca: tuple[int, ...] = (3, 5, 10)
    k: tuple[int, ...] = (5,)
    c: tuple[float, ...] = (0.5, 0.9, 0.99)


def tune_hyperparams(
    EA_tr: NDArray[np.float64],
    EB_tr: NDArray[np.float64],
    grid: Grid,
    strata: NDArray[np.str_],
    n_inner: int = 3,
    seed: int = 0,
) -> dict[str, Any]:
    """Pick ``(n_pca, k, c)`` by inner stratified-KFold CV on the embeddings.

    Scored by the mean *leading* held-out canonical correlation across inner
    folds, with folds stratified by ``strata`` (the training rows' strata).
    Operates only on already-computed TRAIN embeddings, so the expensive
    embedding step is not repeated during tuning.
    """
    kf = StratifiedKFold(n_splits=n_inner, shuffle=True, random_state=seed)
    best: dict[str, Any] | None = None
    best_score = -np.inf
    for n_pca in grid.n_pca:
        for k in grid.k:
            for c in grid.c:
                scores: list[float] = []
                for tr_idx, va_idx in kf.split(EA_tr, strata):
                    r = fit_score_cca(
                        EA_tr[tr_idx], EB_tr[tr_idx],
                        EA_tr[va_idx], EB_tr[va_idx],
                        n_pca=n_pca, k=k, c=c, seed=seed,
                    )
                    scores.append(float(r[0]) if r.size else 0.0)
                mean_score = float(np.mean(scores))
                if mean_score > best_score:
                    best_score = mean_score
                    best = {"n_pca": n_pca, "k": k, "c": c}
    assert best is not None
    best["inner_score"] = best_score
    return best


# --- Outer repeated-split evaluation ----------------------------------------


@dataclass
class EvalResult:
    """Held-out results across repeated outer splits."""

    r_per_split: NDArray[np.float64]  # (n_splits, k) held-out canonical r
    hyperparams: list[dict[str, Any]]
    mode: str
    grid: Grid

    @property
    def mean_r(self) -> NDArray[np.float64]:
        return np.nanmean(self.r_per_split, axis=0)

    @property
    def shared_variance(self) -> NDArray[np.float64]:
        """Per-dimension held-out shared variance (mean r squared)."""
        return self.mean_r ** 2

    def ci(self, dim: int = 0, alpha: float = 0.05) -> tuple[float, float]:
        """Percentile range across splits for a single canonical dimension.

        NOTE: this is *not* a valid confidence interval. Repeated train/test
        splits share data, so their fold statistics are dependent and this
        percentile range understates uncertainty (and narrows as splits are
        added) - see Zeng et al. (2026), bioRxiv 2026.05.17.724301. For valid
        CIs use the SHARP estimator in :mod:`cogmood_analysis.sharp`
        (:func:`sharp.sharp_ci`). This is retained only as a descriptive
        split-to-split spread.
        """
        col = self.r_per_split[:, dim]
        col = col[~np.isnan(col)]
        lo = float(np.percentile(col, 100 * alpha / 2))
        hi = float(np.percentile(col, 100 * (1 - alpha / 2)))
        return lo, hi


def _embed_pair(
    A_tr, A_te, B_tr, B_te, mode: str, fm_kwargs: dict[str, Any]
) -> tuple[NDArray, NDArray, NDArray, NDArray]:
    """Embed both views independently for one split."""
    if mode == "fm":
        EA_tr, EA_te = embed_view_fm(A_tr, A_te, **fm_kwargs)
        EB_tr, EB_te = embed_view_fm(B_tr, B_te, **fm_kwargs)
    elif mode == "raw":
        EA_tr, EA_te = embed_view_raw(A_tr, A_te)
        EB_tr, EB_te = embed_view_raw(B_tr, B_te)
    else:
        raise ValueError(f"unknown mode {mode!r}")
    return EA_tr, EA_te, EB_tr, EB_te


def repeated_eval(
    views: Views,
    mode: str = "fm",
    n_splits: int = 20,
    test_size: float = 0.2,
    grid: Grid | None = None,
    n_inner: int = 3,
    fm_kwargs: dict[str, Any] | None = None,
    seed: int = 0,
    verbose: bool = True,
) -> EvalResult:
    """Estimate held-out shared variance over repeated train/test splits.

    Per split: embed each view (fit on train, transform test), tune
    ``(n_pca, k, c)`` by inner CV on the train embeddings, refit on full
    train, and score on the held-out test. The embedding step dominates
    runtime, so it is done once per split.

    ``mode="fm"`` uses TabPFN embeddings; ``mode="raw"`` is the standardized
    raw-feature control arm.
    """
    grid = grid or Grid()
    fm_kwargs = fm_kwargs or {}
    splitter = StratifiedShuffleSplit(
        n_splits=n_splits, test_size=test_size, random_state=seed
    )

    k_max = max(grid.k)
    r_rows: list[NDArray[np.float64]] = []
    hps: list[dict[str, Any]] = []
    for s, (tr, te) in enumerate(splitter.split(views.A, views.strata)):
        EA_tr, EA_te, EB_tr, EB_te = _embed_pair(
            views.A[tr], views.A[te], views.B[tr], views.B[te], mode, fm_kwargs
        )
        hp = tune_hyperparams(
            EA_tr, EB_tr, grid, views.strata[tr], n_inner=n_inner, seed=seed + s
        )
        r = fit_score_cca(
            EA_tr, EB_tr, EA_te, EB_te,
            n_pca=hp["n_pca"], k=hp["k"], c=hp["c"], seed=seed + s,
        )
        row = np.full(k_max, np.nan)
        row[: r.size] = r
        r_rows.append(row)
        hps.append(hp)
        if verbose:
            print(
                f"[{mode}] split {s + 1}/{n_splits} "
                f"hp={hp} leading r={r[0]:.3f}"
            )
    return EvalResult(
        r_per_split=np.vstack(r_rows), hyperparams=hps, mode=mode, grid=grid
    )


# --- Permutation null -------------------------------------------------------


def permutation_null(
    views: Views,
    mode: str = "fm",
    n_perm: int = 200,
    test_size: float = 0.2,
    grid: Grid | None = None,
    n_inner: int = 3,
    fm_kwargs: dict[str, Any] | None = None,
    seed: int = 0,
    verbose: bool = True,
) -> NDArray[np.float64]:
    """Null distribution of the leading held-out canonical correlation.

    View B's subject order is permuted relative to View A *before* the split,
    breaking the cross-view pairing while preserving each view's marginal
    structure. The identical embed -> tune -> fit -> score pipeline is then
    run and the leading held-out canonical correlation recorded. Compare the
    observed leading r against this null to get a p-value.
    """
    grid = grid or Grid()
    fm_kwargs = fm_kwargs or {}
    rng = np.random.default_rng(seed)
    null = np.empty(n_perm, dtype=np.float64)
    splitter = StratifiedShuffleSplit(
        n_splits=1, test_size=test_size, random_state=seed
    )
    for p in range(n_perm):
        perm = rng.permutation(views.A.shape[0])
        B_perm = views.B[perm]
        tr, te = next(splitter.split(views.A, views.strata))
        EA_tr, EA_te, EB_tr, EB_te = _embed_pair(
            views.A[tr], views.A[te], B_perm[tr], B_perm[te], mode, fm_kwargs
        )
        hp = tune_hyperparams(
            EA_tr, EB_tr, grid, views.strata[tr], n_inner=n_inner, seed=seed + p
        )
        r = fit_score_cca(
            EA_tr, EB_tr, EA_te, EB_te,
            n_pca=hp["n_pca"], k=hp["k"], c=hp["c"], seed=seed + p,
        )
        null[p] = float(r[0]) if r.size else 0.0
        if verbose and (p + 1) % 10 == 0:
            print(f"[{mode}] permutation {p + 1}/{n_perm}")
    return null


def permutation_pvalue(observed: float, null: NDArray[np.float64]) -> float:
    """One-sided permutation p-value with the standard +1 correction."""
    null = np.asarray(null)
    return float((np.sum(null >= observed) + 1) / (null.size + 1))
