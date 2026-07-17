"""Anna Karenina (AK) test: does idiosyncratic cognitive abnormality track symptoms?

The CCA ladder found only a small, essentially linear *shared* cognition<->symptom
axis. The AK hypothesis is that symptomatic individuals are each abnormal in their
own way, so there is no shared axis - instead the *magnitude/extremeness* of a
person's deviation from healthy cognitive norms should track symptoms.

This module consumes the normative deviation matrix
(:mod:`cogmood_analysis.normative`, saved to
``data/exploratory/normative_deviations.pkl``) and tests, on the **training half
only**, whether deviation-derived features predict symptom scores.

Three approaches (deviations encoded as *extremeness*, ``|z|``, since signed
deviations carry a random sign under AK and don't predict - verified by
simulation):

* ``max_abs`` / ``topk_mean`` - unsupervised max / top-k mean of ``|z|``; robust
  to sparsity (only a subset of parameters relevant) -> primary AK statistic.
* ``enet`` - supervised elastic-net on the ``|z|`` matrix; learns the relevant
  subset; evaluated by out-of-fold r^2 to avoid circular selection.
* ``dist`` / ``count`` - global RMS deviation / count of ``|z|>2``; dilution-prone
  baselines the others should beat.

Symptom targets: the summary totals, the subscales, and a general PC1 factor; each
in raw and age/sex-residualized variants (deviations are already age/sex-adjusted).
PC1 (residualized) is the pre-specified primary; the rest are FDR-corrected
secondary. Significance is by permuting the symptom vector.

Held-out half is never touched here; a confirmatory test on it is a separate step.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence
import os
import pickle

import numpy as np
import polars as pl
from numpy.typing import NDArray
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.linear_model import ElasticNet, ElasticNetCV
from sklearn.model_selection import KFold

from . import shared_variance as sv

SUMMARY_TOTALS = ("phq8", "gad7", "baars", "hitop")
INDICATOR_Z = 2.0


# --- loading + alignment ----------------------------------------------------


@dataclass
class AKData:
    sub_ids: NDArray[np.str_]
    params: list[str]
    absz: NDArray[np.float64]      # |z|, NaN where QC failed  (n, p)
    symptom_cols: list[str]
    symptoms: NDArray[np.float64]  # (n, n_symptom_cols)
    age: NDArray[np.float64]
    sex: NDArray[np.float64]
    is_hv: NDArray[np.bool_]


def load_ak_data(
    deviations_pkl: str | os.PathLike,
    training_csv: str | os.PathLike,
    symptom_cols: Sequence[str] | None = None,
) -> AKData:
    """Load the deviation matrix and symptom scores, aligned by ``sub_id``."""
    dev = pickle.loads(Path(deviations_pkl).read_bytes())
    df = pl.read_csv(training_csv, infer_schema_length=20000)
    scols = [c for c in (symptom_cols or sv.VIEW_B_COLUMNS) if c in df.columns]

    # align training_data rows to the deviation matrix order
    order = {s: i for i, s in enumerate(df["sub_id"].to_list())}
    idx = np.array([order[s] for s in dev["sub_ids"]])
    sub = df[idx]

    symptoms = sub.select(scols).to_numpy().astype(np.float64)
    age = sub["age"].to_numpy().astype(np.float64)
    sex = sub["sex"].cast(pl.Float64).to_numpy()
    absz = np.abs(dev["z"])
    return AKData(sub_ids=dev["sub_ids"], params=list(dev["params"]), absz=absz,
                  symptom_cols=scols, symptoms=symptoms, age=age, sex=sex,
                  is_hv=dev["is_hv"])


# --- symptom targets (raw + residualized, plus PC1) -------------------------


def _residualize(y: NDArray, age: NDArray, sex: NDArray) -> NDArray[np.float64]:
    """Regress y on [1, age, age^2, sex], return residuals (age/sex adjusted)."""
    a = (age - age.mean()) / (age.std() or 1.0)
    X = np.column_stack([np.ones_like(a), a, a**2, sex])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return y - X @ beta


@dataclass
class Target:
    name: str
    variant: str  # "raw" or "resid"
    y: NDArray[np.float64]
    primary: bool = False


def symptom_targets(data: AKData) -> list[Target]:
    """Summary totals + subscales + PC1 general factor, each raw and residualized.

    PC1 is the first principal component of the standardized symptom columns
    (oriented so higher = more symptoms). Residualized PC1 is the primary.
    """
    targets: list[Target] = []
    S = data.symptoms
    Sz = (S - S.mean(0)) / S.std(0)
    pc = PCA(n_components=1, random_state=0).fit(Sz)
    pc1 = pc.transform(Sz).ravel()
    if np.corrcoef(pc1, Sz.mean(1))[0, 1] < 0:  # orient: higher = more symptoms
        pc1 = -pc1

    named = list(zip(data.symptom_cols, S.T)) + [("PC1", pc1)]
    for name, y in named:
        y = np.asarray(y, float)
        is_pc1 = name == "PC1"
        targets.append(Target(name, "raw", y, primary=False))
        targets.append(Target(name, "resid", _residualize(y, data.age, data.sex),
                              primary=is_pc1))
    return targets


# --- deviation features -----------------------------------------------------


def deviation_features(absz: NDArray, topk: Sequence[int] = (3, 5)) -> dict[str, NDArray]:
    """Unsupervised per-subject features from ``|z|`` (NaN = QC-failed)."""
    feats: dict[str, NDArray] = {}
    with np.errstate(all="ignore"):
        feats["max_abs"] = np.nanmax(np.where(np.isnan(absz), -np.inf, absz), axis=1)
        feats["max_abs"][~np.isfinite(feats["max_abs"])] = np.nan
        for k in topk:
            srt = np.sort(np.where(np.isnan(absz), -np.inf, absz), axis=1)[:, ::-1]
            feats[f"top{k}_mean"] = srt[:, :k].mean(axis=1)
        feats["count_gt2"] = np.nansum(absz > INDICATOR_Z, axis=1).astype(float)
        feats["dist"] = np.sqrt(np.nanmean(absz**2, axis=1))  # RMS over available
    return feats


def _impute_absz(absz: NDArray) -> NDArray[np.float64]:
    """Column-median impute |z| for the supervised matrix."""
    X = absz.copy()
    for j in range(X.shape[1]):
        col = X[:, j]
        med = np.nanmedian(col)
        col[np.isnan(col)] = med if np.isfinite(med) else 0.0
    return X


# --- tests ------------------------------------------------------------------


def ak_univariate(feature: NDArray, y: NDArray, n_perm: int = 2000,
                  seed: int = 0) -> dict[str, float]:
    """Spearman corr of an unsupervised feature vs symptom + permutation null."""
    ok = np.isfinite(feature) & np.isfinite(y)
    f, yy = feature[ok], y[ok]
    r = stats.spearmanr(f, yy).statistic
    rng = np.random.default_rng(seed)
    fr = stats.rankdata(f)
    null = np.empty(n_perm)
    for i in range(n_perm):
        null[i] = stats.spearmanr(fr, rng.permutation(yy)).statistic
    # one-sided (AK: more extremeness -> more symptoms) with +1 correction
    p = (np.sum(null >= r) + 1) / (n_perm + 1)
    return {"effect": float(r), "p": float(p), "n": int(ok.sum())}


def ak_supervised(absz_imp: NDArray, y: NDArray, n_splits: int = 5,
                  n_repeats: int = 4, n_perm: int = 200, seed: int = 0) -> dict[str, Any]:
    """Elastic-net on |z| -> out-of-fold r^2 + permutation null.

    Hyperparameters (alpha, l1_ratio) are chosen once via CV on the full sample;
    OOF predictions use repeated K-fold refits at those hyperparameters. The
    permutation null redoes the identical OOF procedure on shuffled y, so its
    p-value is valid (the observed r^2 is mildly optimistic but the null matches).
    """
    ok = np.isfinite(y)
    X, yy = absz_imp[ok], y[ok]
    Xs = (X - X.mean(0)) / (X.std(0) + 1e-12)
    cv = ElasticNetCV(l1_ratio=[.3, .5, .7, .9, 1.0], cv=5, n_alphas=40,
                      max_iter=5000, random_state=seed).fit(Xs, yy)
    alpha, l1 = cv.alpha_, cv.l1_ratio_

    def oof_r2(target: NDArray, s: int) -> tuple[float, NDArray]:
        pred = np.zeros_like(target)
        for rep in range(n_repeats):
            kf = KFold(n_splits=n_splits, shuffle=True, random_state=s + rep)
            p = np.zeros_like(target)
            for tr, te in kf.split(Xs):
                m = ElasticNet(alpha=alpha, l1_ratio=l1, max_iter=5000).fit(Xs[tr], target[tr])
                p[te] = m.predict(Xs[te])
            pred += p
        pred /= n_repeats
        r = np.corrcoef(pred, target)[0, 1] if pred.std() > 0 else 0.0
        return float(np.sign(r) * r**2), pred

    obs_r2, _ = oof_r2(yy, seed)
    rng = np.random.default_rng(seed)
    null = np.array([oof_r2(rng.permutation(yy), seed + 1000 + i)[0] for i in range(n_perm)])
    p = (np.sum(null >= obs_r2) + 1) / (n_perm + 1)
    # nonzero-coefficient parameters at the chosen hyperparameters (full-sample fit)
    coef = ElasticNet(alpha=alpha, l1_ratio=l1, max_iter=5000).fit(Xs, yy).coef_
    return {"effect": obs_r2, "p": float(p), "n": int(ok.sum()),
            "n_selected": int((np.abs(coef) > 1e-8).sum()), "alpha": float(alpha),
            "l1_ratio": float(l1), "coef": coef}


# --- orchestration ----------------------------------------------------------


def _bh_fdr(pvals: NDArray) -> NDArray[np.float64]:
    """Benjamini-Hochberg FDR q-values."""
    p = np.asarray(pvals, float)
    n = len(p)
    order = np.argsort(p)
    q = np.empty(n)
    prev = 1.0
    for rank, i in enumerate(order[::-1]):
        r = n - rank
        prev = min(prev, p[i] * n / r)
        q[i] = prev
    return q


def run_ak(
    data: AKData,
    n_perm_uni: int = 2000,
    n_perm_sup: int = 200,
    seed: int = 0,
    verbose: bool = True,
) -> pl.DataFrame:
    """Run all approaches x targets x {raw,resid}; return a tidy results table.

    PC1/resid is flagged as the pre-specified primary; secondary rows get BH-FDR
    q-values (computed within the secondary set, per approach).
    """
    targets = symptom_targets(data)
    feats = deviation_features(data.absz)
    absz_imp = _impute_absz(data.absz)

    rows: list[dict[str, Any]] = []
    for t in targets:
        for fname, fvec in feats.items():
            res = ak_univariate(fvec, t.y, n_perm=n_perm_uni, seed=seed)
            rows.append({"approach": fname, "target": t.name, "variant": t.variant,
                         "primary": t.primary and fname == "max_abs",
                         "effect": res["effect"], "effect_kind": "spearman_r",
                         "p": res["p"], "n": res["n"], "n_selected": None})
        sup = ak_supervised(absz_imp, t.y, n_perm=n_perm_sup, seed=seed)
        rows.append({"approach": "enet", "target": t.name, "variant": t.variant,
                     "primary": t.primary, "effect": sup["effect"],
                     "effect_kind": "oof_r2", "p": sup["p"], "n": sup["n"],
                     "n_selected": sup["n_selected"]})
        if verbose:
            print(f"[ak] {t.name:16s} {t.variant:5s} | "
                  f"max|z| r={rows[-6]['effect']:+.3f} p={rows[-6]['p']:.3f} | "
                  f"enet r2={sup['effect']:+.3f} p={sup['p']:.3f} (sel {sup['n_selected']})",
                  flush=True)

    tbl = pl.DataFrame(rows)
    # BH-FDR within the secondary set (exclude the primary), per approach
    q = np.ones(tbl.height)
    sec = (~tbl["primary"]).to_numpy()
    for ap in tbl["approach"].unique().to_list():
        m = sec & (tbl["approach"] == ap).to_numpy()
        if m.sum():
            q[m] = _bh_fdr(tbl.filter(pl.Series(m))["p"].to_numpy())
    return tbl.with_columns(pl.Series("fdr_q", q))


# --- Westfall-Young max-statistic FWER correction ---------------------------


def maxstat_correction(
    absz: NDArray, targets: Sequence["Target"], topk: Sequence[int] = (3, 5),
    n_perm: int = 10000, seed: int = 0,
) -> tuple[pl.DataFrame, NDArray]:
    """Westfall-Young **step-down** maxT FWER correction across the unsupervised family.

    The family is every (unsupervised approach × target) pair. A **single shared
    permutation of the subject order** is drawn per iteration and applied to the
    whole symptom matrix at once (features fixed), so the joint dependence of the
    correlated tests is preserved.

    Step-down (Westfall & Young 1993): order the observed statistics
    ``t_(1) >= ... >= t_(m)``. For each permutation, form the successive maxima
    ``q_(k) = max`` of the permutation statistics over ranks ``k..m`` (i.e. the
    hypothesis at rank k competes only against itself and the less-significant
    ones). The raw adjusted p at rank k is the fraction of permutations with
    ``q_(k) >= t_(k)``; monotonicity is then enforced across ranks. This is
    uniformly more powerful than single-step maxT (which always uses the
    full-family max) while still controlling FWER under subset pivotality. The
    two coincide for the most significant test.

    The test is **one-sided** (positive): the features are non-negative measures of
    cognitive abnormality (max|z|, top-k mean|z|, count |z|>2, distance) and AK
    predicts more abnormality → more symptoms; a negative association is not a
    meaningful alternative here.

    Spearman is computed as Pearson on standardized ranks, so each permutation is
    a couple of small matrix products (fast; ``n_perm`` can be large). Subjects
    non-finite on any unsupervised feature (essentially none) are dropped so the
    shared permutation aligns exactly.

    Returns a per-(approach, target, variant) table with observed ``effect`` and
    ``adj_p_maxT``, plus the rank-1 full-family max-null vector (for reference).
    """
    feats = deviation_features(absz, topk=topk)
    fnames = list(feats)
    F = np.column_stack([feats[k] for k in fnames])          # (n, n_feat)
    Y = np.column_stack([t.y for t in targets])              # (n, n_targ)
    ok = np.isfinite(F).all(1) & np.isfinite(Y).all(1)
    F, Y = F[ok], Y[ok]
    n = F.shape[0]

    def std_ranks(M: NDArray) -> NDArray[np.float64]:
        R = np.apply_along_axis(stats.rankdata, 0, M).astype(float)
        return (R - R.mean(0)) / R.std(0)

    RF, RY = std_ranks(F), std_ranks(Y)                      # standardized ranks
    S_obs = (RF.T @ RY) / n                                  # (n_feat, n_targ) Spearman r
    n_feat, n_targ = S_obs.shape
    obs = S_obs.ravel()
    order = np.argsort(obs)[::-1]                            # observed, most->least significant
    obs_sorted = obs[order]

    rng = np.random.default_rng(seed)
    counts = np.zeros(obs.size)                              # exceedances per rank
    maxnull = np.empty(n_perm)                               # rank-1 (full-family) max, reference
    for b in range(n_perm):
        Sp = ((RF.T @ RY[rng.permutation(n)]) / n).ravel()[order]
        # successive maxima over ranks k..m (reverse cumulative max)
        q = np.maximum.accumulate(Sp[::-1])[::-1]
        counts += q >= obs_sorted
        maxnull[b] = q[0]
    p_raw = (counts + 1) / (n_perm + 1)
    p_adj_sorted = np.maximum.accumulate(p_raw)              # enforce monotone non-decreasing
    p_adj = np.empty_like(p_adj_sorted)
    p_adj[order] = p_adj_sorted
    P = p_adj.reshape(n_feat, n_targ)

    rows = []
    for i, fn in enumerate(fnames):
        for j, t in enumerate(targets):
            rows.append({"approach": fn, "target": t.name, "variant": t.variant,
                         "effect": float(S_obs[i, j]), "adj_p_maxT": float(P[i, j]),
                         "primary": bool(t.primary)})
    return pl.DataFrame(rows).sort("adj_p_maxT"), maxnull


def maxstat_within_approach(
    absz: NDArray, targets: Sequence["Target"], topk: Sequence[int] = (3, 5),
    n_perm: int = 10000, seed: int = 0,
) -> pl.DataFrame:
    """Step-down maxT FWER correction computed SEPARATELY within each approach.

    Same one-sided step-down Westfall-Young procedure as :func:`maxstat_correction`,
    but the family is a single approach's targets (not the joint approach×target
    grid). Reported as ``within_maxT``. This controls FWER *within* each approach
    (across its symptom targets) but NOT jointly across the 5 approaches - use it
    when the approaches are treated as separate analyses rather than one family.
    Permutations are shared across targets within each approach (same seed across
    approaches for comparability).
    """
    feats = deviation_features(absz, topk=topk)
    fnames = list(feats)
    F = np.column_stack([feats[k] for k in fnames])
    Y = np.column_stack([t.y for t in targets])
    ok = np.isfinite(F).all(1) & np.isfinite(Y).all(1)
    F, Y = F[ok], Y[ok]
    n = F.shape[0]

    def std_ranks(M: NDArray) -> NDArray[np.float64]:
        R = np.apply_along_axis(stats.rankdata, 0, M).astype(float)
        return (R - R.mean(0)) / R.std(0)

    RF, RY = std_ranks(F), std_ranks(Y)
    rows = []
    for i, fn in enumerate(fnames):
        rf = RF[:, i]
        obs = (rf @ RY) / n                       # (n_targ,) Spearman r for this approach
        order = np.argsort(obs)[::-1]
        obs_sorted = obs[order]
        rng = np.random.default_rng(seed)         # same perms across approaches
        counts = np.zeros(obs.size)
        for _ in range(n_perm):
            sp = ((rf @ RY[rng.permutation(n)]) / n)[order]
            q = np.maximum.accumulate(sp[::-1])[::-1]
            counts += q >= obs_sorted
        p_adj_sorted = np.maximum.accumulate((counts + 1) / (n_perm + 1))
        p_adj = np.empty_like(p_adj_sorted)
        p_adj[order] = p_adj_sorted
        for j, t in enumerate(targets):
            rows.append({"approach": fn, "target": t.name, "variant": t.variant,
                         "effect": float(obs[j]), "within_maxT": float(p_adj[j]),
                         "primary": bool(t.primary)})
    return pl.DataFrame(rows).sort(["approach", "within_maxT"])
