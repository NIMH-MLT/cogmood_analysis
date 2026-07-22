"""A ladder of CCA arms compared with the SHARP test.

This builds on :mod:`cogmood_analysis.shared_variance` to compare a *ladder* of
increasingly flexible cross-view association methods on the same held-out data:

* ``"raw"``    - regularized linear CCA on standardized raw features (baseline)
* ``"kernel"`` - kernel CCA (RBF) on raw features
* ``"deep"``   - Deep CCA: trained MLP encoders, then linear CCA on the learned
  representations (so canonical dimensions are ordered/comparable)
* ``"fm"``     - frozen tabular foundation-model (TabPFN) embeddings + linear CCA
* ``"fm_deep"`` - TabPFN **fine-tuned jointly** with a deep CCA head: both views'
  TabPFN transformers and MLP heads are trained end-to-end with the CCA
  correlation loss, then linear CCA on the fine-tuned representations

Inference uses the **SHARP (Split-HAlf RePeated) test** of Zeng et al. (2026,
bioRxiv 2026.05.17.724301), "Widespread use of invalid statistical tests in
biomedical machine learning". The key point of that paper: performance
estimates across cross-validation folds are *not independent* (folds share
training data), so naive CIs / paired tests across folds - and especially
*repeated* CV - inflate false positives (toward 100% as repetitions grow). A
percentile interval across repeated overlapping splits (what an earlier version
of this analysis reported) is exactly the invalid construct.

SHARP fixes this: in each of ``J`` repetitions, the subjects are split into two
**disjoint** halves A and B, and K-fold CV is run *within each half*. Averaging
the fold-level statistics within a half gives one statistic per half, so each
repetition yields a pair ``(D_Aj, D_Bj)`` that is **independent within the
repetition** (the halves share no subjects) while statistics across repetitions
remain correlated. That structure identifies both the per-statistic variance
``sigma^2`` and the across-repetition correlation ``rho``, giving

    D_bar      = (mean(D_A) + mean(D_B)) / 2
    Var(D_bar) = sigma^2 * (1/(2J) + (J-1)/J * rho)

from which valid CIs and a valid model-comparison test follow.

Inference is the paper's **score test** (its recommended choice, Supplementary
S7.6/S7.8): the ``2J`` half-statistics are jointly Gaussian with mean ``mu``,
variance ``sigma^2`` and common correlation ``rho`` on every off-diagonal pair
*except* the paired ``(D_Aj, D_Bj)`` (independent halves). To test ``H0: mu = mu0``
we estimate ``sigma^2`` and ``rho`` by maximizing that Gaussian likelihood with
the mean fixed at ``mu0`` (null-constrained MLE, ``_null_constrained_mle``), then
``Z = (D_bar - mu0) / sqrt(Var(D_bar; sigma0^2, rho0))``. Confidence intervals are
built by **test inversion** (the set of ``mu0`` with ``p >= alpha``,
``_invert_score_test``), matching the paper. The earlier method-of-moments + Wald
path (``_sharp_moments``) is retained only as a reference for the calibration test
- it inflates the false-positive rate and must not be used for reported inference.

Because comparisons must be paired, :func:`sharp_eval` runs *all* arms on the
*same* folds, so any pair of arms can be compared via the per-half difference of
their statistics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
from numpy.typing import NDArray
from scipy import optimize, stats
from sklearn.model_selection import StratifiedKFold

from . import shared_variance as sv

#: The ladder, from least to most flexible.
LADDER: tuple[str, ...] = (
    "raw", "kernel", "deep", "fm", "fm_deep", "tabfm", "tabfm_deep",
)


@dataclass
class ArmConfig:
    """Fixed hyperparameters for one arm.

    SHARP needs many model fits, so (per the preprint's own practice of fixing
    one algorithm per dataset) we use fixed, principled hyperparameters rather
    than nested per-fold tuning - this also keeps the held-out statistic a clean
    measure of that arm's generalization.
    """

    k: int = 5            # canonical dimensions
    c: float = 0.9        # rCCA/KCCA regularization (ridge toward PLS)
    n_pca: int = 10       # PCA components before linear CCA (raw/fm arms)
    # kernel arm
    kernel: str = "rbf"
    gamma: float | None = None
    # deep arm
    layers: tuple[int, ...] = (64, 32)
    max_epochs: int = 100
    lr: float = 1e-3
    batch_size: int = 128
    # fm arm
    pseudo_target: str = "pca1"
    device: str = "auto"
    # fm_deep arm (joint TabPFN + deep-CCA fine-tuning)
    ft_epochs: int = 60
    ft_lr: float = 3e-5
    ft_weight_decay: float = 1e-4
    # (1) disjoint support/query split per step + (2) validation early stopping
    ft_query_frac: float = 0.3      # fraction of the fine-tune pool used as query
    ft_val_frac: float = 0.2        # held-out fraction for early stopping
    ft_early_stopping: bool = True
    ft_eval_every: int = 3          # epochs between validation checks
    ft_patience: int = 4            # validation checks w/o improvement before stop
    # (3) parameter-efficient fine-tuning (off by default)
    ft_trainable_blocks: int | None = None  # train only last N transformer blocks
    ft_fm_lr_scale: float = 1.0     # LR multiplier for the FM vs the head
    # (4) stay-near-prior penalties (off by default)
    ft_anchor_l2: float = 0.0       # L2 of trainable params toward pretrained values
    ft_cca_eps: float = 1e-4        # CCALoss ridge on the covariances
    ft_head_dropout: float = 0.0


DEFAULT_CONFIGS: dict[str, ArmConfig] = {
    "raw": ArmConfig(n_pca=10),
    "kernel": ArmConfig(),
    "deep": ArmConfig(),
    "fm": ArmConfig(n_pca=5),
    "fm_deep": ArmConfig(n_pca=5),
    "tabfm": ArmConfig(n_pca=5),
    "tabfm_deep": ArmConfig(n_pca=5),
}


# --- per-arm scorers --------------------------------------------------------


def _deep_representations(
    A_tr: NDArray, B_tr: NDArray, A_te: NDArray, B_te: NDArray, cfg: ArmConfig, seed: int
) -> tuple[NDArray, NDArray, NDArray, NDArray]:
    """Train Deep CCA encoders on TRAIN, return train/test representations.

    Canonical ordering is *not* guaranteed by the encoders, so the caller fits a
    linear CCA on the train representations to obtain ordered canonical
    correlations (cf. Andrew et al. 2013).
    """
    import logging
    import warnings

    import torch
    import torch.nn as nn
    import lightning.pytorch as pl
    from torch.utils.data import DataLoader, Dataset
    from cca_zoo.deep import DCCA

    # Quiet Lightning's per-fit banners (TPU/seed/tip lines) and cosmetic
    # warnings (no val loop, dataloader worker count).
    for name in (
        "lightning.pytorch",
        "lightning.pytorch.utilities.rank_zero",
        "lightning.fabric.utilities.seed",
    ):
        logging.getLogger(name).setLevel(logging.ERROR)
    warnings.filterwarnings("ignore", message=".*does not have many workers.*")
    warnings.filterwarnings("ignore", message=".*validation_step.*")
    pl.seed_everything(seed, workers=True, verbose=False)

    def encoder(p: int) -> nn.Module:
        sizes = [p, *cfg.layers, cfg.k]
        mods: list[nn.Module] = []
        for i in range(len(sizes) - 1):
            mods.append(nn.Linear(sizes[i], sizes[i + 1]))
            if i < len(sizes) - 2:
                mods.append(nn.ReLU())
        return nn.Sequential(*mods)

    class _DS(Dataset):
        def __init__(self, *views: NDArray) -> None:
            self.views = [torch.tensor(v, dtype=torch.float32) for v in views]

        def __len__(self) -> int:
            return len(self.views[0])

        def __getitem__(self, i: int) -> dict[str, list[torch.Tensor]]:
            return {"views": [v[i] for v in self.views]}

    device = cfg.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    accelerator = "gpu" if device == "cuda" else "cpu"

    model = DCCA(
        latent_dimensions=cfg.k,
        encoders=[encoder(A_tr.shape[1]), encoder(B_tr.shape[1])],
        lr=cfg.lr,
        max_epochs=cfg.max_epochs,
    )
    loader = DataLoader(
        _DS(A_tr, B_tr), batch_size=cfg.batch_size, shuffle=True, drop_last=False
    )
    trainer = pl.Trainer(
        max_epochs=cfg.max_epochs,
        accelerator=accelerator,
        devices=1,
        enable_progress_bar=False,
        enable_model_summary=False,
        enable_checkpointing=False,
        logger=False,
        limit_val_batches=0,
    )
    trainer.fit(model, loader)

    def rep(*views: NDArray) -> list[NDArray]:
        out = model.transform(DataLoader(_DS(*views), batch_size=512, shuffle=False))
        return [np.asarray(o, dtype=np.float64) for o in out]

    ZA_tr, ZB_tr = rep(A_tr, B_tr)
    ZA_te, ZB_te = rep(A_te, B_te)
    return ZA_tr, ZB_tr, ZA_te, ZB_te


def _finetune_fm_dcca(
    A_tr: NDArray, B_tr: NDArray, A_te: NDArray, B_te: NDArray, cfg: ArmConfig, seed: int
) -> tuple[NDArray, NDArray, NDArray, NDArray]:
    """Jointly fine-tune two TabPFN encoders + MLP heads with the CCA loss.

    Each view's TabPFN transformer is made trainable and produces *differentiable*
    in-context embeddings; an MLP head projects to the latent space. Both views'
    transformers and heads are optimized end-to-end with
    ``cca_zoo.deep.objectives.CCALoss`` (negative sum of squared canonical
    correlations). Fine-tuned weights are then loaded into inference regressors to
    embed train/test, and the heads project those embeddings; the caller runs a
    final linear CCA.

    Overfitting controls:

    * **(1) disjoint support/query split** - each step draws a disjoint
      support/query partition of the fine-tune pool (the *same* subject indices
      for both views, so the query embeddings stay paired); the CCA loss is
      computed on the *query* rows, which are not part of the in-context support.
      This rewards a transferable mapping rather than memorizing the context.
    * **(2) validation early stopping** - a held-out slice of TRAIN is never used
      for support/query; the leading held-out canonical correlation on it is
      tracked, and the best weights are restored.

    Off-by-default knobs: parameter-efficient fine-tuning (``ft_trainable_blocks``,
    ``ft_fm_lr_scale``) and stay-near-prior penalties (``ft_anchor_l2``,
    ``ft_cca_eps``, ``ft_head_dropout``, ``ft_weight_decay``).

    Everything is fit on TRAIN only; test rows are embedded against the train
    context with no label leakage. This is the heaviest arm.
    """
    import copy

    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    from tabpfn import TabPFNRegressor
    from tabpfn.utils import meta_dataset_collator
    from cca_zoo.deep.objectives import CCALoss

    torch.manual_seed(seed)
    device = cfg.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dev = torch.device(device)
    rng = np.random.default_rng(seed)

    def standardize(tr, te):
        s = StandardScaler().fit(tr)
        return s.transform(tr).astype(np.float32), s.transform(te).astype(np.float32)

    AtrS, AteS = standardize(A_tr, A_te)
    BtrS, BteS = standardize(B_tr, B_te)
    yA = PCA(1, random_state=seed).fit_transform(AtrS).ravel().astype(np.float32)
    yB = PCA(1, random_state=seed).fit_transform(BtrS).ravel().astype(np.float32)

    def make_regressor():
        return TabPFNRegressor(
            fit_mode="batched", n_estimators=1, differentiable_input=False,
            device=device, random_state=seed, ignore_pretraining_limits=True,
        )

    regA, regB = make_regressor(), make_regressor()

    def embed(reg, Xstd, y, sup_idx, qry_idx, grad):
        """Differentiable (or not) query embeddings: support is the in-context set."""
        ns = len(sup_idx)
        Xcat = np.concatenate([Xstd[sup_idx], Xstd[qry_idx]], axis=0)
        ycat = np.concatenate(
            [y[sup_idx], np.zeros(len(qry_idx), dtype=np.float32)]
        )  # query labels are placeholders and never used as context

        def split_fn(Xf, yf):  # positional slice -> preserves order, no leakage
            return Xf[:ns], Xf[ns:], yf[:ns], yf[ns:]

        ds = reg.get_preprocessed_datasets(Xcat, ycat, split_fn, max_data_size=10**9)
        batch = next(iter(DataLoader(ds, batch_size=1, collate_fn=meta_dataset_collator)))
        Xtr, Xte, ytr, _, cat_ix, confs = batch[:6]
        reg.fit_from_preprocessed(Xtr, ytr, cat_ix, confs)  # no_refit=True keeps weights
        reg.model_.to(dev)
        ex = reg.executor_
        ctx = ex.X_trains[0].to(dev)
        cty = ex.y_trains[0].to(dev)
        q = (Xte[0] if isinstance(Xte, list) else Xte).to(dev)
        cats = [c[0] for c in cat_ix]
        full = torch.cat([ctx, q], dim=-2)
        with (torch.enable_grad() if grad else torch.no_grad()):
            out = reg.model_(
                full.transpose(0, 1), cty.transpose(0, 1),
                only_return_standard_out=False, categorical_inds=cats,
            )
        return out["test_embeddings"].squeeze(1).float()

    def head() -> nn.Module:
        sizes = [192, *cfg.layers, cfg.k]
        mods: list[nn.Module] = []
        for i in range(len(sizes) - 1):
            mods.append(nn.Linear(sizes[i], sizes[i + 1]))
            if i < len(sizes) - 2:
                mods.append(nn.ReLU())
                if cfg.ft_head_dropout > 0:
                    mods.append(nn.Dropout(cfg.ft_head_dropout))
        return nn.Sequential(*mods).to(dev)

    # train/val split of TRAIN: val is held out from all support/query draws (2)
    n = AtrS.shape[0]
    perm = rng.permutation(n)
    n_val = max(int(round(cfg.ft_val_frac * n)), cfg.k + 2) if cfg.ft_early_stopping else 0
    val_idx = perm[:n_val]
    pool_idx = perm[n_val:]

    # instantiate model_ (one embed call) before setting trainable params / optimizer
    q0 = pool_idx[: max(int(round(cfg.ft_query_frac * len(pool_idx))), cfg.k + 2)]
    s0 = pool_idx[len(q0):]
    embed(regA, AtrS, yA, s0, q0, grad=False)
    embed(regB, BtrS, yB, s0, q0, grad=False)

    def set_trainable(model):
        if cfg.ft_trainable_blocks is None:
            for p in model.parameters():
                p.requires_grad_(True)
            return
        for p in model.parameters():
            p.requires_grad_(False)
        layers = model.transformer_encoder.layers
        for i in range(max(0, len(layers) - cfg.ft_trainable_blocks), len(layers)):
            for p in layers[i].parameters():
                p.requires_grad_(True)

    set_trainable(regA.model_)
    set_trainable(regB.model_)
    hA, hB = head(), head()

    # anchor-to-prior snapshot of trainable FM params (4)
    fm_params = [p for m in (regA.model_, regB.model_) for p in m.parameters() if p.requires_grad]
    anchors = [p.detach().clone() for p in fm_params] if cfg.ft_anchor_l2 > 0 else None
    head_params = list(hA.parameters()) + list(hB.parameters())
    opt = torch.optim.Adam(
        [
            {"params": fm_params, "lr": cfg.ft_lr * cfg.ft_fm_lr_scale},
            {"params": head_params, "lr": cfg.ft_lr},
        ],
        weight_decay=cfg.ft_weight_decay,
    )
    loss_fn = CCALoss(eps=cfg.ft_cca_eps)

    def snapshot():
        return (
            copy.deepcopy(regA.model_.state_dict()),
            copy.deepcopy(regB.model_.state_dict()),
            copy.deepcopy(hA.state_dict()),
            copy.deepcopy(hB.state_dict()),
        )

    def validate():
        regA.model_.eval(); regB.model_.eval(); hA.eval(); hB.eval()
        with torch.no_grad():
            EAp = embed(regA, AtrS, yA, pool_idx, pool_idx, grad=False)
            EAv = embed(regA, AtrS, yA, pool_idx, val_idx, grad=False)
            EBp = embed(regB, BtrS, yB, pool_idx, pool_idx, grad=False)
            EBv = embed(regB, BtrS, yB, pool_idx, val_idx, grad=False)
            ZAp, ZAv = hA(EAp).cpu().numpy(), hA(EAv).cpu().numpy()
            ZBp, ZBv = hB(EBp).cpu().numpy(), hB(EBv).cpu().numpy()
        r = sv.fit_score_cca(ZAp, ZBp, ZAv, ZBv, n_pca=cfg.k, k=cfg.k, c=cfg.c, seed=seed)
        return float(r[0]) if r.size else 0.0

    best_val, best_state, bad = -np.inf, None, 0
    for epoch in range(cfg.ft_epochs):
        # (1) disjoint support/query split of the pool, aligned across views
        pe = rng.permutation(len(pool_idx))
        n_q = max(int(round(cfg.ft_query_frac * len(pool_idx))), cfg.k + 2)
        qry = pool_idx[pe[:n_q]]
        sup = pool_idx[pe[n_q:]]
        regA.model_.train(); regB.model_.train(); hA.train(); hB.train()
        opt.zero_grad()
        zA = hA(embed(regA, AtrS, yA, sup, qry, grad=True))
        zB = hB(embed(regB, BtrS, yB, sup, qry, grad=True))
        loss = loss_fn([zA, zB])
        if anchors is not None:
            loss = loss + cfg.ft_anchor_l2 * sum(
                ((p - a) ** 2).sum() for p, a in zip(fm_params, anchors)
            )
        loss.backward()
        opt.step()

        if cfg.ft_early_stopping and (
            epoch % cfg.ft_eval_every == 0 or epoch == cfg.ft_epochs - 1
        ):
            v = validate()
            if v > best_val + 1e-5:
                best_val, best_state, bad = v, snapshot(), 0
            else:
                bad += 1
                if bad >= cfg.ft_patience:
                    break

    if best_state is not None:
        regA.model_.load_state_dict(best_state[0])
        regB.model_.load_state_dict(best_state[1])
        hA.load_state_dict(best_state[2])
        hB.load_state_dict(best_state[3])

    # Evaluate: load fine-tuned weights into inference regressors, embed, project.
    def eval_embed(reg_ft, Xtr, Xte, y):
        re = TabPFNRegressor(
            device=device, n_estimators=1, ignore_pretraining_limits=True,
            random_state=seed,
        )
        re.fit(Xtr, y)
        re.model_.load_state_dict(reg_ft.model_.state_dict())
        Etr = np.asarray(re.get_embeddings(Xtr, "test")).squeeze(0)
        Ete = np.asarray(re.get_embeddings(Xte, "test")).squeeze(0)
        return Etr, Ete

    hA.eval(); hB.eval()
    with torch.no_grad():
        EAtr, EAte = eval_embed(regA, AtrS, AteS, yA)
        EBtr, EBte = eval_embed(regB, BtrS, BteS, yB)

        def proj(h, E):
            return h(torch.tensor(E, dtype=torch.float32, device=dev)).cpu().numpy()

        ZA_tr, ZA_te = proj(hA, EAtr), proj(hA, EAte)
        ZB_tr, ZB_te = proj(hB, EBtr), proj(hB, EBte)

    del regA, regB, hA, hB, opt, fm_params, head_params
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return ZA_tr, ZB_tr, ZA_te, ZB_te


def _finetune_tabfm_dcca(
    A_tr: NDArray, B_tr: NDArray, A_te: NDArray, B_te: NDArray, cfg: ArmConfig, seed: int
) -> tuple[NDArray, NDArray, NDArray, NDArray]:
    """Jointly fine-tune Google TabFM + MLP heads with the CCA loss.

    TabFM is ~1.6B params, so a full fine-tune of two encoders won't fit a 32 GB
    GPU. This is a **parameter-efficient** fine-tune: the frozen TabFM bulk (a
    single shared instance) runs under ``no_grad`` up to ``row_interactor_2`` via
    :func:`shared_variance.tabfm_prefix`, and only the **last block**
    (``row_interactor_2``, deep-copied per view) plus an MLP head per view are
    trained end-to-end with ``CCALoss``. Same overfitting controls as
    ``_finetune_fm_dcca`` (disjoint support/query split + validation early
    stopping). Everything fit on TRAIN; test embedded against the train context.
    """
    import copy

    import torch
    import torch.nn as nn
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    from cca_zoo.deep.objectives import CCALoss

    torch.manual_seed(seed)
    device = cfg.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dev = torch.device(device)
    rng = np.random.default_rng(seed)

    def standardize(tr, te):
        s = StandardScaler().fit(tr)
        return s.transform(tr).astype(np.float32), s.transform(te).astype(np.float32)

    AtrS, AteS = standardize(A_tr, A_te)
    BtrS, BteS = standardize(B_tr, B_te)
    yA = PCA(1, random_state=seed).fit_transform(AtrS).ravel().astype(np.float32)
    yB = PCA(1, random_state=seed).fit_transform(BtrS).ravel().astype(np.float32)

    frozen = sv._load_tabfm(device)  # shared frozen bulk
    for p in frozen.parameters():
        p.requires_grad_(False)
    frozen.eval()
    tailA = copy.deepcopy(frozen.row_interactor_2).to(dev)
    tailB = copy.deepcopy(frozen.row_interactor_2).to(dev)

    def head() -> nn.Module:
        sizes = [sv.TABFM_EMBED_DIM, *cfg.layers, cfg.k]
        mods: list[nn.Module] = []
        for i in range(len(sizes) - 1):
            mods.append(nn.Linear(sizes[i], sizes[i + 1]))
            if i < len(sizes) - 2:
                mods.append(nn.ReLU())
                if cfg.ft_head_dropout > 0:
                    mods.append(nn.Dropout(cfg.ft_head_dropout))
        return nn.Sequential(*mods).to(dev)

    hA, hB = head(), head()

    def embed_q(ctx_X, ctx_y, qry_X, tail, grad):
        n = len(ctx_X)
        Xcat = np.concatenate([ctx_X, qry_X], axis=0).astype(np.float32)
        yc = np.concatenate([ctx_y, np.full(len(qry_X), -100.0, np.float32)]).astype(np.float32)
        x_t = torch.tensor(Xcat, device=dev).unsqueeze(0)
        y_t = torch.tensor(yc, device=dev).unsqueeze(0)
        ts = torch.tensor([n], device=dev)
        with torch.no_grad():  # frozen bulk
            pre = sv.tabfm_prefix(frozen, x_t, y_t, ts)
        with (torch.enable_grad() if grad else torch.no_grad()):
            reps = tail(pre, d=None)
        return reps[0, n:].float()

    n = AtrS.shape[0]
    perm = rng.permutation(n)
    n_val = max(int(round(cfg.ft_val_frac * n)), cfg.k + 2) if cfg.ft_early_stopping else 0
    val_idx, pool_idx = perm[:n_val], perm[n_val:]

    params = (
        list(tailA.parameters()) + list(tailB.parameters())
        + list(hA.parameters()) + list(hB.parameters())
    )
    opt = torch.optim.Adam(params, lr=cfg.ft_lr, weight_decay=cfg.ft_weight_decay)
    loss_fn = CCALoss(eps=cfg.ft_cca_eps)
    modules = (tailA, tailB, hA, hB)

    def snapshot():
        return tuple(copy.deepcopy(m.state_dict()) for m in modules)

    def validate():
        for m in modules:
            m.eval()
        with torch.no_grad():
            ZAp = hA(embed_q(AtrS[pool_idx], yA[pool_idx], AtrS[pool_idx], tailA, False)).cpu().numpy()
            ZAv = hA(embed_q(AtrS[pool_idx], yA[pool_idx], AtrS[val_idx], tailA, False)).cpu().numpy()
            ZBp = hB(embed_q(BtrS[pool_idx], yB[pool_idx], BtrS[pool_idx], tailB, False)).cpu().numpy()
            ZBv = hB(embed_q(BtrS[pool_idx], yB[pool_idx], BtrS[val_idx], tailB, False)).cpu().numpy()
        r = sv.fit_score_cca(ZAp, ZBp, ZAv, ZBv, n_pca=cfg.k, k=cfg.k, c=cfg.c, seed=seed)
        return float(r[0]) if r.size else 0.0

    best_val, best_state, bad = -np.inf, None, 0
    for epoch in range(cfg.ft_epochs):
        pe = rng.permutation(len(pool_idx))
        n_q = max(int(round(cfg.ft_query_frac * len(pool_idx))), cfg.k + 2)
        qry, sup = pool_idx[pe[:n_q]], pool_idx[pe[n_q:]]
        for m in modules:
            m.train()
        opt.zero_grad()
        zA = hA(embed_q(AtrS[sup], yA[sup], AtrS[qry], tailA, True))
        zB = hB(embed_q(BtrS[sup], yB[sup], BtrS[qry], tailB, True))
        loss_fn([zA, zB]).backward()
        opt.step()
        if cfg.ft_early_stopping and (
            epoch % cfg.ft_eval_every == 0 or epoch == cfg.ft_epochs - 1
        ):
            v = validate()
            if v > best_val + 1e-5:
                best_val, best_state, bad = v, snapshot(), 0
            else:
                bad += 1
                if bad >= cfg.ft_patience:
                    break
    if best_state is not None:
        for m, st in zip(modules, best_state):
            m.load_state_dict(st)

    for m in modules:
        m.eval()
    with torch.no_grad():
        ZA_tr = hA(embed_q(AtrS, yA, AtrS, tailA, False)).cpu().numpy()
        ZB_tr = hB(embed_q(BtrS, yB, BtrS, tailB, False)).cpu().numpy()
        ZA_te = hA(embed_q(AtrS, yA, AteS, tailA, False)).cpu().numpy()
        ZB_te = hB(embed_q(BtrS, yB, BteS, tailB, False)).cpu().numpy()

    del tailA, tailB, hA, hB, opt, params
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return ZA_tr, ZB_tr, ZA_te, ZB_te


def score_arm(
    A_tr: NDArray,
    B_tr: NDArray,
    A_te: NDArray,
    B_te: NDArray,
    arm: str,
    cfg: ArmConfig,
    seed: int = 0,
) -> NDArray[np.float64]:
    """Held-out canonical correlations (per dimension, absolute) for one arm.

    Everything is fit on TRAIN and scored on TEST. All arms terminate in a
    linear CCA so the returned correlations are ordered and comparable across
    arms.
    """
    if arm == "raw":
        EA_tr, EA_te = sv._standardize(A_tr, A_te)
        EB_tr, EB_te = sv._standardize(B_tr, B_te)
        return sv.fit_score_cca(EA_tr, EB_tr, EA_te, EB_te, cfg.n_pca, cfg.k, cfg.c, seed)
    if arm == "fm":
        EA_tr, EA_te = sv.embed_view_fm(
            A_tr, A_te, pseudo_target=cfg.pseudo_target, device=cfg.device, seed=seed
        )
        EB_tr, EB_te = sv.embed_view_fm(
            B_tr, B_te, pseudo_target=cfg.pseudo_target, device=cfg.device, seed=seed
        )
        return sv.fit_score_cca(EA_tr, EB_tr, EA_te, EB_te, cfg.n_pca, cfg.k, cfg.c, seed)
    if arm == "kernel":
        from cca_zoo.nonparametric import KCCA

        EA_tr, EA_te = sv._standardize(A_tr, A_te)
        EB_tr, EB_te = sv._standardize(B_tr, B_te)
        k_eff = int(min(cfg.k, EA_tr.shape[1], EB_tr.shape[1]))
        model = KCCA(
            latent_dimensions=k_eff,
            kernel=cfg.kernel,
            c=[cfg.c, cfg.c],
            gamma=cfg.gamma,
        ).fit((EA_tr, EB_tr))
        sa, sb = model.transform((EA_te, EB_te))
        return sv._heldout_corr(np.asarray(sa), np.asarray(sb))  # signed (see fit_score_cca)
    if arm == "deep":
        EA_tr, EA_te = sv._standardize(A_tr, A_te)
        EB_tr, EB_te = sv._standardize(B_tr, B_te)
        ZA_tr, ZB_tr, ZA_te, ZB_te = _deep_representations(
            EA_tr, EB_tr, EA_te, EB_te, cfg, seed
        )
        # linear CCA on the learned representations for ordered canonical corrs
        return sv.fit_score_cca(ZA_tr, ZB_tr, ZA_te, ZB_te, cfg.k, cfg.k, cfg.c, seed)
    if arm == "fm_deep":
        ZA_tr, ZB_tr, ZA_te, ZB_te = _finetune_fm_dcca(
            A_tr, B_tr, A_te, B_te, cfg, seed
        )
        return sv.fit_score_cca(ZA_tr, ZB_tr, ZA_te, ZB_te, cfg.k, cfg.k, cfg.c, seed)
    if arm == "tabfm":
        EA_tr, EA_te = sv.embed_view_tabfm(
            A_tr, A_te, pseudo_target=cfg.pseudo_target, device=cfg.device, seed=seed
        )
        EB_tr, EB_te = sv.embed_view_tabfm(
            B_tr, B_te, pseudo_target=cfg.pseudo_target, device=cfg.device, seed=seed
        )
        return sv.fit_score_cca(EA_tr, EB_tr, EA_te, EB_te, cfg.n_pca, cfg.k, cfg.c, seed)
    if arm == "tabfm_deep":
        ZA_tr, ZB_tr, ZA_te, ZB_te = _finetune_tabfm_dcca(
            A_tr, B_tr, A_te, B_te, cfg, seed
        )
        return sv.fit_score_cca(ZA_tr, ZB_tr, ZA_te, ZB_te, cfg.k, cfg.k, cfg.c, seed)
    raise ValueError(f"unknown arm {arm!r}")


# --- SHARP evaluation -------------------------------------------------------


@dataclass
class SharpResult:
    """Per-arm SHARP half-statistics.

    ``D_A``/``D_B`` map each arm to a length-``J`` array of leading held-out
    canonical correlations, one per repetition-half. Because all arms share the
    same folds, ``D_A[arm1] - D_A[arm2]`` is a valid paired difference.
    """

    D_A: dict[str, NDArray[np.float64]]
    D_B: dict[str, NDArray[np.float64]]
    arms: list[str]
    J: int
    K: int
    dim: int = 0  # canonical dimension the statistic is taken from (leading)


def sharp_eval(
    views: sv.Views,
    arms: Sequence[str] = LADDER,
    J: int = 30,
    K: int = 5,
    configs: dict[str, ArmConfig] | None = None,
    dim: int = 0,
    seed: int = 0,
    verbose: bool = True,
) -> SharpResult:
    """Run the SHARP procedure for all ``arms`` on shared folds.

    For each of ``J`` repetitions: split subjects into two disjoint halves
    (stratified by ``views.strata``); within each half run stratified ``K``-fold
    CV; for each fold score every arm on the held-out fold; average the
    leading-dimension held-out canonical correlation across folds to one
    statistic per arm per half.
    """
    configs = configs or DEFAULT_CONFIGS
    arms = list(arms)
    D_A = {a: np.full(J, np.nan) for a in arms}
    D_B = {a: np.full(J, np.nan) for a in arms}

    for j in range(J):
        outA, outB = _eval_one_rep(views, arms, K, configs, dim, seed + j)
        for a in arms:
            D_A[a][j] = outA[a]
            D_B[a][j] = outB[a]
        if verbose:
            msg = "  ".join(f"{a}={outA[a]:.3f}/{outB[a]:.3f}" for a in arms)
            print(f"[sharp] rep {j + 1}/{J} (A/B): {msg}")
    return SharpResult(D_A=D_A, D_B=D_B, arms=arms, J=J, K=K, dim=dim)


def _eval_one_rep(
    views: sv.Views,
    arms: Sequence[str],
    K: int,
    configs: dict[str, ArmConfig],
    dim: int,
    rep_seed: int,
) -> tuple[dict[str, float], dict[str, float]]:
    """One SHARP repetition: disjoint halves, K-fold within each, all arms.

    Returns ``(outA, outB)`` mapping each arm to its half-A and half-B statistic
    (mean leading held-out canonical correlation across the inner folds). All
    arms share the same folds so the two outputs are paired across arms. This is
    the unit of work parallelized across GPUs in :mod:`cogmood_analysis.sharp_parallel`.
    """
    arms = list(arms)
    hs = StratifiedKFold(n_splits=2, shuffle=True, random_state=rep_seed)
    halves = [idx for _, idx in hs.split(views.A, views.strata)]
    out: list[dict[str, float]] = []
    for half_idx in halves:
        Ah, Bh, strat_h = views.A[half_idx], views.B[half_idx], views.strata[half_idx]
        inner = StratifiedKFold(n_splits=K, shuffle=True, random_state=rep_seed)
        per_arm_folds: dict[str, list[float]] = {a: [] for a in arms}
        for tr, te in inner.split(Ah, strat_h):
            for a in arms:
                r = score_arm(Ah[tr], Bh[tr], Ah[te], Bh[te], a, configs[a], seed=rep_seed)
                per_arm_folds[a].append(float(r[dim]) if r.size > dim else np.nan)
        out.append({a: float(np.nanmean(per_arm_folds[a])) for a in arms})
    return out[0], out[1]


# --- SHARP inference: paper's null-constrained score test -------------------
#
# Model (Zeng et al. 2026, main text + Supplementary S7): the 2J half-statistics
# d = [D_A1..D_AJ, D_B1..D_BJ] are jointly Gaussian, mean mu, variance sigma^2, and
# common correlation rho on every off-diagonal pair EXCEPT the paired (D_Aj, D_Bj)
# which are independent. The structured correlation matrix R has eigenvalues
#   lambda1 = 1 + 2 rho (J-1)   (x1, the all-ones direction),
#   lambda2 = 1 - 2 rho         (x(J-1), symmetric, orthogonal to 1),
#   lambda3 = 1                 (xJ, antisymmetric),
# so R is positive-definite iff rho in (-1/(2(J-1)), 1/2) -- note rho CAN be
# negative. Var(D_bar) = sigma^2 lambda1 / (2J) = sigma^2 (1/(2J) + (J-1)/J rho).


def _sharp_moments(
    D_A: NDArray[np.float64], D_B: NDArray[np.float64]
) -> tuple[float, float, float, float]:
    """Method-of-moments estimate (paper S7.3) -- REFERENCE ONLY.

    Retained so the calibration test can demonstrate its false-positive
    inflation; reported inference uses :func:`sharp_score_test`. ``sigma^2`` from
    within-pair differences (``E[(D_Aj-D_Bj)^2]=2 sigma^2``); ``rho`` from the
    centred variance of the pair means; both mean-free, ``rho`` clipped to
    ``[0, .499]`` (part of why it miscalibrates).
    """
    mask = ~(np.isnan(D_A) | np.isnan(D_B))
    a, b = D_A[mask], D_B[mask]
    J = a.size
    if J < 2:
        return float(np.mean([a.mean(), b.mean()])), float("nan"), float("nan"), float("nan")
    D_bar = 0.5 * (a.mean() + b.mean())
    sigma2 = float(np.mean((a - b) ** 2) / 2.0)
    s = 0.5 * (a + b)
    V_s = float(np.var(s, ddof=1))
    rho = 0.5 - (V_s / sigma2 if sigma2 > 1e-12 else 0.0)
    rho = float(np.clip(rho, 0.0, 0.499))
    var_Dbar = sigma2 * (1.0 / (2 * J) + (J - 1) / J * rho)
    return float(D_bar), float(max(var_Dbar, 0.0)), sigma2, rho


def _quadratic_form(a: NDArray, b: NDArray, mu0: float, rho: float, J: int) -> float:
    """r^T R(rho)^-1 r for r = [a-mu0, b-mu0], via the eigendecomposition of R."""
    rA, rB = a - mu0, b - mu0
    u = (rA + rB) / np.sqrt(2.0)          # symmetric coords
    v = (rA - rB) / np.sqrt(2.0)          # antisymmetric coords (lambda3 = 1)
    Su2, Sv2 = float(u @ u), float(v @ v)
    s0sq = (u.sum() ** 2) / J             # energy on the all-ones direction
    lam1 = 1.0 + 2.0 * rho * (J - 1)
    lam2 = 1.0 - 2.0 * rho
    return s0sq / lam1 + (Su2 - s0sq) / lam2 + Sv2


def _null_constrained_mle(a: NDArray, b: NDArray, mu0: float) -> tuple[float, float]:
    """MLE of (sigma^2, rho) with the mean fixed at ``mu0`` (Gaussian likelihood).

    Concentrates sigma^2 = qform/(2J) and profiles the 1-D objective
    ``2J log(qform(rho)) + log|R(rho)|`` over the PD interval for rho.
    """
    J = a.size
    lo, hi = -1.0 / (2.0 * (J - 1)) + 1e-9, 0.5 - 1e-9

    def neg2ll(rho: float) -> float:
        q = _quadratic_form(a, b, mu0, rho, J)
        logdetR = np.log(1.0 + 2.0 * rho * (J - 1)) + (J - 1) * np.log(1.0 - 2.0 * rho)
        return 2 * J * np.log(max(q, 1e-300)) + logdetR

    res = optimize.minimize_scalar(neg2ll, bounds=(lo, hi), method="bounded")
    rho0 = float(res.x)
    sigma2_0 = _quadratic_form(a, b, mu0, rho0, J) / (2 * J)
    return sigma2_0, rho0


def sharp_score_test(
    D_A: NDArray[np.float64],
    D_B: NDArray[np.float64],
    mu0: float = 0.0,
    alternative: str = "two-sided",
) -> dict[str, float]:
    """Paper-faithful SHARP score test of ``H0: mu = mu0``.

    ``alternative`` is ``"two-sided"``, ``"greater"`` (mu > mu0) or ``"less"``.
    Returns ``D_bar``, the score ``z``, ``p``, and the null-constrained
    ``sigma2``/``rho`` and ``var_Dbar``.
    """
    mask = ~(np.isnan(D_A) | np.isnan(D_B))
    a, b = np.asarray(D_A, float)[mask], np.asarray(D_B, float)[mask]
    J = a.size
    D_bar = 0.5 * (a.mean() + b.mean()) if J else float("nan")
    if J < 2:
        return {"D_bar": D_bar, "z": float("nan"), "p": float("nan"),
                "sigma2": float("nan"), "rho": float("nan"),
                "var_Dbar": float("nan"), "J": int(J)}
    sigma2_0, rho0 = _null_constrained_mle(a, b, mu0)
    var_Dbar = sigma2_0 * (1.0 / (2 * J) + (J - 1) / J * rho0)
    se = float(np.sqrt(max(var_Dbar, 0.0)))
    z = (D_bar - mu0) / se if se > 1e-12 else 0.0
    if alternative == "greater":
        p = float(1 - stats.norm.cdf(z))
    elif alternative == "less":
        p = float(stats.norm.cdf(z))
    else:
        p = float(2 * (1 - stats.norm.cdf(abs(z))))
    return {"D_bar": float(D_bar), "z": float(z), "p": p, "sigma2": float(sigma2_0),
            "rho": float(rho0), "var_Dbar": float(var_Dbar), "J": int(J)}


def _invert_score_test(
    a: NDArray, b: NDArray, alpha: float = 0.05
) -> tuple[float, float]:
    """Test-inversion CI: the set of ``mu0`` with two-sided ``p >= alpha``."""
    D_bar = 0.5 * (a.mean() + b.mean())

    def p_of(mu0: float) -> float:
        return sharp_score_test(a, b, mu0=mu0, alternative="two-sided")["p"]

    def bound(direction: int) -> float:
        # expand a step outward until the test rejects, then bisect the boundary
        step = max(abs(D_bar), 1e-3)
        far = D_bar
        for _ in range(60):
            far = far + direction * step
            if p_of(far) < alpha:
                break
            step *= 2.0
        else:
            return far  # never rejected within range
        near = far - direction * step  # last non-rejecting point (p >= alpha)
        for _ in range(80):
            mid = 0.5 * (near + far)
            if p_of(mid) >= alpha:
                near = mid
            else:
                far = mid
        return near

    return bound(-1), bound(+1)


def sharp_ci(
    res: SharpResult, arm: str, alpha: float = 0.05
) -> dict[str, float]:
    """SHARP confidence interval for one arm via score-test inversion."""
    mask = ~(np.isnan(res.D_A[arm]) | np.isnan(res.D_B[arm]))
    a, b = res.D_A[arm][mask], res.D_B[arm][mask]
    st = sharp_score_test(a, b, mu0=0.0)
    if a.size < 2:
        lo = hi = float("nan")
    else:
        lo, hi = _invert_score_test(a, b, alpha=alpha)
    D_bar = st["D_bar"]
    return {
        "mean": D_bar,
        "lo": lo,
        "hi": hi,
        "shared_variance": D_bar ** 2,
        "sigma2": st["sigma2"],
        "rho": st["rho"],
        "z": st["z"],
        "p": st["p"],
        "J": int(a.size),
    }


def sharp_compare(
    res: SharpResult, arm1: str, arm2: str, alpha: float = 0.05
) -> dict[str, float]:
    """SHARP score test that arm1 and arm2 differ in held-out canonical r.

    Uses the paired per-half differences (same folds), so the comparison is
    fold-dependence-aware, with a test-inversion CI on the difference.
    """
    dA = res.D_A[arm1] - res.D_A[arm2]
    dB = res.D_B[arm1] - res.D_B[arm2]
    st = sharp_score_test(dA, dB, mu0=0.0, alternative="two-sided")
    mask = ~(np.isnan(dA) | np.isnan(dB))
    lo, hi = (_invert_score_test(dA[mask], dB[mask], alpha=alpha)
              if mask.sum() >= 2 else (float("nan"), float("nan")))
    return {
        "arm1": arm1,
        "arm2": arm2,
        "diff": st["D_bar"],
        "z": st["z"],
        "p": st["p"],
        "lo": lo,
        "hi": hi,
        "sigma2": st["sigma2"],
        "rho": st["rho"],
    }


# --- within-strata permutation null (above-chance test) ---------------------


def sharp_permutation_null(
    views: sv.Views,
    arm: str,
    J: int = 20,
    K: int = 5,
    n_perm: int = 200,
    config: ArmConfig | None = None,
    dim: int = 0,
    block_within_strata: bool = True,
    seed: int = 0,
    verbose: bool = False,
) -> dict[str, Any]:
    """Permutation test that ``arm``'s held-out canonical r exceeds chance.

    Implements the two fixes discussed:

    * **Within-strata (block) permutation** (default): View B is shuffled only
      *within* each ``backfilled_prolific_screen_group`` stratum, so the null
      respects the screening-group structure that the splits are stratified on
      (free permutation would let group-mediated covariance count as signal).
    * **Matched statistic**: the observed and null statistics are computed with
      the *identical* SHARP procedure (same ``D_bar`` estimator over the same
      ``J``/``K`` structure), so they are directly comparable.
    """
    config = config or DEFAULT_CONFIGS.get(arm, ArmConfig())
    rng = np.random.default_rng(seed)

    def observed_stat(B: NDArray) -> float:
        v = sv.Views(
            A=views.A, B=B, sub_ids=views.sub_ids, strata=views.strata,
            a_columns=views.a_columns, b_columns=views.b_columns, n_total=views.n_total,
        )
        res = sharp_eval(v, arms=[arm], J=J, K=K, configs={arm: config},
                         dim=dim, seed=seed, verbose=False)
        return _sharp_moments(res.D_A[arm], res.D_B[arm])[0]

    obs = observed_stat(views.B)

    def permute_B() -> NDArray:
        perm = np.arange(views.B.shape[0])
        if block_within_strata:
            for g in np.unique(views.strata):
                idx = np.where(views.strata == g)[0]
                perm[idx] = rng.permutation(idx)
        else:
            perm = rng.permutation(perm)
        return views.B[perm]

    null = np.empty(n_perm, dtype=np.float64)
    for p in range(n_perm):
        null[p] = observed_stat(permute_B())
        if verbose and (p + 1) % 10 == 0:
            print(f"[perm:{arm}] {p + 1}/{n_perm}")
    pval = float((np.sum(null >= obs) + 1) / (n_perm + 1))
    return {"arm": arm, "observed": obs, "null": null, "p": pval,
            "block_within_strata": block_within_strata}
