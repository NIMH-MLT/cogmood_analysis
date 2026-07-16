"""Multi-GPU distribution of the SHARP ladder.

The SHARP procedure is embarrassingly parallel: the ``J`` repetitions of the
main evaluation are independent, and the permutations of the null are
independent. This module spreads that work across all available GPUs.

Mechanism: a ``ProcessPoolExecutor`` with the ``"spawn"`` start method and one
worker per GPU. Each worker's initializer pins it to a distinct GPU by setting
``CUDA_VISIBLE_DEVICES`` *before* torch is imported (the package imports torch
lazily inside functions, so by the time a task touches CUDA the worker sees only
its one GPU as ``cuda:0``). Tasks pass plain numpy arrays — never CUDA tensors —
across the process boundary.

Results are numerically identical to the serial :func:`cogmood_analysis.sharp.sharp_eval`
for the same ``seed`` (each repetition uses ``seed + j`` exactly as before).
"""

from __future__ import annotations

import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
from typing import Any, Sequence

import numpy as np
from numpy.typing import NDArray

from . import shared_variance as sv
from . import sharp


def detect_gpus() -> list[int]:
    """Return available CUDA device indices (empty list if none)."""
    try:
        import torch

        return list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
    except Exception:
        return []


def _worker_init(gpu_queue: "mp.Queue") -> None:
    gid = gpu_queue.get()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gid)
    os.environ.setdefault("TABPFN_ALLOW_CPU_LARGE_DATASET", "1")
    try:  # belt-and-suspenders: also cap torch's intra-op threads
        import torch

        torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "4")))
    except Exception:
        pass


def _make_pool(gpu_ids: Sequence[int]) -> ProcessPoolExecutor:
    # Cap BLAS/OMP threads per worker so N workers don't each grab all cores
    # (that oversubscription makes everything thrash). Set in the PARENT so the
    # spawned children inherit it before numpy/torch import.
    n_threads = str(max(1, (os.cpu_count() or 8) // max(1, len(gpu_ids))))
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[var] = n_threads
    ctx = mp.get_context("spawn")
    q: "mp.Queue" = ctx.Queue()
    for g in gpu_ids:
        q.put(g)
    return ProcessPoolExecutor(
        max_workers=len(gpu_ids), mp_context=ctx,
        initializer=_worker_init, initargs=(q,),
    )


def _views_from_arrays(A, B, strata, a_cols, b_cols) -> sv.Views:
    return sv.Views(
        A=A, B=B, sub_ids=np.arange(len(A)).astype(str), strata=strata,
        a_columns=list(a_cols), b_columns=list(b_cols), n_total=len(A),
    )


# --- parallel main evaluation (over repetitions) ----------------------------


def _rep_task(payload: tuple) -> tuple[dict[str, float], dict[str, float]]:
    A, B, strata, a_cols, b_cols, arms, K, configs, dim, rep_seed = payload
    v = _views_from_arrays(A, B, strata, a_cols, b_cols)
    cfgs = {a: replace(configs[a], device="cuda") for a in arms}
    return sharp._eval_one_rep(v, arms, K, cfgs, dim, rep_seed)


def sharp_eval_parallel(
    views: sv.Views,
    arms: Sequence[str] = sharp.LADDER,
    J: int = 30,
    K: int = 5,
    configs: dict[str, sharp.ArmConfig] | None = None,
    dim: int = 0,
    seed: int = 0,
    gpu_ids: Sequence[int] | None = None,
    verbose: bool = True,
) -> sharp.SharpResult:
    """SHARP main evaluation with the ``J`` repetitions spread across GPUs.

    Falls back to the serial implementation if no GPUs are available.
    """
    configs = configs or sharp.DEFAULT_CONFIGS
    arms = list(arms)
    gpu_ids = list(gpu_ids) if gpu_ids is not None else detect_gpus()
    if not gpu_ids:
        return sharp.sharp_eval(views, arms=arms, J=J, K=K, configs=configs,
                                dim=dim, seed=seed, verbose=verbose)

    D_A = {a: np.full(J, np.nan) for a in arms}
    D_B = {a: np.full(J, np.nan) for a in arms}
    payloads = [
        (views.A, views.B, views.strata, views.a_columns, views.b_columns,
         arms, K, configs, dim, seed + j)
        for j in range(J)
    ]
    done = 0
    with _make_pool(gpu_ids) as ex:
        futs = {ex.submit(_rep_task, p): j for j, p in enumerate(payloads)}
        for fut in as_completed(futs):
            j = futs[fut]
            outA, outB = fut.result()
            for a in arms:
                D_A[a][j] = outA[a]
                D_B[a][j] = outB[a]
            done += 1
            if verbose:
                print(f"[sharp||] rep {done}/{J} done (on {len(gpu_ids)} GPUs)")
    return sharp.SharpResult(D_A=D_A, D_B=D_B, arms=arms, J=J, K=K, dim=dim)


# --- parallel permutation null (over permutations) --------------------------


def _perm_task(payload: tuple) -> float:
    (A, B, strata, a_cols, b_cols, arm, cfg, J, K, dim,
     block, base_seed, perm_seed) = payload
    rng = np.random.default_rng(perm_seed)
    perm = np.arange(len(B))
    if block:
        for g in np.unique(strata):
            idx = np.where(strata == g)[0]
            perm[idx] = rng.permutation(idx)
    else:
        perm = rng.permutation(perm)
    v = _views_from_arrays(A, B[perm], strata, a_cols, b_cols)
    cfg = replace(cfg, device="cuda")
    res = sharp.sharp_eval(v, arms=[arm], J=J, K=K, configs={arm: cfg},
                           dim=dim, seed=base_seed, verbose=False)
    return sharp._sharp_moments(res.D_A[arm], res.D_B[arm])[0]


def sharp_permutation_null_parallel(
    views: sv.Views,
    arm: str,
    J: int = 5,
    K: int = 3,
    n_perm: int = 100,
    config: sharp.ArmConfig | None = None,
    dim: int = 0,
    block_within_strata: bool = True,
    seed: int = 0,
    gpu_ids: Sequence[int] | None = None,
    verbose: bool = True,
) -> dict[str, Any]:
    """Within-strata permutation null with permutations spread across GPUs.

    The observed statistic is computed once with the identical SHARP procedure
    (matched). Each permutation re-runs the procedure on a within-strata shuffle
    of View B. Falls back to the serial null if no GPUs are available.
    """
    config = config or sharp.DEFAULT_CONFIGS.get(arm, sharp.ArmConfig())
    gpu_ids = list(gpu_ids) if gpu_ids is not None else detect_gpus()
    if not gpu_ids:
        return sharp.sharp_permutation_null(
            views, arm, J=J, K=K, n_perm=n_perm, config=config, dim=dim,
            block_within_strata=block_within_strata, seed=seed, verbose=verbose,
        )

    obs_res = sharp.sharp_eval(
        views, arms=[arm], J=J, K=K, configs={arm: replace(config, device="cuda")},
        dim=dim, seed=seed, verbose=False,
    )
    obs = sharp._sharp_moments(obs_res.D_A[arm], obs_res.D_B[arm])[0]

    payloads = [
        (views.A, views.B, views.strata, views.a_columns, views.b_columns,
         arm, config, J, K, dim, block_within_strata, seed, seed + 1 + p)
        for p in range(n_perm)
    ]
    null = np.empty(n_perm, dtype=np.float64)
    done = 0
    with _make_pool(gpu_ids) as ex:
        futs = {ex.submit(_perm_task, p): i for i, p in enumerate(payloads)}
        for fut in as_completed(futs):
            null[futs[fut]] = fut.result()
            done += 1
            if verbose and done % 50 == 0:
                print(f"[perm||:{arm}] {done}/{n_perm}")
    pval = float((np.sum(null >= obs) + 1) / (n_perm + 1))
    return {"arm": arm, "observed": obs, "null": null, "p": pval,
            "block_within_strata": block_within_strata}
