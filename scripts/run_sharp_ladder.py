"""Run the full SHARP CCA ladder across all GPUs and save results.

The heavy compute (SHARP main evaluation + permutation nulls for every arm) is
distributed across GPUs via :mod:`cogmood_analysis.sharp_parallel`. Results are
pickled to ``data/exploratory/sharp_ladder_results.pkl`` for the notebook to
load and plot. Running this as a script (with the ``__main__`` guard) is required
for the ``spawn`` multiprocessing start method used for GPU isolation.

Usage:
    uv run python scripts/run_sharp_ladder.py            # full run
    SV_QUICK=1 uv run python scripts/run_sharp_ladder.py # fast smoke
"""

from __future__ import annotations

import itertools
import os
import pickle
import time
from pathlib import Path

import numpy as np

from cogmood_analysis import shared_variance as sv
from cogmood_analysis import sharp
from cogmood_analysis import sharp_parallel as sp
from cogmood_analysis import provenance as prov

REPO = Path(__file__).resolve().parents[1]
CSV = REPO / "data" / "exploratory" / "training_data.csv"
OUT = REPO / "data" / "exploratory" / "sharp_ladder_results.pkl"


def main() -> None:
    quick = os.environ.get("SV_QUICK") == "1"
    # SV_MAX_RHAT (e.g. "1.1") excludes subjects with any-task max_rhat above it
    # (model-fit convergence exclusion) and writes to a separate results file so
    # the unfiltered results are preserved for comparison.
    max_rhat_env = os.environ.get("SV_MAX_RHAT")
    exclude_rhat = float(max_rhat_env) if max_rhat_env else None
    out = OUT
    if exclude_rhat is not None:
        tag = f"rhat{str(exclude_rhat).replace('.', 'p')}"
        out = REPO / "data" / "exploratory" / f"sharp_ladder_results_{tag}.pkl"

    arms = list(sharp.LADDER)
    gpu_ids = sp.detect_gpus()

    if quick:
        J, K = 4, 3
        J_perm, K_perm = 2, 2
        n_perm = {a: 5 for a in arms}
        configs = {
            "raw": sharp.ArmConfig(n_pca=10),
            "kernel": sharp.ArmConfig(),
            "deep": sharp.ArmConfig(max_epochs=40),
            "fm": sharp.ArmConfig(n_pca=5),
            "fm_deep": sharp.ArmConfig(n_pca=5, ft_epochs=8),
            "tabfm": sharp.ArmConfig(n_pca=5),
            "tabfm_deep": sharp.ArmConfig(n_pca=5, ft_epochs=6),
        }
    else:
        # paper-scale repetitions (J=60, K=5). Permutation nulls kept at >=100 so no
        # arm sits at a coarse p-floor; the cross-fitted FM arms are the compute long pole.
        J, K = 60, 5
        J_perm, K_perm = 5, 3
        n_perm = {"raw": 1000, "kernel": 1000, "deep": 200, "fm": 100, "tabfm": 100}
        configs = sharp.DEFAULT_CONFIGS

    perm_arms = arms

    views = sv.load_views(CSV, exclude_rhat_above=exclude_rhat)
    rhat_msg = f"exclude max_rhat>{exclude_rhat}" if exclude_rhat is not None else "no rhat exclusion"
    print(f"{'QUICK' if quick else 'FULL'} run | N={views.A.shape[0]} of {views.n_total} "
          f"({rhat_msg}) | GPUs={gpu_ids} | J={J} K={K} | arms={arms}", flush=True)
    print(f"results -> {out}", flush=True)

    t0 = time.time()
    res = sp.sharp_eval_parallel(
        views, arms=arms, J=J, K=K, configs=configs, seed=0,
        gpu_ids=gpu_ids, verbose=True,
    )
    print(f"[main eval] done in {time.time() - t0:.0f}s", flush=True)

    cis = {a: sharp.sharp_ci(res, a) for a in arms}
    comparisons = {
        f"{a1}_vs_{a2}": sharp.sharp_compare(res, a1, a2)
        for a1, a2 in itertools.combinations(arms, 2)
    }

    provenance = prov.provenance(
        CSV, views.sub_ids,
        config={"analysis": "sharp_cca_ladder", "quick": quick, "arms": arms,
                "J": J, "K": K, "J_perm": J_perm, "K_perm": K_perm, "n_perm": n_perm,
                "exclude_rhat_above": exclude_rhat, "seed": 0,
                "inference": "score_test", "signed_correlations": True},
    )
    results = {
        "quick": quick, "arms": arms, "J": J, "K": K, "J_perm": J_perm,
        "K_perm": K_perm, "n_perm": n_perm, "n_subjects": int(views.A.shape[0]),
        "a_columns": views.a_columns, "b_columns": views.b_columns,
        "D_A": res.D_A, "D_B": res.D_B, "cis": cis, "comparisons": comparisons,
        "perms": {}, "perms_done": [], "provenance": provenance,
    }
    print(f"[provenance] commit={str(provenance['analysis_source_commit'])[:9]}"
          f" dirty={provenance['dirty']} subjset={provenance['subject_set_sha256'][:12]}", flush=True)

    def checkpoint():
        out.write_bytes(pickle.dumps(results))

    checkpoint()  # save the (expensive) main eval + CIs + comparisons immediately
    print(f"[checkpoint] main eval saved -> {out}", flush=True)

    for a in perm_arms:
        t = time.time()
        results["perms"][a] = sp.sharp_permutation_null_parallel(
            views, a, J=J_perm, K=K_perm, n_perm=n_perm[a], config=configs[a],
            block_within_strata=True, seed=0, gpu_ids=gpu_ids, verbose=True,
        )
        results["perms_done"].append(a)
        checkpoint()  # update after each arm so nothing is lost on interruption
        print(f"[perm {a}] n={n_perm[a]} observed={results['perms'][a]['observed']:.3f} "
              f"p={results['perms'][a]['p']:.4f} ({time.time() - t:.0f}s) [checkpointed]",
              flush=True)

    prov.write_sidecar(out, results["provenance"])
    print(f"\nSaved results -> {out}  (+ provenance sidecar) (total {time.time() - t0:.0f}s)",
          flush=True)

    # console summary
    perms = results["perms"]
    print("\n=== Per-arm held-out leading canonical r (SHARP 95% CI) ===")
    for a in arms:
        c = cis[a]
        pp = perms[a]["p"] if a in perms else float("nan")
        print(f"  {a:8s} r={c['mean']:.3f} CI[{c['lo']:+.3f},{c['hi']:+.3f}] "
              f"sharedvar={c['shared_variance']:.4f} p_perm={pp:.4f}")
    print("\n=== Pairwise SHARP comparisons ===")
    for name, c in comparisons.items():
        print(f"  {name:22s} diff={c['diff']:+.3f} z={c['z']:+.2f} p={c['p']:.4f}")


if __name__ == "__main__":
    main()
