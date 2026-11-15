"""
Hypothesis tests for the policy comparison

Everything here is *paired*: all policies are evaluated on the same 1022
queries, so the right question is not "are the two accuracies different" but
"is the per-query difference centred on zero".

Tests
-----
1. Paired bootstrap (10,000 resamples) on the per-query differences of
   * accuracy, * cost and * LLM calls.  Percentile 95% CI and a two-sided
   p-value are reported.  No normality assumption, which matters because
   correctness is binary.
2. McNemar's exact test on the discordant pairs of two policies: it tests
   whether one policy fixes more queries than it breaks.
3. Paired bootstrap on the AUC difference between the LR and RF risk models.

Output: ``results/statistical_tests.json`` and a readable console table.
"""

from __future__ import annotations

import json
import os
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score

from bayesian_decision import CLARIFY
from config import DATASET_RAW_PATH, RESULTS_DIR, CostModel, COST_PRESETS, DEFAULT_COST_PRESET

B = 10000
SEED = 42


# --------------------------------------------------------------------------
def _per_query(trace: pd.DataFrame, policy: str, cost: CostModel) -> Dict[str, np.ndarray]:
    action = trace[f"action_{policy}"].astype(str).values
    calls = trace[f"calls_{policy}"].astype(float).values
    wrong = trace[f"wrong_delivered_{policy}"].astype(int).values
    delivered = (action != CLARIFY).astype(int)
    correct = delivered - wrong                      # 1 only if right answer returned
    q_cost = (calls
              + cost.C_wrong_accepted * wrong
              + cost.C_clarify * (action == CLARIFY))
    return {"correct": correct, "calls": calls, "cost": q_cost}


def paired_bootstrap(a: np.ndarray, b: np.ndarray, n_boot: int = B,
                     seed: int = SEED) -> Dict:
    """Bootstrap the mean of ``a - b`` (paired differences)."""
    d = a.astype(float) - b.astype(float)
    rng = np.random.default_rng(seed)
    n = len(d)
    idx = rng.integers(0, n, size=(n_boot, n))
    means = d[idx].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    if np.allclose(d, d[0]):
        # Degenerate case (e.g. a constant overhead of 0.03 calls): the
        # bootstrap distribution collapses to a point, so a p-value has no
        # meaning.  Report the point estimate and say so instead of printing
        # a spuriously significant 0.0000.
        return {"mean_diff": float(d.mean()), "ci95": [float(lo), float(hi)],
                "p_value": None, "degenerate": True, "n_pairs": int(n)}
    p = 2.0 * min((means <= 0).mean(), (means >= 0).mean())
    return {"mean_diff": float(d.mean()),
            "ci95": [float(lo), float(hi)],
            "p_value": float(min(p, 1.0)),
            "degenerate": False,
            "n_pairs": int(n)}


def mcnemar(a_correct: np.ndarray, b_correct: np.ndarray) -> Dict:
    """Exact McNemar test on the discordant cells."""
    a = a_correct.astype(int)
    b = b_correct.astype(int)
    n01 = int(((a == 0) & (b == 1)).sum())   # a wrong, b right
    n10 = int(((a == 1) & (b == 0)).sum())   # a right, b wrong
    if n01 + n10 == 0:
        return {"b_a_wrong_b_right": n01, "c_a_right_b_wrong": n10,
                "statistic": 0.0, "p_value": 1.0}
    res = stats.binomtest(n01, n01 + n10, 0.5)
    return {"b_a_wrong_b_right": n01, "c_a_right_b_wrong": n10,
            "statistic": float(n01 - n10), "p_value": float(res.pvalue)}


def bootstrap_auc_diff(y: np.ndarray, p1: np.ndarray, p2: np.ndarray,
                       n_boot: int = 2000, seed: int = SEED) -> Dict:
    rng = np.random.default_rng(seed)
    n = len(y)
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(np.unique(y[idx])) < 2:
            diffs[i] = 0.0
            continue
        diffs[i] = roc_auc_score(y[idx], p1[idx]) - roc_auc_score(y[idx], p2[idx])
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    p = 2.0 * min((diffs <= 0).mean(), (diffs >= 0).mean())
    return {"auc_diff": float(roc_auc_score(y, p1) - roc_auc_score(y, p2)),
            "ci95": [float(lo), float(hi)], "p_value": float(min(p, 1.0))}


# --------------------------------------------------------------------------
def main() -> None:
    trace = pd.read_csv(os.path.join(RESULTS_DIR, "decision_trace.csv"))
    oof = pd.read_csv(os.path.join(RESULTS_DIR, "oof_predictions.csv"))
    dataset = pd.read_csv(DATASET_RAW_PATH)
    usable = set(dataset.loc[dataset["y_exec_fail"] == 0, "query_id"])
    trace = trace[trace["query_id"].isin(usable)].reset_index(drop=True)

    cost = COST_PRESETS[DEFAULT_COST_PRESET]
    policies = ["always_execute", "fixed_retry_3", "threshold_0.5",
                "bayes_min_risk", "bayes_budget_3", "bayes_budget_2"]
    per = {p: _per_query(trace, p, cost) for p in policies}

    comparisons: List[Dict] = []
    ref = "fixed_retry_3"
    for pol in policies:
        if pol == ref:
            continue
        acc = paired_bootstrap(per[pol]["correct"], per[ref]["correct"])
        cst = paired_bootstrap(per[pol]["cost"], per[ref]["cost"])
        cll = paired_bootstrap(per[pol]["calls"], per[ref]["calls"])
        mc = mcnemar(per[pol]["correct"], per[ref]["correct"])
        comparisons.append({
            "policy": pol, "reference": ref,
            "accuracy": acc, "cost": cst, "calls": cll,
            "mcnemar": mc,
        })

    # classifier AUC comparison on the same queries
    merged = oof.merge(pd.DataFrame({"query_id": trace["query_id"]}),
                       on="query_id", how="inner")
    y = merged["y_wrong"].astype(int).values
    auc_cmp = bootstrap_auc_diff(y, merged["p_lr"].values, merged["p_rf"].values)

    out = {
        "n_queries": int(len(trace)),
        "bootstrap_resamples": B,
        "cost_model": cost.as_dict(),
        "comparisons_vs_fixed_retry_3": comparisons,
        "auc_lr_vs_rf": auc_cmp,
    }
    with open(os.path.join(RESULTS_DIR, "statistical_tests.json"), "w",
              encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)

    # ---- console --------------------------------------------------------
    print(f"Paired bootstrap (B={B}) against the original policy "
          f"'{ref}', n={len(trace)} queries\n")
    hdr = (f"{'policy':16s} {'dEX':>8s} {'95% CI':>20s} {'p':>8s} "
           f"{'dCost':>9s} {'95% CI':>20s} {'p':>8s} {'McNemar p':>10s}")
    print(hdr)
    print("-" * len(hdr))
    for c in comparisons:
        a, k, m = c["accuracy"], c["cost"], c["mcnemar"]
        pa = "   n/a  " if a["p_value"] is None else f"{a['p_value']:8.4f}"
        pk = "   n/a  " if k["p_value"] is None else f"{k['p_value']:8.4f}"
        print(f"{c['policy']:16s} {a['mean_diff']:+8.4f} "
              f"[{a['ci95'][0]:+7.4f},{a['ci95'][1]:+7.4f}] {pa} "
              f"{k['mean_diff']:+9.2f} "
              f"[{k['ci95'][0]:+8.2f},{k['ci95'][1]:+8.2f}] {pk} "
              f"{m['p_value']:10.4f}")
    print(f"\nAUC difference LR - RF = {auc_cmp['auc_diff']:+.4f} "
          f"(95% CI [{auc_cmp['ci95'][0]:+.4f}, {auc_cmp['ci95'][1]:+.4f}], "
          f"p={auc_cmp['p_value']:.4f})")
    print(f"\nSaved -> {os.path.join(RESULTS_DIR, 'statistical_tests.json')}")


if __name__ == "__main__":
    main()
