""" 
replay of the decision layer on the Spider dev set.
replay the decision on *fixed generations*: for every dev query we already know

    x  (features of the statement that the agent was about to execute)
    y  (whether that statement matched the gold result set)

and the classifier gives an **out-of-fold** p(wrong|x).  What the policy
changes is only the *action*, so the comparison isolates exactly the
contribution of the decision layer, with everything else held constant.

Policies
--------
``always_execute``    accept the first draft                       (1 call)
``fixed_retry_3``     the original paper: error-log driven repair, ceiling of
                      3 attempts.  On this benchmark only 1.2% of the
                      statements fail to execute, so the realised cost is
                      ~1.04 calls/query while *silent* semantic errors are
                      never detected at all.
``threshold_0.5``     naive baseline: verify when p >= 0.5
``bayes_min_risk``    minimum expected risk, unconstrained budget
``bayes_budget_3``    same budget as the original (3 calls/query) but spent
                      on the riskiest 40% of queries  <- headline comparison
``bayes_budget_2``    half that budget (2 calls/query, top 20% verified)

Cost model
----------
Costs are expressed in "LLM round-trips".  A verification round costs K = 5
extra generations, a delivered wrong answer costs ``C_wrong``, asking the
user costs ``C_clarify``.  ``rho`` is the probability that a verification
round actually repairs a wrong statement; it cannot be measured from the
existing logs, so it is reported as a *swept parameter* rather than hidden
inside a single number.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from typing import Callable, Dict, List, Sequence

import numpy as np
import pandas as pd

from bayesian_decision import CLARIFY, EXECUTE, RETRY, BayesianSQLDecision
from config import (COST_PRESETS, DATASET_RAW_PATH, RESULTS_DIR, CostModel,
                    DEFAULT_COST_PRESET, MAX_REFLECTION_RETRIES as MAX_REFLECTION)

K_SAMPLES = 5          # extra generations per verification round


# --------------------------------------------------------------------------
@dataclass
class PolicyOutcome:
    name: str
    action: np.ndarray          # per query: EXECUTE / RETRY / CLARIFY
    calls: np.ndarray           # per query LLM calls
    delivered: np.ndarray       # 1 if an answer is returned to the user
    wrong_delivered: np.ndarray # 1 if a *wrong* answer is returned


def simulate(actions: Sequence[str], y_wrong: np.ndarray, rho: float,
             seed: int, k_samples: int = K_SAMPLES, name: str = "") -> PolicyOutcome:
    """Turn a sequence of actions into realised outcomes.

    A RETRY is resolved with the common-random-numbers trick: the draw for
    query i is seeded by ``seed + i`` so the *same* query recovers or not in
    every policy.  This makes policy differences paired and their bootstrap
    intervals much tighter (variance reduction, not cheating).
    """
    n = len(y_wrong)
    actions = np.asarray(actions)
    calls = np.ones(n, dtype=float)
    delivered = (actions != CLARIFY).astype(int)
    wrong = np.zeros(n, dtype=int)

    retried = actions == RETRY
    calls[retried] += k_samples

    for i in np.where(retried)[0]:
        if y_wrong[i] == 1:
            rng = np.random.default_rng(seed + i)
            recovered = rng.random() < rho
            if not recovered:
                wrong[i] = 1
        # if the statement was already correct, a verification round keeps it
    # untouched queries keep their realised label
    untouched = ~retried
    wrong[untouched] = y_wrong[untouched]
    # clarified queries deliver nothing -> no wrong answer reaches the user
    wrong[actions == CLARIFY] = 0

    return PolicyOutcome(name=name, action=actions, calls=calls,
                         delivered=delivered, wrong_delivered=wrong)


def summarise(outcome: PolicyOutcome, y_wrong: np.ndarray, cost: CostModel) -> Dict:
    n = len(y_wrong)
    n_delivered = int(outcome.delivered.sum())
    n_wrong = int(outcome.wrong_delivered.sum())
    n_retry = int((outcome.action == RETRY).sum())
    n_clarify = int((outcome.action == CLARIFY).sum())
    total_wrong = int(y_wrong.sum())

    ex = 1.0 - (n_wrong / max(n_delivered, 1))
    cost_per_query = (outcome.calls.sum()
                      + cost.C_wrong_accepted * n_wrong
                      + cost.C_clarify * n_clarify) / n
    return {
        "policy": outcome.name,
        "coverage": n_delivered / n,
        "ex_on_answered": ex,
        "ex_over_all": (n_delivered - n_wrong) / n,
        "wrong_delivered": n_wrong,
        "errors_intercepted": 1.0 - n_wrong / max(total_wrong, 1),
        "retry_rate": n_retry / n,
        "clarify_rate": n_clarify / n,
        "calls_per_query": float(outcome.calls.mean()),
        "cost_per_query": cost_per_query,
        "cost_total": cost_per_query * n,
    }


def false_alarm_rate(p: np.ndarray, actions: np.ndarray, y_wrong: np.ndarray,
                     model: BayesianSQLDecision) -> float:
    """Share of *correct* statements that were needlessly escalated."""
    escalated = (actions == RETRY) | (actions == CLARIFY)
    ok = y_wrong == 0
    return float(escalated[ok].mean()) if ok.any() else 0.0


# --------------------------------------------------------------------------
def build_policies(p: np.ndarray, model: BayesianSQLDecision,
                   k_samples: int = K_SAMPLES) -> Dict[str, np.ndarray]:
    n = len(p)
    decisions = [model.decide(pi).action for pi in p]
    bayes = np.array(decisions)

    def budget_policy(target_calls: float) -> np.ndarray:
        """Verify the riskiest share m so that 1 + m*K == target_calls."""
        m = int(round((target_calls - 1.0) * n / k_samples))
        m = max(0, min(m, n))
        order = np.argsort(-p)
        out = np.array([EXECUTE] * n, dtype=object)
        out[order[:m]] = RETRY
        return out.astype(str)

    return {
        "always_execute": np.array([EXECUTE] * n),
        "threshold_0.5": np.where(p >= 0.5, RETRY, EXECUTE).astype(str),
        "bayes_min_risk": bayes,
        "bayes_budget_3": budget_policy(3.0),
        "bayes_budget_2": budget_policy(2.0),
    }


def sensitivity(p: np.ndarray, y: np.ndarray, model: BayesianSQLDecision,
                rhos: Sequence[float], seed: int) -> pd.DataFrame:
    rows = []
    for rho in rhos:
        cm = CostModel(**{**model.cost.as_dict(), "retry_recovery_rate": float(rho)})
        dec = BayesianSQLDecision(cm)
        policies = build_policies(p, dec)
        for name, actions in policies.items():
            out = simulate(actions, y, rho, seed, name=name)
            s = summarise(out, y, cm)
            s["rho"] = float(rho)
            rows.append(s)
        # the original policy does not react to rho at all: a silent error is
        # never retried, so its cost is a flat line in the sweep
        out_fixed = PolicyOutcome(
            name="fixed_retry_3", action=np.array([EXECUTE] * len(y)),
            calls=np.full(len(y), 1.04), delivered=np.ones(len(y), dtype=int),
            wrong_delivered=y.copy())
        s = summarise(out_fixed, y, cm)
        s["rho"] = float(rho)
        rows.append(s)
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description="Replay the decision policies.")
    ap.add_argument("--model", default="LR", choices=["LR", "RF"])
    ap.add_argument("--preset", default=DEFAULT_COST_PRESET,
                    choices=list(COST_PRESETS))
    ap.add_argument("--rho", type=float, default=None,
                    help="override the retry-recovery probability")
    ap.add_argument("--suffix", default="",
                    help="suffix for output files (e.g. 'rf' for the RF model)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    suf = f"_{args.suffix}" if args.suffix else ""

    dataset = pd.read_csv(DATASET_RAW_PATH)
    oof = pd.read_csv(os.path.join(RESULTS_DIR, "oof_predictions.csv"))
    df = dataset.merge(oof.drop(columns=["y_wrong"]), on="query_id", how="inner")
    df = df[df["y_exec_fail"] == 0].reset_index(drop=True)
    y = df["y_wrong"].astype(int).values
    p = df[f"p_{args.model.lower()}"].astype(float).values

    cm = COST_PRESETS[args.preset]
    if args.rho is not None:
        cm = CostModel(**{**cm.as_dict(), "retry_recovery_rate": args.rho})
    model = BayesianSQLDecision(cm)

    print(f"Replay on {len(df)} queries | classifier={args.model} "
          f"| preset={args.preset} | rho={cm.retry_recovery_rate}")
    print(f"Base failure rate: {y.mean():.3f}  ({(y.sum())} wrong of {len(y)})")
    print(f"Decision thresholds: tau_low={cm.execute_threshold():.3f}  "
          f"tau_high={cm.clarify_threshold():.3f}\n")

    policies = build_policies(p, model)
    table: List[Dict] = []
    traces = df[["query_id", "db_id", "question", "y_wrong"]].copy()
    traces[f"p_{args.model.lower()}"] = p

    for name, actions in policies.items():
        out = simulate(actions, y, cm.retry_recovery_rate, args.seed, name=name)
        s = summarise(out, y, cm)
        s["false_alarm_rate"] = false_alarm_rate(p, out.action, y, model)
        table.append(s)
        traces[f"action_{name}"] = out.action
        traces[f"calls_{name}"] = out.calls
        traces[f"wrong_delivered_{name}"] = out.wrong_delivered

    # ---- the original policy, modelled explicitly ----------------------
    # Error-log driven repair: only statements that *fail to execute* trigger
    # a retry (1.2% of the benchmark), and a silent semantic error is returned
    # to the user untouched.  The realised call count is therefore ~1.04, not 3.
    n_exec_fail = int(dataset["y_exec_fail"].sum())
    n_total = len(dataset)
    fixed_calls = 1.0 + float(MAX_REFLECTION) * n_exec_fail / max(n_total, 1)
    out_fixed = PolicyOutcome(
        name="fixed_retry_3",
        action=np.array([EXECUTE] * len(y)),
        calls=np.full(len(y), fixed_calls),
        delivered=np.ones(len(y), dtype=int),
        wrong_delivered=y.copy(),
    )
    s_fixed = summarise(out_fixed, y, cm)
    s_fixed["false_alarm_rate"] = 0.0
    table.append(s_fixed)
    traces["action_fixed_retry_3"] = out_fixed.action
    traces["calls_fixed_retry_3"] = out_fixed.calls
    traces["wrong_delivered_fixed_retry_3"] = out_fixed.wrong_delivered

    res = pd.DataFrame(table)
    order = ["always_execute", "fixed_retry_3", "threshold_0.5",
             "bayes_min_risk", "bayes_budget_3", "bayes_budget_2"]
    res["policy"] = pd.Categorical(res["policy"], categories=order, ordered=True)
    res = res.sort_values("policy")

    # ---- console table -------------------------------------------------
    show = res[["policy", "coverage", "ex_on_answered", "errors_intercepted",
                "retry_rate", "calls_per_query", "cost_per_query"]]
    print(show.to_string(index=False, float_format=lambda v: f"{v:8.4f}"))

    # ---- artefacts -----------------------------------------------------
    res.to_csv(os.path.join(RESULTS_DIR, f"policy_comparison{suf}.csv"), index=False)
    traces.to_csv(os.path.join(RESULTS_DIR, f"decision_trace{suf}.csv"), index=False)

    summary = {
        "n_queries": int(len(df)),
        "classifier": args.model,
        "cost_model": cm.as_dict(),
        "thresholds": {"tau_low": cm.execute_threshold(),
                       "tau_high": cm.clarify_threshold()},
        "base_failure_rate": float(y.mean()),
        "policies": res.to_dict(orient="records"),
    }
    with open(os.path.join(RESULTS_DIR, f"policy_summary{suf}.json"), "w",
              encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    # ---- rho sensitivity ----------------------------------------------
    rhos = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60]
    sens = sensitivity(p, y, model, rhos, args.seed)
    sens.to_csv(os.path.join(RESULTS_DIR, f"sensitivity_rho{suf}.csv"), index=False)

    # ---- cost-matrix sensitivity ---------------------------------------
    rows = []
    for preset, preset_cm in COST_PRESETS.items():
        m = BayesianSQLDecision(preset_cm)
        acts = np.array([m.decide(pi).action for pi in p])
        out = simulate(acts, y, preset_cm.retry_recovery_rate, args.seed,
                       name=preset)
        s = summarise(out, y, preset_cm)
        s["preset"] = preset
        s["tau_low"] = preset_cm.execute_threshold()
        s["tau_high"] = preset_cm.clarify_threshold()
        rows.append(s)
    pd.DataFrame(rows).to_csv(os.path.join(RESULTS_DIR,
                                           f"sensitivity_cost_matrix{suf}.csv"),
                              index=False)

    print(f"\nArtefacts -> results/policy_comparison{suf}.csv, "
          f"decision_trace{suf}.csv,")
    print(f"             policy_summary{suf}.json, sensitivity_rho{suf}.csv,")
    print(f"             sensitivity_cost_matrix{suf}.csv")


if __name__ == "__main__":
    main()
