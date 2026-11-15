"""
minimum-expected-risk (Bayesian) decision

State space  S = {CORRECT, EXEC_FAIL, WRONG}
Action space A = {EXECUTE, RETRY, CLARIFY}

The posterior over states comes from the statistical classifier
(P(EXEC_FAIL|X), P(WRONG|X)) optionally refined by multi-sample consistency.
Given the cost matrix ``C[a, s]`` the optimal action is

        a* = argmin_a  R(a),      R(a) = sum_s C[a, s] * P(s | X)

One detail makes the model non-trivial: RETRY is not an outcome, it is a
*gamble*.  With probability rho the extra round repairs the statement,
otherwise the same risk is still there.  Hence

        R(RETRY) = C_retry + (1 - rho) * [C_exec * P(EXEC_FAIL) + C_wrong * P(WRONG)]

Without that term RETRY would be optimal for every non-zero risk, which is
exactly the blind "always retry 3 times" behaviour the project replaces.

Closed-form decision regions (two-state approximation, P(EXEC_FAIL)=0):

        P(WRONG) <  tau_low   = C_retry / (rho * C_wrong)          -> EXECUTE
        tau_low <= P(WRONG) < tau_high                              -> RETRY
        P(WRONG) >= tau_high  = (C_clarify - C_retry)/((1-rho) C_wrong) -> CLARIFY

``tau_low`` and ``tau_high`` are derived, not hand-tuned: they change
automatically when the cost matrix or the retry-recovery rate changes.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, List, Sequence, Tuple

import numpy as np

from config import COST_PRESETS, CostModel, DEFAULT_COST_PRESET

EXECUTE, RETRY, CLARIFY = "EXECUTE", "RETRY", "CLARIFY"
ACTIONS: Tuple[str, ...] = (EXECUTE, RETRY, CLARIFY)
STATES: Tuple[str, ...] = ("CORRECT", "EXEC_FAIL", "WRONG")


@dataclass
class Decision:
    action: str
    expected_risk: Dict[str, float]
    posterior: Dict[str, float]
    p_wrong: float
    p_exec_fail: float
    margin: float          # risk gap between best and second-best action
    tau_low: float
    tau_high: float

    def to_dict(self) -> Dict:
        return asdict(self)


class BayesianSQLDecision:
    """Risk-minimising action selector."""

    def __init__(self, cost_model: CostModel | None = None):
        self.cost = cost_model or COST_PRESETS[DEFAULT_COST_PRESET]

    # ------------------------------------------------------------------
    # Core
    # ------------------------------------------------------------------
    def risk_matrix(self) -> np.ndarray:
        c = self.cost
        rho = c.retry_recovery_rate
        return np.array([
            # CORRECT,          EXEC_FAIL,                   WRONG
            [0.0,              c.C_exec_fail,               c.C_wrong_accepted],
            [c.C_retry,        c.C_retry + (1 - rho) * c.C_exec_fail,
             c.C_retry + (1 - rho) * c.C_wrong_accepted],
            [c.C_clarify,      c.C_clarify,                 c.C_clarify],
        ])

    @staticmethod
    def _normalise(p_exec_fail: float, p_wrong: float) -> Tuple[float, float]:
        """Clip to [0,1] and renormalise if the two heads disagree.

        The exec-fail and wrong heads come from different models, so nothing
        guarantees P(EXEC_FAIL) + P(WRONG) <= 1.  Rescaling keeps the
        posterior a proper distribution instead of silently producing risks
        that add up to more than the loss of a certainly-wrong answer.
        """
        pe = float(min(max(p_exec_fail, 0.0), 1.0))
        pw = float(min(max(p_wrong, 0.0), 1.0))
        total = pe + pw
        if total > 1.0:
            pe, pw = pe / total, pw / total
        return pe, pw

    def expected_risk(self, p_exec_fail: float, p_wrong: float) -> np.ndarray:
        """Expected risk of each action for a given posterior over states."""
        pe, pw = self._normalise(p_exec_fail, p_wrong)
        posterior = np.array([1.0 - pe - pw, pe, pw])
        return self.risk_matrix() @ posterior

    def decide(self, p_wrong: float, p_exec_fail: float = 0.0) -> Decision:
        p_exec_fail, p_wrong = self._normalise(p_exec_fail, p_wrong)
        risks = self.expected_risk(p_exec_fail, p_wrong)
        order = np.argsort(risks)
        best, second = int(order[0]), int(order[1])
        return Decision(
            action=ACTIONS[best],
            expected_risk={a: float(r) for a, r in zip(ACTIONS, risks)},
            posterior={
                "CORRECT": float(1 - min(p_wrong + p_exec_fail, 1.0)),
                "EXEC_FAIL": float(p_exec_fail),
                "WRONG": float(p_wrong),
            },
            p_wrong=float(p_wrong),
            p_exec_fail=float(p_exec_fail),
            margin=float(risks[second] - risks[best]),
            tau_low=self.cost.execute_threshold(),
            tau_high=self.cost.clarify_threshold(),
        )

    # ------------------------------------------------------------------
    # Batch helpers (vectorised, used by the experiment replay)
    # ------------------------------------------------------------------
    def decide_batch(self, p_wrong: Sequence[float],
                     p_exec_fail: Sequence[float] | None = None) -> List[Decision]:
        pe = list(p_exec_fail) if p_exec_fail is not None else [0.0] * len(p_wrong)
        return [self.decide(pw, pe_i) for pw, pe_i in zip(p_wrong, pe)]

    def action_grid(self, n: int = 101) -> Tuple[np.ndarray, List[str]]:
        grid = np.linspace(0.0, 1.0, n)
        return grid, [self.decide(p).action for p in grid]

    # ------------------------------------------------------------------
    # Sensitivity analysis on the cost matrix (reported in the thesis)
    # ------------------------------------------------------------------
    def sensitivity(self, p_wrong_values: Sequence[float]) -> List[Dict]:
        rows: List[Dict] = []
        for name, cm in COST_PRESETS.items():
            dec = BayesianSQLDecision(cm)
            for p in p_wrong_values:
                d = dec.decide(p)
                rows.append({"preset": name, "p_wrong": float(p),
                             "action": d.action,
                             "risk_execute": d.expected_risk[EXECUTE],
                             "risk_retry": d.expected_risk[RETRY],
                             "risk_clarify": d.expected_risk[CLARIFY]})
        return rows


def fuse_posterior(p_classifier: float, consistency: float | None,
                   w: float = 0.6) -> float:
    """Stage 2 evidence fusion.

    ``consistency`` in [0, 1] is the agreement of K sampled candidates; low
    agreement is evidence *for* an error, hence ``1 - consistency``.
    The classifier probability keeps the larger weight because it is fitted
    on held-out data, while the consistency signal is a single small sample.
    """
    if consistency is None:
        return float(p_classifier)
    return float(w * p_classifier + (1.0 - w) * (1.0 - consistency))


if __name__ == "__main__":
    dec = BayesianSQLDecision()
    c = dec.cost
    print("cost model:", c.as_dict())
    print(f"tau_low={c.execute_threshold():.3f}  tau_high={c.clarify_threshold():.3f}")
    print("\n  p_wrong  action    R(EXEC)  R(RETRY)  R(CLARIFY)")
    grid, actions = dec.action_grid(11)
    for p, a in zip(grid, actions):
        d = dec.decide(p)
        print(f"  {p:7.2f}  {a:8s} {d.expected_risk[EXECUTE]:8.2f} "
              f"{d.expected_risk[RETRY]:9.2f} {d.expected_risk[CLARIFY]:10.2f}")
