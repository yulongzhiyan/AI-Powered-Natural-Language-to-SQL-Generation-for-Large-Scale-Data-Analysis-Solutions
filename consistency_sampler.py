"""
multi-sample consistency as extra evidence.

Triggered only for *borderline* risk, which is the whole economic point: the
expensive sampling budget is spent where the classifier is undecided, not on
every query (the original "always retry 3 times" policy).

Procedure
---------
1. Draw K candidates at temperature 0.7 (diverse but still coherent).
2. Execute each one in the read-only sandbox.
3. Group the returned result sets into equivalence classes.
4. ``consistency`` = share of the K candidates belonging to the largest
   class.  It is a one-look empirical posterior:

       high consistency  ->  the model "knows" the answer  -> evidence for CORRECT
       low  consistency  ->  the model is guessing         -> evidence for WRONG

Because K is small (5 by default) the estimate is noisy, so the Bayesian
layer fuses it with the classifier instead of trusting it directly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from feature_extractor import SQLFeatureExtractor
from sql_sandbox import execute_sql, results_match


@dataclass
class ConsistencyReport:
    k: int
    n_exec_ok: int
    consistency: float                 # share of samples in the largest class
    exec_success_rate: float
    best_sql: str
    cluster_sizes: List[int] = field(default_factory=list)
    candidate_sqls: List[str] = field(default_factory=list)
    error_rate_estimate: float = 0.0   # 1 - exec_success_rate, kept for logging

    def to_dict(self) -> Dict[str, Any]:
        return {
            "k": self.k, "n_exec_ok": self.n_exec_ok,
            "consistency": self.consistency,
            "exec_success_rate": self.exec_success_rate,
            "cluster_sizes": self.cluster_sizes,
            "best_sql": self.best_sql,
        }


def _cluster(rows_by_candidate: Sequence[Optional[Sequence]]) -> Tuple[List[int], int]:
    """Group candidates by result-set equality; return (sizes, best_cluster)."""
    clusters: List[List[int]] = []
    for idx, rows in enumerate(rows_by_candidate):
        if rows is None:            # execution failed -> its own (ignored) class
            continue
        placed = False
        for cl in clusters:
            if results_match(rows_by_candidate[cl[0]], rows):
                cl.append(idx)
                placed = True
                break
        if not placed:
            clusters.append([idx])
    if not clusters:
        return [], -1
    sizes = [len(c) for c in clusters]
    best = int(max(range(len(clusters)), key=lambda i: sizes[i]))
    return sizes, best


class ConsistencySampler:
    def __init__(self, client, k: int = 5, temperature: float = 0.7):
        self.client = client
        self.k = k
        self.temperature = temperature

    def sample(self, question: str, schema_text: str, db_path: str,
               base_sql: str = "") -> ConsistencyReport:
        candidates: List[str] = []
        results: List[Optional[Sequence]] = []
        for _ in range(self.k):
            sql, _usage = self.client.generate_sql(question, schema_text,
                                                   temperature=self.temperature)
            if not sql:
                continue
            candidates.append(sql)
            ok, out = execute_sql(db_path, sql)
            results.append(list(out) if ok else None)

        if not candidates:
            return ConsistencyReport(k=0, n_exec_ok=0, consistency=0.0,
                                     exec_success_rate=0.0, best_sql=base_sql)

        sizes, best = _cluster(results)
        n_ok = sum(1 for r in results if r is not None)
        k_eff = len(candidates)
        consistency = (sizes[best] / k_eff) if best >= 0 else 0.0
        best_sql = base_sql
        if best >= 0:
            # pick a member of the majority class (deterministic tie-break)
            members = [i for i, r in enumerate(results)
                       if r is not None and results_match(r, results[best])]
            best_sql = candidates[members[0]]
        return ConsistencyReport(
            k=k_eff, n_exec_ok=n_ok, consistency=float(consistency),
            exec_success_rate=float(n_ok / k_eff) if k_eff else 0.0,
            best_sql=best_sql, cluster_sizes=sizes,
            candidate_sqls=candidates,
            error_rate_estimate=1.0 - (n_ok / k_eff if k_eff else 0.0),
        )


def cluster_features(candidates: Sequence[str], extractor: SQLFeatureExtractor,
                     schema) -> Dict[str, float]:
    """Diversity statistics of the candidate set (reported, not required)."""
    if not candidates:
        return {"cand_mean_len": 0.0, "cand_len_std": 0.0, "cand_unique_ratio": 0.0}
    lens = [len(c) for c in candidates]
    mean = sum(lens) / len(lens)
    var = sum((x - mean) ** 2 for x in lens) / len(lens)
    return {
        "cand_mean_len": mean,
        "cand_len_std": var ** 0.5,
        "cand_unique_ratio": len({c.strip().lower() for c in candidates}) / len(candidates),
    }
