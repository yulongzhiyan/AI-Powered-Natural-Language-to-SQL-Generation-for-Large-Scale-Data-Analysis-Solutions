"""
The integrated Bayesian risk-controlled NL2SQL

Data flow (the original pipeline plus one inserted decision layer):

    question
      -> schema retrieval ................... rag_store.SchemaIndex
      -> first SQL draft (T=0.0) ............ llm_client
      -> sandbox execution
           |- error -> reflective repair (the original self-correction loop)
           `- ok    -> feature extraction ... feature_extractor
                       P(wrong | x) ........ error_classifier
                       decision ............ bayesian_decision
                         EXECUTE -> deliver the result
                         RETRY   -> multi-sample consistency (K=5, T=0.7)
                                    fuse evidence, decide again, bounded 1 round
                         CLARIFY -> return a clarification request

The agent is deliberately *budgeted*: at most ``max_repair`` reflective
repairs and at most one consistency round.  That is what turns the fixed
"retry 3 times" policy into a risk-proportional one.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Sequence

from bayesian_decision import (CLARIFY, EXECUTE, RETRY, BayesianSQLDecision,
                               fuse_posterior)
from config import (CONSISTENCY_SAMPLES, MAX_REFLECTION_RETRIES,
                    TEMPERATURE_GREEDY)
from consistency_sampler import ConsistencySampler
from error_classifier import SQLErrorClassifier, TrainedModel
from feature_extractor import SQLFeatureExtractor, load_db_schema
from rag_store import SchemaIndex
from sql_sandbox import execute_sql, results_match


@dataclass
class AgentResult:
    question: str
    db_id: str
    status: str                      # answered | clarify | exec_failed
    action_path: List[str] = field(default_factory=list)
    final_sql: str = ""
    p_wrong_initial: float = 0.0
    p_wrong_final: float = 0.0
    consistency: Optional[float] = None
    n_calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_s: float = 0.0
    executed_ok: bool = False
    is_correct: Optional[int] = None
    n_repairs: int = 0
    error: str = ""

    def to_dict(self) -> Dict:
        return asdict(self)


class BayesianNL2SQLAgent:
    def __init__(self, client, classifier: Dict[str, TrainedModel],
                 decision: BayesianSQLDecision | None = None,
                 extractor: SQLFeatureExtractor | None = None,
                 index: SchemaIndex | None = None,
                 model_key: str = "LR",
                 k_samples: int = CONSISTENCY_SAMPLES,
                 max_repair: int = MAX_REFLECTION_RETRIES,
                 fusion_weight: float = 0.6,
                 enable_consistency: bool = True):
        self.client = client
        self.classifier = classifier
        self.model_key = model_key
        self.decision = decision or BayesianSQLDecision()
        self.extractor = extractor or SQLFeatureExtractor()
        self.index = index or SchemaIndex(embedder=self.extractor.embedder)
        self.sampler = ConsistencySampler(client, k=k_samples)
        self.max_repair = max_repair
        self.fusion_weight = fusion_weight
        # Ablation switch: without the sampling budget the RETRY action can
        # only accept the current statement, which isolates the contribution
        # of the multi-sample evidence.
        self.enable_consistency = enable_consistency

    # ------------------------------------------------------------------
    def _predict(self, features: Dict[str, float]) -> float:
        tm = self.classifier[self.model_key]
        import numpy as np
        vec = np.array([[features.get(k, 0.0) for k in tm.feature_names]])
        return float(tm.predict_proba(vec)[0])

    # ------------------------------------------------------------------
    def run(self, question: str, db_id: str, db_path: str,
            gold_rows: Optional[Sequence] = None,
            force_p_wrong: Optional[float] = None) -> AgentResult:
        t0 = time.time()
        usage = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0}
        res = AgentResult(question=question, db_id=db_id, status="unknown")

        def add_usage(u: Dict) -> None:
            for k in usage:
                usage[k] += int(u.get(k, 0))

        schema = load_db_schema(db_path, db_id)
        schema_text, indices = self.index.retrieve(question, db_id, db_path)

        # ---- 1. first draft ------------------------------------------
        sql, u = self.client.generate_sql(question, schema_text,
                                          temperature=TEMPERATURE_GREEDY)
        add_usage(u)
        ok, out = execute_sql(db_path, sql)

        # ---- 2. reflective repair (unchanged original behaviour) -----
        repairs = 0
        while not ok and repairs < self.max_repair:
            repairs += 1
            sql, u = self.client.repair_sql(question, schema_text, sql, str(out),
                                            temperature=TEMPERATURE_GREEDY)
            add_usage(u)
            ok, out = execute_sql(db_path, sql)

        res.n_repairs = repairs
        res.final_sql = sql
        res.executed_ok = ok
        if not ok:
            res.status = "exec_failed"
            res.error = str(out)[:300]
            res.n_calls = usage["calls"]
            res.prompt_tokens = usage["prompt_tokens"]
            res.completion_tokens = usage["completion_tokens"]
            res.latency_s = time.time() - t0
            return res

        # ---- 3. risk scoring ----------------------------------------
        features = self.extractor.extract(question, sql, schema, indices)
        p_wrong = (force_p_wrong if force_p_wrong is not None
                   else self._predict(features))
        res.p_wrong_initial = res.p_wrong_final = p_wrong

        decision = self.decision.decide(p_wrong)
        res.action_path.append(f"{decision.action}(p={p_wrong:.3f})")

        # ---- 4. act --------------------------------------------------
        if decision.action == EXECUTE:
            res.status = "answered"

        elif decision.action == CLARIFY:
            res.status = "clarify"

        elif not self.enable_consistency:
            # Ablation: no sampling budget, so a RETRY decision can only fall
            # back to accepting the statement that is already generated.
            res.action_path.append("ACCEPT(no_budget)")
            res.status = "answered"

        else:  # RETRY -> spend the sampling budget exactly once
            report = self.sampler.sample(question, schema_text, db_path, base_sql=sql)
            # account for K extra generations
            usage["calls"] += report.k
            res.consistency = report.consistency
            p_fused = fuse_posterior(p_wrong, report.consistency,
                                     self.fusion_weight)
            res.p_wrong_final = p_fused
            decision2 = self.decision.decide(p_fused)
            res.action_path.append(f"{decision2.action}(p={p_fused:.3f})")

            if decision2.action == CLARIFY:
                res.status = "clarify"
            else:
                # accept the majority-vote statement when it executes
                if report.best_sql:
                    ok2, out2 = execute_sql(db_path, report.best_sql)
                    if ok2:
                        sql, out, ok = report.best_sql, out2, True
                        res.final_sql = sql
                res.status = "answered"

        # ---- 5. outcome (only when a gold result set is available) ---
        if gold_rows is not None and res.status == "answered":
            res.is_correct = int(ok and results_match(gold_rows, out))

        res.n_calls = usage["calls"]
        res.prompt_tokens = usage["prompt_tokens"]
        res.completion_tokens = usage["completion_tokens"]
        res.latency_s = time.time() - t0
        return res


class FixedRetryAgent:
    """Reference policy: always 3 attempts, no risk model (the original paper).

    Implemented with the same interface so the two policies can be compared
    query by query in a paired setting.
    """

    def __init__(self, client, max_repair: int = MAX_REFLECTION_RETRIES):
        self.client = client
        self.max_repair = max_repair
        self.index = SchemaIndex()
        self.extractor = SQLFeatureExtractor(embedder=self.index.embedder)

    def run(self, question: str, db_id: str, db_path: str,
            gold_rows: Optional[Sequence] = None, **_) -> AgentResult:
        t0 = time.time()
        calls, ptok, ctok = 0, 0, 0
        schema_text, _ = self.index.retrieve(question, db_id, db_path)
        sql, u = self.client.generate_sql(question, schema_text, 0.0)
        calls += 1; ptok += u["prompt_tokens"]; ctok += u["completion_tokens"]
        ok, out = execute_sql(db_path, sql)
        repairs = 0
        while not ok and repairs < self.max_repair:
            repairs += 1
            sql, u = self.client.repair_sql(question, schema_text, sql, str(out), 0.0)
            calls += 1; ptok += u["prompt_tokens"]; ctok += u["completion_tokens"]
            ok, out = execute_sql(db_path, sql)
        correct = None
        if gold_rows is not None:
            correct = int(ok and results_match(gold_rows, out))
        return AgentResult(
            question=question, db_id=db_id,
            status="answered" if ok else "exec_failed",
            action_path=[f"RETRYx{repairs}"] if repairs else ["EXECUTE"],
            final_sql=sql, n_calls=calls, prompt_tokens=ptok,
            completion_tokens=ctok, latency_s=time.time() - t0,
            executed_ok=ok, is_correct=correct, n_repairs=repairs,
        )
