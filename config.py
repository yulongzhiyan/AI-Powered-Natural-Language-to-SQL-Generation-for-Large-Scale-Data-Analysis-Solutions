"""
Central configuration for the Bayesian risk-controlled NL2SQL.

This module is the single source of truth for paths, model settings, the
cost model used by the Bayesian decision layer, and the experiment.  

"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, asdict
from typing import Dict

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
DATABASE_DIR = os.path.join(DATA_DIR, "database")
SPIDER_DEV_PATH = os.path.join(DATA_DIR, "dev.json")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
CACHE_DIR = os.path.join(PROJECT_ROOT, ".cache")

# Inputs produced by the *original* pipeline (used to rebuild the labelled set)
LEGACY_EVAL_CSV = os.path.join(PROJECT_ROOT, "spider_evaluation_results.csv")
LEGACY_ABLATION_LLM_ONLY = os.path.join(PROJECT_ROOT, "ablation_results_llm_only.csv")
LEGACY_ABLATION_LLM_RAG = os.path.join(PROJECT_ROOT, "ablation_results_llm_rag.csv")

# Artefacts of the new pipeline
DATASET_RAW_PATH = os.path.join(DATA_DIR, "dataset_raw.csv")
FEATURES_PATH = os.path.join(DATA_DIR, "features_train.csv")
FEATURE_NAMES_PATH = os.path.join(DATA_DIR, "feature_names.json")

for _d in (RESULTS_DIR, MODEL_DIR, CACHE_DIR):
    os.makedirs(_d, exist_ok=True)

# --------------------------------------------------------------------------
# LLM settings (unchanged from the original project)
# --------------------------------------------------------------------------
# Never hard-code credentials. Export DEEPSEEK_API_KEY in the shell, or put it
# into a local .env file (already git-ignored). See .env.example.
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
MODEL_NAME = "deepseek-chat"

# Temperature 0.0 = deterministic first draft (original behaviour).
# Temperature 0.7 = diverse sampling used only when the decision layer asks
# for extra evidence (multi-sample consistency).
TEMPERATURE_GREEDY = 0.0
TEMPERATURE_SAMPLING = 0.7

# RAG / schema retrieval
SCHEMA_TOP_K = 3
MAX_REFLECTION_RETRIES = 3      # M in the thesis (self-correction budget)
CONSISTENCY_SAMPLES = 5         # K candidates for the multi-sample stage

# --------------------------------------------------------------------------
# Cost model of the Bayesian decision layer
# --------------------------------------------------------------------------
@dataclass
class CostModel:
    """Cost matrix expressed in *normalised cost units*.

    One cost unit corresponds to the price of one extra LLM round-trip
    (token fee + latency).  This keeps the decision layer independent of
    the exact pricing of the LLM provider and makes the numbers readable:

        C_wrong_accepted : a SQL that executes but returns a wrong result
                           and is delivered to the user (business loss)
        C_exec_fail      : a SQL that raises a sandbox error -> the loop
                           notices it for free, only one call is wasted
        C_retry          : token + latency cost of one extra generation /
                           verification round
        C_clarify        : UX cost of asking the user to rephrase

    ``retry_recovery_rate`` (rho) is the probability that one additional
    round actually repairs a wrong SQL.  It is a *chance node*: RETRY is
    therefore not free, it only buys a chance of avoiding the loss.
    """

    C_wrong_accepted: float = 200.0
    C_exec_fail: float = 1.0
    C_retry: float = 5.0
    C_clarify: float = 120.0
    retry_recovery_rate: float = 0.25

    def as_dict(self) -> Dict[str, float]:
        return asdict(self)

    # Analytic decision thresholds -------------------------------------
    def execute_threshold(self) -> float:
        """tau_low: below this P(wrong) it is cheaper to EXECUTE.

        Derived from  C_retry = rho * p * C_wrong  ->  p = C_retry / (rho*C_wrong)
        """
        denom = max(self.retry_recovery_rate * self.C_wrong_accepted, 1e-9)
        return self.C_retry / denom

    def clarify_threshold(self) -> float:
        """tau_high: above this P(wrong) asking the user is cheapest."""
        denom = max((1.0 - self.retry_recovery_rate) * self.C_wrong_accepted, 1e-9)
        return (self.C_clarify - self.C_retry) / denom


# Three presets, mirroring the "cost sensitivity" section of the proposal.
# The unit of cost is one LLM round-trip, so C_retry is literally the number
# of extra generations a verification round consumes (K = 5 samples).
COST_PRESETS: Dict[str, CostModel] = {
    # Business errors are catastrophic -> verify aggressively.
    "conservative": CostModel(C_wrong_accepted=500.0, C_exec_fail=1.0,
                              C_retry=5.0, C_clarify=300.0,
                              retry_recovery_rate=0.25),
    # Balanced / default: a wrong answer costs as much as 200 calls.
    "balanced": CostModel(C_wrong_accepted=200.0, C_exec_fail=1.0,
                          C_retry=5.0, C_clarify=120.0,
                          retry_recovery_rate=0.25),
    # Retries are relatively expensive compared to the loss -> almost never
    # clarify, accept more risk.
    "aggressive": CostModel(C_wrong_accepted=50.0, C_exec_fail=1.0,
                            C_retry=5.0, C_clarify=60.0,
                            retry_recovery_rate=0.25),
}
DEFAULT_COST_PRESET = "balanced"


# --------------------------------------------------------------------------
# Experiment settings
# --------------------------------------------------------------------------
@dataclass
class ExperimentConfig:
    """Everything that is *assumed* rather than measured.

    These knobs are reported alongside the results so that the reader can
    see which numbers are empirical and which are simulation parameters.
    """

    cost_preset: str = DEFAULT_COST_PRESET
    fusion_weight_classifier: float = 0.6   # w in p_fused = w*p_clf + (1-w)*(1-consistency)
    bootstrap_resamples: int = 10000
    random_seed: int = 42
    label: str = "wrong"                    # "wrong" (semantic) or "exec" (sandbox error)
    extra: Dict[str, float] = field(default_factory=dict)

    def cost_model(self) -> CostModel:
        return COST_PRESETS[self.cost_preset]


def ensure_dirs() -> None:
    for _d in (RESULTS_DIR, MODEL_DIR, CACHE_DIR):
        os.makedirs(_d, exist_ok=True)


if __name__ == "__main__":  # tiny self-check, useful when debugging setups
    cm = COST_PRESETS[DEFAULT_COST_PRESET]
    print("Cost model      :", cm.as_dict())
    print("tau_low (EXEC)  :", round(cm.execute_threshold(), 4))
    print("tau_high(CLARIF):", round(cm.clarify_threshold(), 4))
    print("data dir        :", DATA_DIR)
