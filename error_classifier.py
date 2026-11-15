"""
a statistical classifier for P(Error | X)

Models
------
* Logistic Regression  -- interpretable, convex, well-calibrated by design,
  coefficients can be read as log-odds per feature.
* Random Forest        -- captures non-linearities and interactions, gives a
  free permutation-free importance ranking, more expressive but less
  calibrated.

Protocol
--------
5-fold *stratified* cross-validation; all metrics are computed on the
out-of-fold predictions so that no query is scored by a model that has seen
it.  This is essential because the out-of-fold scores are later fed to the
Bayesian decision layer -- in-sample scores would make the decision look
better than it is.

Class imbalance
---------------
Failures are a minority class, so both models are fitted with
``class_weight='balanced'``.  That weighting deliberately *distorts* the
output probability (it pretends the prior is 50/50).  For a risk-minimising
decision the probability has to be calibrated again, so apply the exact
closed-form prior correction

        logit(p_cal) = logit(p_bal) + log(pi / (1 - pi))

where ``pi`` is the empirical failure rate.  AUC is unaffected; Brier score
and the cost analysis are not, which is why this step matters.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict
from typing import Dict, List, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, brier_score_loss,
                            confusion_matrix, f1_score, precision_recall_curve,
                            precision_score, recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from config import FEATURES_PATH, FEATURE_NAMES_PATH, MODEL_DIR, RESULTS_DIR


# --------------------------------------------------------------------------
# Probability helpers
# --------------------------------------------------------------------------
def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def prior_correction(p_balanced: np.ndarray, prior: float) -> np.ndarray:
    """Undo the effect of ``class_weight='balanced'`` on the intercept."""
    if not (0.0 < prior < 1.0):
        return p_balanced
    return _sigmoid(_logit(np.asarray(p_balanced)) + np.log(prior / (1 - prior)))


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------
def binary_metrics(y_true: np.ndarray, p: np.ndarray, threshold: float = 0.5) -> Dict:
    y_hat = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_hat, labels=[0, 1]).ravel()
    return {
        "n": int(len(y_true)),
        "n_pos": int(y_true.sum()),
        "base_rate": float(y_true.mean()),
        "roc_auc": float(roc_auc_score(y_true, p)),
        "pr_auc": float(average_precision_score(y_true, p)),
        "accuracy": float(accuracy_score(y_true, y_hat)),
        "precision": float(precision_score(y_true, y_hat, zero_division=0)),
        "recall": float(recall_score(y_true, y_hat, zero_division=0)),
        "f1": float(f1_score(y_true, y_hat, zero_division=0)),
        "brier": float(brier_score_loss(y_true, p)),
        "threshold": float(threshold),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def expected_calibration_error(y_true: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    """Binned |confidence - accuracy|, sample-weighted."""
    edges = np.linspace(0.0, 1.0, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    ece = 0.0
    for b in range(bins):
        m = idx == b
        if not m.any():
            continue
        ece += m.mean() * abs(p[m].mean() - y_true[m].mean())
    return float(ece)


# --------------------------------------------------------------------------
# The classifier wrapper used at inference time
# --------------------------------------------------------------------------
@dataclass
class TrainedModel:
    name: str
    pipeline: Pipeline
    prior: float
    feature_names: List[str]

    def predict_proba(self, X: np.ndarray, calibrate: bool = True) -> np.ndarray:
        p = self.pipeline.predict_proba(X)[:, 1]
        return prior_correction(p, self.prior) if calibrate else p


class SQLErrorClassifier:
    """Train / evaluate / persist the two candidate models."""

    def __init__(self, seed: int = 42, n_splits: int = 5):
        self.seed = seed
        self.n_splits = n_splits
        self.models: Dict[str, TrainedModel] = {}
        self.metrics: Dict[str, Dict] = {}
        self.oof: Dict[str, np.ndarray] = {}

    # -- model zoo --------------------------------------------------------
    def _make_lr(self) -> Pipeline:
        return Pipeline([
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000, C=1.0,
                                       class_weight="balanced",
                                       random_state=self.seed)),
        ])

    def _make_rf(self) -> Pipeline:
        return Pipeline([
            ("clf", RandomForestClassifier(
                n_estimators=400, max_depth=None, min_samples_leaf=2,
                class_weight="balanced_subsample", n_jobs=-1,
                random_state=self.seed)),
        ])

    # -- training ---------------------------------------------------------
    def fit_evaluate(self, X: np.ndarray, y: np.ndarray,
                     feature_names: Sequence[str]) -> Dict[str, Dict]:
        prior = float(y.mean())
        skf = StratifiedKFold(n_splits=self.n_splits, shuffle=True,
                              random_state=self.seed)
        folds = list(skf.split(X, y))

        for name, factory in (("LR", self._make_lr), ("RF", self._make_rf)):
            oof = np.zeros(len(y), dtype=float)
            fold_scores: List[Dict] = []
            for tr, te in folds:
                pipe = factory()
                pipe.fit(X[tr], y[tr])
                oof[te] = pipe.predict_proba(X[te])[:, 1]
            oof_cal = prior_correction(oof, prior)

            m_raw = binary_metrics(y, oof)
            m_cal = binary_metrics(y, oof_cal)
            m_cal["ece"] = expected_calibration_error(y, oof_cal)
            m_raw["ece"] = expected_calibration_error(y, oof)
            self.metrics[name] = {"balanced_prob": m_raw, "calibrated_prob": m_cal}
            self.oof[name] = oof_cal

            # final model on all data, used at inference time
            pipe = factory()
            pipe.fit(X, y)
            self.models[name] = TrainedModel(name=name, pipeline=pipe,
                                             prior=prior,
                                             feature_names=list(feature_names))
        return self.metrics

    # -- inference --------------------------------------------------------
    def predict_proba(self, features: Dict[str, float] | np.ndarray,
                      model: str = "LR") -> float:
        tm = self.models[model]
        if isinstance(features, dict):
            vec = np.array([[features.get(k, 0.0) for k in tm.feature_names]])
        else:
            vec = np.atleast_2d(features)
        return float(tm.predict_proba(vec)[0])

    def save(self, model_dir: str = MODEL_DIR) -> None:
        """Persist as a plain dict, not as a class instance.

        Pickling the wrapper dataclass would embed ``__main__.TrainedModel``
        when this file is executed as a script, which then fails to unpickle
        from another process.  A dict of primitives + an sklearn Pipeline has
        no such problem.
        """
        os.makedirs(model_dir, exist_ok=True)
        for name, tm in self.models.items():
            joblib.dump(
                {"name": tm.name, "pipeline": tm.pipeline, "prior": tm.prior,
                 "feature_names": tm.feature_names},
                os.path.join(model_dir, f"{name.lower()}_risk_model.joblib"),
            )

    @staticmethod
    def load(model_dir: str = MODEL_DIR) -> Dict[str, TrainedModel]:
        out = {}
        for name in ("lr", "rf"):
            path = os.path.join(model_dir, f"{name}_risk_model.joblib")
            if os.path.isfile(path):
                obj = joblib.load(path)
                if isinstance(obj, dict):
                    out[name.upper()] = TrainedModel(**obj)
                else:                       # older artefacts
                    out[name.upper()] = obj
        return out


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------
def _plot_curves(y: np.ndarray, oof: Dict[str, np.ndarray], out_dir: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # ROC + PR
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    for name, p in oof.items():
        fpr, tpr, _ = roc_curve(y, p)
        axes[0].plot(fpr, tpr, lw=1.8,
                     label=f"{name} (AUC={roc_auc_score(y, p):.3f})")
    axes[0].plot([0, 1], [0, 1], "k--", lw=0.8)
    axes[0].set_xlabel("False positive rate")
    axes[0].set_ylabel("True positive rate")
    axes[0].set_title("ROC - out-of-fold predictions")
    axes[0].legend(loc="lower right")

    for name, p in oof.items():
        prec, rec, _ = precision_recall_curve(y, p)
        axes[1].plot(rec, prec, lw=1.8,
                     label=f"{name} (AP={average_precision_score(y, p):.3f})")
    axes[1].axhline(y.mean(), color="k", ls="--", lw=0.8,
                    label=f"base rate={y.mean():.3f}")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].set_title("Precision-Recall - out-of-fold predictions")
    axes[1].legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "classifier_roc_pr.png"), dpi=160)
    plt.close(fig)

    # calibration / reliability diagram
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    ax.plot([0, 1], [0, 1], "k--", lw=0.8, label="perfect")
    for name, p in oof.items():
        frac_pos, mean_pred = calibration_curve(y, p, n_bins=8, strategy="quantile")
        ax.plot(mean_pred, frac_pos, "o-", lw=1.5, ms=4,
                label=f"{name} (ECE={expected_calibration_error(y, p):.3f})")
    ax.set_xlabel("Predicted P(wrong)")
    ax.set_ylabel("Observed failure rate")
    ax.set_title("Reliability diagram (prior-corrected)")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "classifier_calibration.png"), dpi=160)
    plt.close(fig)

    # risk-coverage curve: the single most decision-relevant plot
    fig, ax = plt.subplots(figsize=(5.6, 4.6))
    for name, p in oof.items():
        order = np.argsort(p)                  # most confident (lowest risk) first
        sorted_y = y[order]
        coverages = np.arange(1, len(p) + 1) / len(p)
        remaining_error = np.cumsum(sorted_y) / np.arange(1, len(p) + 1)
        ax.plot(coverages, remaining_error * 100, lw=1.8,
                label=f"{name}")
    ax.axhline(y.mean() * 100, color="k", ls="--", lw=0.8,
               label=f"always-execute ({y.mean() * 100:.1f}%)")
    ax.set_xlabel("Coverage (share of queries executed)")
    ax.set_ylabel("Error rate among executed (%)")
    ax.set_title("Risk-coverage trade-off")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "risk_coverage.png"), dpi=160)
    plt.close(fig)


def _feature_importance(model: TrainedModel, out_dir: str) -> List[Dict]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    clf = model.pipeline.named_steps["clf"]
    names = model.feature_names
    if hasattr(clf, "coef_"):
        vals = clf.coef_[0]
        label = "LR coefficient (log-odds per 1 SD)"
    else:
        vals = clf.feature_importances_
        label = "RF impurity importance"
    order = np.argsort(-np.abs(vals))[:20]
    rows = [{"feature": names[i], "importance": float(vals[i])} for i in order]

    fig, ax = plt.subplots(figsize=(6.4, 6.0))
    ax.barh(range(len(order))[::-1], [vals[i] for i in order],
            color="#4c78a8")
    ax.set_yticks(range(len(order))[::-1])
    ax.set_yticklabels([names[i] for i in order], fontsize=8)
    ax.axvline(0, color="k", lw=0.6)
    ax.set_xlabel(label)
    ax.set_title(f"Top-20 features - {model.name}")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"feature_importance_{model.name.lower()}.png"),
                dpi=160)
    plt.close(fig)
    return rows


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def main() -> None:
    if not os.path.isfile(FEATURES_PATH):
        raise SystemExit(f"{FEATURES_PATH} not found - run build_features.py first.")
    df = pd.read_csv(FEATURES_PATH)
    with open(FEATURE_NAMES_PATH, encoding="utf-8") as fh:
        feature_names = json.load(fh)

    y = df["y_wrong"].astype(int).values
    X = df[feature_names].values.astype(float)

    clf = SQLErrorClassifier(seed=42, n_splits=5)
    metrics = clf.fit_evaluate(X, y, feature_names)
    clf.save()

    # artefacts ---------------------------------------------------------
    metrics_blob = {"n_samples": int(len(y)), "base_rate": float(y.mean()),
                    "metrics": metrics}
    meta_path = os.path.join(os.path.dirname(FEATURES_PATH), "features_meta.json")
    if os.path.isfile(meta_path):
        with open(meta_path, encoding="utf-8") as fh:
            metrics_blob["feature_meta"] = json.load(fh)
    with open(os.path.join(RESULTS_DIR, "classifier_metrics.json"), "w",
              encoding="utf-8") as fh:
        json.dump(metrics_blob, fh, indent=2)

    oof_df = pd.DataFrame({
        "query_id": df["query_id"].values,
        "y_wrong": y,
        "p_lr": clf.oof["LR"],
        "p_rf": clf.oof["RF"],
    })
    oof_df.to_csv(os.path.join(RESULTS_DIR, "oof_predictions.csv"), index=False)

    _plot_curves(y, clf.oof, RESULTS_DIR)
    importance = {name: _feature_importance(tm, RESULTS_DIR)
                  for name, tm in clf.models.items()}
    with open(os.path.join(RESULTS_DIR, "feature_importance.json"), "w",
              encoding="utf-8") as fh:
        json.dump(importance, fh, indent=2)

    # console summary ---------------------------------------------------
    print(f"\nTraining set : {len(y)} queries, failure rate {y.mean():.3f}")
    print(f"CV           : {clf.n_splits}-fold stratified, out-of-fold metrics\n")
    header = f"{'model':6s} {'AUC':>6s} {'AP':>6s} {'PRE':>6s} {'REC':>6s} " \
             f"{'F1':>6s} {'Brier':>6s} {'ECE':>6s}"
    print(header)
    print("-" * len(header))
    for name, m in metrics.items():
        c = m["calibrated_prob"]
        print(f"{name:6s} {c['roc_auc']:6.3f} {c['pr_auc']:6.3f} "
              f"{c['precision']:6.3f} {c['recall']:6.3f} {c['f1']:6.3f} "
              f"{c['brier']:6.3f} {c['ece']:6.3f}")
    print("\nArtefacts -> results/classifier_metrics.json, classifier_roc_pr.png,")
    print("             classifier_calibration.png, risk_coverage.png,")
    print("             feature_importance_*.png, oof_predictions.csv")


if __name__ == "__main__":
    main()
