"""Figures for the Bayesian decision layer.

All plots are written to ``results/`` as 160-dpi PNGs:

    decision_regions.png     expected risk of each action vs P(wrong)
    risk_distribution.png    P(wrong) distribution, correct vs wrong statements
    policy_comparison.png    cost / calls / accuracy of the five policies
    sensitivity_rho.png      cost per query as a function of the recovery rate
"""

from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from bayesian_decision import (ACTIONS, BayesianSQLDecision)
from config import (COST_PRESETS, DATASET_RAW_PATH, DEFAULT_COST_PRESET,
                    RESULTS_DIR)

ACCENT = {"EXECUTE": "#2a9d8f", "RETRY": "#e9c46a", "CLARIFY": "#e76f51"}


def plot_decision_regions(cost_model) -> None:
    dec = BayesianSQLDecision(cost_model)
    grid = np.linspace(0, 1, 401)
    risks = np.array([dec.expected_risk(0.0, p) for p in grid])   # (n, 3)
    actions = [ACTIONS[i] for i in risks.argmin(axis=1)]

    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    for i, name in enumerate(ACTIONS):
        ax.plot(grid, risks[:, i], lw=1.9, label=f"E[R | {name}]",
                color=ACCENT[name])
    # shade decision regions
    prev, start = actions[0], 0.0
    for x, a in zip(grid, actions):
        if a != prev:
            ax.axvspan(start, x, color=ACCENT[prev], alpha=0.08)
            start, prev = x, a
    ax.axvspan(start, 1.0, color=ACCENT[prev], alpha=0.08)

    top = float(risks.max())
    t_lo = cost_model.execute_threshold()
    t_hi = cost_model.clarify_threshold()
    ax.axvline(t_lo, ls=":", color="k", lw=1)
    ax.axvline(t_hi, ls=":", color="k", lw=1)
    # threshold labels at the bottom so the legend cannot cover them
    ax.text(t_lo, -0.11 * top, f" tau_low={t_lo:.3f}", fontsize=8, va="bottom")
    ax.text(t_hi, -0.11 * top, f" tau_high={t_hi:.3f}", fontsize=8, va="bottom")
    # name each region in the middle of its span
    bounds = [(0.0, t_lo, "EXECUTE"), (t_lo, min(t_hi, 1.0), "RETRY")]
    if t_hi < 1.0:
        bounds.append((t_hi, 1.0, "CLARIFY"))
    for lo, hi, label in bounds:
        if hi - lo > 0.05:
            ax.text((lo + hi) / 2, top * 0.86, label, ha="center",
                    fontsize=9, color="#333333", alpha=0.75)

    ax.set_xlabel("Posterior P(wrong | x)")
    ax.set_ylabel("Expected risk (LLM round-trip units)")
    ax.set_title("Minimum-risk decision regions")
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.14 * top, top * 1.05)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.34), ncol=3,
              frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "decision_regions.png"), dpi=160,
                bbox_inches="tight")
    plt.close(fig)


def plot_risk_distribution() -> None:
    oof = pd.read_csv(os.path.join(RESULTS_DIR, "oof_predictions.csv"))
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.9), sharey=True)
    for ax, key in zip(axes, ("p_lr", "p_rf")):
        for label, colour in ((0, "#2a9d8f"), (1, "#e76f51")):
            ax.hist(oof.loc[oof["y_wrong"] == label, key], bins=25, alpha=0.6,
                    color=colour, density=True,
                    label="correct" if label == 0 else "wrong")
        ax.set_xlabel(f"{key.upper()} - out-of-fold P(wrong)")
        ax.set_title(key.upper())
        ax.legend(fontsize=8)
    axes[0].set_ylabel("density")
    fig.suptitle("Separation of the two classes by the risk model", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "risk_distribution.png"), dpi=160)
    plt.close(fig)


def plot_policy_comparison() -> None:
    res = pd.read_csv(os.path.join(RESULTS_DIR, "policy_comparison.csv"))
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 3.9))
    metric_specs = [
        ("cost_per_query", "Expected cost per query", "{:.1f}"),
        ("calls_per_query", "LLM calls per query", "{:.2f}"),
        ("ex_on_answered", "Execution accuracy on answered", "{:.3f}"),
    ]
    for ax, (col, title, fmt) in zip(axes, metric_specs):
        vals = res[col].values
        colours = ["#8d99ae" if p in ("always_execute", "fixed_retry_3",
                                      "threshold_0.5") else "#2a6f97"
                   for p in res["policy"]]
        bars = ax.bar(range(len(res)), vals, color=colours)
        ax.set_xticks(range(len(res)))
        ax.set_xticklabels(res["policy"], rotation=30, ha="right", fontsize=8)
        ax.set_title(title, fontsize=10)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v, fmt.format(v),
                    ha="center", va="bottom", fontsize=7)
        ax.margins(y=0.15)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "policy_comparison.png"), dpi=160)
    plt.close(fig)


def plot_rho_sensitivity() -> None:
    path = os.path.join(RESULTS_DIR, "sensitivity_rho.csv")
    if not os.path.isfile(path):
        return
    sens = pd.read_csv(path)
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 3.9))
    for name, grp in sens.groupby("policy"):
        grp = grp.sort_values("rho")
        axes[0].plot(grp["rho"], grp["cost_per_query"], marker="o", ms=3,
                     lw=1.6, label=name)
        axes[1].plot(grp["rho"], grp["calls_per_query"], marker="o", ms=3,
                     lw=1.6, label=name)
    axes[0].set_xlabel("retry recovery rate rho")
    axes[0].set_ylabel("expected cost per query")
    axes[0].set_title("Cost vs. verification effectiveness")
    axes[1].set_xlabel("retry recovery rate rho")
    axes[1].set_ylabel("LLM calls per query")
    axes[1].set_title("Compute vs. verification effectiveness")
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "sensitivity_rho.png"), dpi=160)
    plt.close(fig)


def main() -> None:
    cost_model = COST_PRESETS[DEFAULT_COST_PRESET]
    plot_decision_regions(cost_model)
    if os.path.isfile(os.path.join(RESULTS_DIR, "oof_predictions.csv")):
        plot_risk_distribution()
    if os.path.isfile(os.path.join(RESULTS_DIR, "policy_comparison.csv")):
        plot_policy_comparison()
    plot_rho_sensitivity()
    print("Figures written to", RESULTS_DIR)
    for f in ("decision_regions.png", "risk_distribution.png",
              "policy_comparison.png", "sensitivity_rho.png"):
        p = os.path.join(RESULTS_DIR, f)
        print(f"  {'OK ' if os.path.isfile(p) else 'MISSING'} {f}")


if __name__ == "__main__":
    main()
