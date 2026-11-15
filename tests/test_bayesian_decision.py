"""
Unit tests for the Bayesian decision layer

Run with either

    python -m unittest discover -s tests -v
    pytest tests -q


"""

from __future__ import annotations

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bayesian_decision import (ACTIONS, CLARIFY, EXECUTE, RETRY,
                               BayesianSQLDecision, fuse_posterior)
from config import COST_PRESETS, CostModel


class TestDecisionRule(unittest.TestCase):
    def setUp(self) -> None:
        self.cm = COST_PRESETS["balanced"]
        self.dec = BayesianSQLDecision(self.cm)

    # -- basic regions ----------------------------------------------------
    def test_no_risk_executes(self):
        self.assertEqual(self.dec.decide(0.0).action, EXECUTE)

    def test_high_risk_clarifies(self):
        d = self.dec.decide(0.9)
        self.assertEqual(d.action, CLARIFY)

    def test_moderate_risk_retries(self):
        self.assertEqual(self.dec.decide(0.30).action, RETRY)

    def test_action_regions_are_ordered(self):
        """EXECUTE region must be low-p, CLARIFY region high-p: no zig-zag."""
        _, actions = self.dec.action_grid(101)
        rank = {EXECUTE: 0, RETRY: 1, CLARIFY: 2}
        ranks = [rank[a] for a in actions]
        self.assertEqual(ranks, sorted(ranks),
                         "action must be monotone in P(wrong)")

    def test_all_three_actions_reachable(self):
        _, actions = self.dec.action_grid(201)
        self.assertEqual(set(actions), set(ACTIONS))

    # -- thresholds are real indifference points --------------------------
    def test_execute_threshold_is_indifference_point(self):
        p = self.cm.execute_threshold()
        r = self.dec.expected_risk(0.0, p)
        self.assertAlmostEqual(r[0], r[1], places=6)   # EXECUTE == RETRY

    def test_clarify_threshold_is_indifference_point(self):
        p = self.cm.clarify_threshold()
        r = self.dec.expected_risk(0.0, p)
        self.assertAlmostEqual(r[1], r[2], places=6)   # RETRY == CLARIFY

    def test_aggressive_preset_never_clarifies(self):
        dec = BayesianSQLDecision(COST_PRESETS["aggressive"])
        ops = {dec.decide(p).action for p in np.linspace(0, 1, 101)}
        self.assertNotIn(CLARIFY, ops)

    # -- robustness -------------------------------------------------------
    def test_posterior_is_normalised(self):
        d = self.dec.decide(0.8, 0.7)          # inconsistent on purpose
        total = sum(d.posterior.values())
        self.assertAlmostEqual(total, 1.0, places=9)

    def test_zero_risk_has_zero_execute_cost(self):
        r = self.dec.expected_risk(0.0, 0.0)
        self.assertEqual(r[0], 0.0)

    def test_cost_is_non_negative(self):
        for p in np.linspace(0, 1, 11):
            self.assertTrue(np.all(self.dec.expected_risk(0.0, p) >= 0))

    def test_margin_zero_at_boundary(self):
        p = self.cm.execute_threshold()
        self.assertLess(abs(self.dec.decide(p).margin), 0.5)

    def test_higher_error_cost_moves_threshold_down(self):
        cheap = CostModel(C_wrong_accepted=50.0)
        costly = CostModel(C_wrong_accepted=500.0)
        self.assertLess(costly.execute_threshold(), cheap.execute_threshold())


class TestEvidenceFusion(unittest.TestCase):
    def test_none_consistency_returns_classifier(self):
        self.assertAlmostEqual(fuse_posterior(0.4, None), 0.4)

    def test_full_consistency_lowers_risk(self):
        self.assertLess(fuse_posterior(0.6, 1.0), 0.6)

    def test_zero_consistency_raises_risk(self):
        self.assertGreater(fuse_posterior(0.2, 0.0), 0.2)

    def test_weight_is_respected(self):
        p = fuse_posterior(1.0, 0.0, w=0.6)     # 0.6*1 + 0.4*1 = 1.0
        self.assertAlmostEqual(p, 1.0)
        p2 = fuse_posterior(0.0, 1.0, w=0.6)    # 0.6*0 + 0.4*0 = 0.0
        self.assertAlmostEqual(p2, 0.0)


class TestRiskMatrix(unittest.TestCase):
    def test_retry_discounts_the_loss(self):
        cm = COST_PRESETS["balanced"]
        dec = BayesianSQLDecision(cm)
        rho = cm.retry_recovery_rate
        for p in (0.2, 0.5, 0.8):
            r = dec.expected_risk(0.0, p)
            self.assertAlmostEqual(r[1], cm.C_retry + (1 - rho) * r[0])

    def test_execute_is_dominated_when_p_wrong_is_one(self):
        r = BayesianSQLDecision(COST_PRESETS["balanced"]).expected_risk(0.0, 1.0)
        self.assertGreater(r[0], r[1])   # executing a certainly-wrong SQL is worst


if __name__ == "__main__":
    unittest.main(verbosity=2)
