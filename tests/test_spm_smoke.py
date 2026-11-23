"""Smoke and correctness tests for the SPM layer. No API key required.

Run: python -m pytest tests/test_spm_smoke.py -q
"""
from __future__ import annotations

import json
import logging
import math
import os
import sys
import tempfile

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
logging.disable(logging.INFO)

from spm import analyze, runner                    # noqa: E402
from spm.agents import SPMAgent                    # noqa: E402
from spm.config import Config                      # noqa: E402
from spm.schema import logit, sigmoid              # noqa: E402
from spm.te import (permutation_null, suspicion_scores,  # noqa: E402
                    transfer_entropy)

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def base_config(**over) -> Config:
    with open(os.path.join(REPO, "configs", "spm_experiment.json")) as f:
        raw = json.load(f)
    raw.update(over)
    return Config.from_dict(raw)


# ------------------------------------------------------------- Theorem 1'

@pytest.mark.parametrize("ref_q,eps,eta,expected_phase", [
    (0.50, 0.15, 0.5, "misdirect"),   # target interior: anti-truth floor is met
    (0.50, 0.02, 0.5, "drag"),        # interval too narrow to reach the floor
    (0.85, 0.15, 0.5, "drag"),        # interval entirely above 0.5: floor unreachable
    (0.20, 0.15, 0.5, "misdirect"),   # interval entirely below 0.5
])
def test_best_response_phase(ref_q, eps, eta, expected_phase):
    cfg = base_config()
    cfg.attack.epsilon, cfg.attack.eta = eps, eta
    agent = SPMAgent("u", "", cfg, theta=1, seed=0)
    m, lo, hi, psi, phase = agent.best_response(ref_q)
    assert lo - 1e-9 <= m <= hi + 1e-9, "m* must stay inside the plausibility interval"
    assert phase == expected_phase
    assert m == pytest.approx(min(max(sigmoid(-eta), lo), hi), abs=1e-9)


def test_best_response_never_leaves_interval_and_mirrors():
    """theta = 0 must be the exact mirror of theta = 1 under m -> 1 - m."""
    cfg = base_config()
    up = SPMAgent("u", "", cfg, theta=1, seed=0)
    down = SPMAgent("d", "", cfg, theta=0, seed=0)
    for ref in (0.2, 0.4, 0.5, 0.6, 0.8):
        m1, *_ = up.best_response(ref)
        m0, *_ = down.best_response(1 - ref)
        assert m0 == pytest.approx(1 - m1, abs=1e-9)


def test_stall_objective_prefers_endpoint_nearest_half():
    """The error the draft makes in Sec. III-D, pinned as a test.

    On an interval strictly below 0.5 with no anti-truth floor (eta = 0), the
    minimiser of psi^2 is the endpoint NEAREST 0.5 -- the larger one -- not the
    smallest value in the interval.
    """
    cfg = base_config()
    cfg.attack.eta = 0.0
    agent = SPMAgent("u", "", cfg, theta=1, seed=0)
    m, lo, hi, _, _ = agent.best_response(0.30)   # I = [0.15, 0.45]
    assert m == pytest.approx(hi, abs=1e-9)
    assert logit(hi) ** 2 < logit(lo) ** 2


# -------------------------------------------------------------- estimators

def test_open_loop_series_has_zero_incoming_te():
    """Thm. 5: a constant or jittered output cannot depend on its input."""
    rng = __import__("random").Random(0)
    driver = [rng.random() for _ in range(200)]
    constant = [0.37] * 200
    jittered = [min(max(0.37 + rng.gauss(0, 0.02), 0.01), 0.99) for _ in range(200)]
    assert transfer_entropy(driver, constant, bins=2) == pytest.approx(0.0, abs=1e-12)
    assert transfer_entropy(driver, jittered, bins=2) == pytest.approx(0.0, abs=0.05)


def test_responsive_series_has_positive_incoming_te():
    """Judged against the permutation null, which is how tau is set in practice --
    absolute TE values are small at B = 3 and are not interpretable on their own."""
    import random as _random

    rng = _random.Random(1)
    driver, follower = [], []
    q = 0.5
    for _ in range(300):
        d = rng.choice([0.15, 0.85])          # a driver that actually carries a symbol
        driver.append(d)
        q = min(max(0.2 * q + 0.8 * d + rng.gauss(0, 0.02), 0.01), 0.99)
        follower.append(q)
    observed = transfer_entropy(driver, follower, bins=2)
    null = permutation_null(driver, follower, bins=2, draws=200, rng=_random.Random(2))
    assert observed > float(np.quantile(null, 0.95))


def test_suspicion_score_is_one_sided_low():
    te = {"quiet": 0.0, "a": 0.5, "b": 0.55, "c": 0.6}
    scores = suspicion_scores(te)                      # median_ratio, the default
    assert scores["quiet"] == max(scores.values()) == 1.0
    assert scores["c"] == 0.0, "at or above the median is not suspicious at all"
    z = suspicion_scores(te, method="mad_z")           # Def. 4, kept for ablation
    assert z["quiet"] == max(z.values())
    assert z["c"] < 0


def test_mad_z_cannot_fire_on_a_perfectly_open_loop_target():
    """The ceiling that made Def. 4 inert, pinned as a regression.

    TE is non-negative and roughly exponential across a neighbourhood, so MAD is
    of the order of the median. A target at exactly zero therefore scores at most
    median / (1.4826 * MAD) ~ 0.674 -- below the paper's tau = 1.5 no matter how
    clean the separation is. The median ratio has no such ceiling.
    """
    te = {"quiet": 0.0, "a": 0.0044, "b": 0.0087, "c": 0.0219,
          "d": 0.0655, "e": 0.0825}
    assert suspicion_scores(te, method="mad_z")["quiet"] < 1.5
    assert suspicion_scores(te)["quiet"] == 1.0


# ------------------------------------------------------------ full pipeline

@pytest.fixture(scope="module")
def honest_run():
    with tempfile.TemporaryDirectory() as tmp:
        cfg = base_config(run_id="null", n_agents=12, n_steps=100, out_dir=tmp)
        cfg.claim.signal_accuracy = 0.65
        summary = runner.run(cfg)
        yield cfg, summary, os.path.join(tmp, "null", "simulation.db")


def test_honest_run_completes_and_learns(honest_run):
    cfg, summary, db = honest_run
    assert summary["n_agents"] == 12
    assert summary["model_calls"] == 12 * 100
    assert summary["honest_consensus_rate"] > 0.8, "honest agents should find the truth"


def test_calibration_recovers_mock_coefficients(honest_run):
    """The mock backend IS Def. 2 in closed form, with known coefficients.

    If the regression cannot recover them, the bridge from LLM behaviour to rho_c
    is broken and every rho_c reported for the case study is meaningless.
    """
    from spm.llm import MockBackend

    truth = MockBackend()
    cfg, _, db = honest_run
    fit = analyze.calibrate(db, cfg.claim.signal_accuracy, cfg.attack.epsilon)
    assert fit["r2"] > 0.8
    # delta*alpha is the coefficient rho_c is built from, so it has to be right.
    assert fit["delta_alpha"] == pytest.approx(truth.delta_alpha, abs=0.06)
    # And the fit must identify which social rule the agents are running, because
    # that is what decides whether the population pins and the detector goes blind.
    assert fit["social_form"] == truth.social_form
    # kappa is attenuated: the mock reports q to 3 decimals (a real agent gives 2),
    # so the regressors carry measurement error. Direction and scale, not precision.
    assert 0.5 * truth.kappa < fit["kappa"] < 1.5 * truth.kappa
    assert fit["s_bar"] > 0
    assert 0.0 < fit["rho_c"] < 1.0


def test_honest_null_reports_a_converged_population_as_undetectable(honest_run):
    """The gate's job is to say whether responsiveness is measurable at all.

    On a converged population it is not, and the gate must say so rather than
    quietly passing. Incoming TE is only observable while the network still
    disagrees: under the additive form of Def. 2 the population pins at the
    elicitation bound and every series goes constant; under a convex form the
    series keep moving but the social term (mean_j psi - L) goes to zero at
    consensus, so there is again no coupling to measure. Either way the detector's
    usable window is the disagreement phase, not the run length.

    This is why the gate is run on p = 0 before spending anything on a sweep.
    """
    _, _, db = honest_run
    gate = analyze.honest_null(db, bins=2, warmup=10)
    assert set(gate) >= {"hedge_fraction", "te_observed_mean", "te_null_p95",
                         "transient_window", "gate_passed"}
    assert gate["hedge_fraction"] < 0.5, "agents should commit to numbers, not hedge"
    assert gate["median_abs_psi_final"] > 0.5, "beliefs should actually move"
    assert gate["gate_passed"] is False
    assert gate["te_observed_mean"] < gate["te_null_p95"]


def test_detector_ranks_open_loop_neighbours_last_given_enough_data():
    """Positive control for the detector itself, away from the convergence problem.

    One observer, four responsive neighbours, two constant ones. Precision is
    perfect and recall grows with series length -- so when the detector fails on a
    converged population it is the population, not the statistic.
    """
    import math
    import random as _random

    def neighbourhood(T, seed):
        rng = _random.Random(seed)
        li, obs = 0.0, []
        lh = [0.0] * 4
        honest = [[] for _ in range(4)]
        for _ in range(T):
            li = max(-3, min(3, li + 0.035 * rng.choice([1, -1]) + rng.gauss(0, 0.08)))
            obs.append(1 / (1 + math.exp(-li)))
            for k in range(4):
                lh[k] = max(-3, min(3, lh[k] + 0.035 * rng.choice([1, -1])
                                    + 0.15 * li + rng.gauss(0, 0.08)))
                honest[k].append(1 / (1 + math.exp(-lh[k])))
        return obs, honest, [[0.37] * T, [0.37] * T]

    precisions = {}
    for T in (60, 250):
        tp = fp = fn = 0
        for s in range(30):
            obs, honest, manip = neighbourhood(T, s)
            te = {f"h{k}": transfer_entropy(obs, h, 2) for k, h in enumerate(honest)}
            te.update({f"m{k}": transfer_entropy(obs, m, 2) for k, m in enumerate(manip)})
            for name, score in suspicion_scores(te).items():
                flagged, actual = score > 0.95, name.startswith("m")
                tp += flagged and actual
                fp += flagged and not actual
                fn += (not flagged) and actual
        # Not fp == 0. The median ratio buys its recall by giving up the
        # z-score's perfect-precision-zero-recall corner: an honest neighbour
        # that happens to fall below 5% of the neighbourhood median scores above
        # tau too. Measured here that is 2 in 120. Precision, not perfection.
        # Recall is already 1.000 at T = 60 under the median ratio, so what more
        # data buys is precision, not recall: 2 false positives in 120 at T = 60,
        # 0 at T = 250. Under the Def. 4 z-score recall at both lengths was ~0,
        # which is where the old "T >~ 40*B^2" rule of thumb came from -- it was a
        # property of the threshold, not of the estimator.
        assert tp / (tp + fn) == 1.0, "an open-loop target must always be caught"
        precisions[T] = tp / (tp + fp)
    assert precisions[250] >= precisions[60], "more data must not cost precision"
    assert precisions[250] == 1.0


def test_wmsr_retains_in_range_attacker_better_than_byzantine():
    """P5, stated conditionally -- because the unconditional version is false.

    MD Prop. 1 claims an in-range message is never trimmed, "since m* lies between
    two honest values". That only holds if at least f honest neighbours report
    BELOW m*, and the stealth-optimal m* is the lowest plausible value, so it is
    precisely the message most likely to be the neighbourhood minimum. What is
    actually true, and what this pins, is the comparison: an in-range SPM message
    survives an f-trim far more often than a classical extreme one.
    """
    with tempfile.TemporaryDirectory() as tmp:
        kept = {}
        for policy in ("byzantine", "adaptive"):
            cfg = base_config(run_id=policy, n_agents=16, n_steps=40, out_dir=tmp)
            cfg.attack.fraction = 0.25
            cfg.attack.policy = policy
            cfg.defense.kind = "wmsr"
            runner.run(cfg)
            kept[policy] = analyze.retention(
                os.path.join(tmp, policy, "simulation.db"))["manipulator_retention"]
        # Measured: the two are *equal*, and that is the finding.
        #
        # W-MSR trims by order statistic, not by value. The stealth-optimal
        # message is the lowest plausible value in the interval, so it is the
        # rank minimum of its neighbourhood in 69% of logged steps -- exactly
        # where a Byzantine extreme sits. Rank cannot tell 0.378 from 0.01, so
        # both are trimmed at the same rate and both retain 0.273.
        #
        # MD Prop. 1 reads a value filter into W-MSR that is not there. SPM's
        # stealth defeats plausibility thresholds and buys nothing against an
        # f-trim; that is why the defense has to be responsiveness-based at all.
        assert kept["adaptive"] == pytest.approx(kept["byzantine"], abs=0.05)
        assert kept["adaptive"] < 0.5, "an in-range message is NOT reliably retained"



def test_swmsr_severs_open_loop_attacker():
    """The responsiveness filter should remove a constant-message attacker that
    W-MSR keeps."""
    with tempfile.TemporaryDirectory() as tmp:
        results = {}
        for kind in ("wmsr", "swmsr"):
            cfg = base_config(run_id=kind, n_agents=16, n_steps=60, out_dir=tmp)
            cfg.attack.fraction = 0.25
            cfg.attack.policy = "constant"
            cfg.defense.kind = kind
            runner.run(cfg)
            results[kind] = analyze.retention(
                os.path.join(tmp, kind, "simulation.db"))["manipulator_retention"]
        assert results["swmsr"] < results["wmsr"]
