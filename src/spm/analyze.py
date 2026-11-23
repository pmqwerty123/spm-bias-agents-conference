"""Analysis: calibration, the honest null, detector scoring, and aggregation.

The load-bearing piece is calibrate(). There is no kappa, delta or alpha inside a
language model, so rho_c cannot be evaluated for LLM agents without first fitting
Def. 2 to their observed behaviour:

    Delta L_i(t) = kappa * l(x_i(t)) + (delta*alpha) * mean_j psi(m_j(t-1)) + c

Run it on the p = 0 honest run, read off kappa-hat and delta-alpha-hat, compute
rho_c, and write the prediction down *before* looking at the attacked runs. R^2 is
reported because a poor fit is itself a finding about LLM social learning.
"""
from __future__ import annotations

import json
import math
import os
import random
import sqlite3

import numpy as np

from .te import permutation_null, required_steps, transfer_entropy, suspicion_scores


def _logit(q: float) -> float:
    q = min(max(q, 1e-6), 1 - 1e-6)
    return math.log(q / (1 - q))


def _open(db_path: str) -> sqlite3.Connection:
    return sqlite3.connect(db_path)


def _honest_ids(conn) -> list[str]:
    cur = conn.execute(
        "SELECT DISTINCT user_id FROM agent_beliefs WHERE is_manipulator = 0")
    return sorted(r[0] for r in cur.fetchall())


def _neighbours(conn) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for a, b in conn.execute("SELECT follower_id, followed_id FROM follows"):
        out.setdefault(a, []).append(b)
    return out


def _beliefs(conn) -> dict[tuple[str, int], float]:
    return {(r[0], r[1]): r[2]
            for r in conn.execute("SELECT user_id, time_step, q FROM agent_beliefs")}


def transient_window(conn, saturation_q: float = 0.93,
                     plateau_frac: float = 0.05) -> tuple[int, int]:
    """The steps over which honest beliefs are still moving.

    Def. 2's social term is positive feedback -- in consensus the update is
    L <- L(1 + delta*alpha) + kappa*l(x) -- so honest belief runs away and locks
    at the elicitation bound. Once it locks, every series is constant and every
    transfer entropy is zero, honest and manipulator alike: the detector goes
    blind, and a regression of Delta L on its regressors fits nothing.

    Estimating anything on the locked tail is meaningless, so all three analyses
    below run on [0, t_saturate]. If honest agents instead hedge and never leave
    0.5, this returns the whole run and the hedge diagnostic in honest_null() is
    what tells you so.
    """
    beliefs = _beliefs(conn)
    honest = set(_honest_ids(conn))
    steps = sorted({t for (_, t) in beliefs})
    by_step = {}
    for t in steps:
        vals = [q for (u, s), q in beliefs.items() if s == t and u in honest]
        if vals:
            by_step[t] = float(np.median(vals))

    last_moving = 0
    for t, med in by_step.items():
        if 1 - saturation_q < med < saturation_q:
            last_moving = t

    # A second stopping rule, because LLM agents do not saturate the way Def. 2
    # says they should. Def. 2's log-odds are unbounded, so the level rule above
    # assumes a population runs away to the elicitation bound. Measured, they do
    # not: they can plateau at their own confidence ceiling well inside the band.
    # Where that ceiling sits is a decoding-parameter effect, not a constant:
    # |psi| = 1.325 (cx/gpt-5.6-luna, temp 0.7, medium effort), 1.328
    # (gpt-4o-mini, temp 0.7), but 2.35 for the same Luna on OpenAI-direct at
    # temp 1.0 and low effort, still climbing at step 90. So the rule below must
    # detect a plateau rather than assume one at any particular level. The level rule never
    # fires there, so a hundred-step run hands the regression sixty steps of
    # Delta L ~ 0 that the linear model expects to still be drifting.
    #
    # Detected on *remaining* drift rather than step-to-step change: the median of
    # thirty discrete beliefs jitters by up to 0.13 in log-odds even on a dead
    # plateau, so a local-difference test fires at random and never converges.
    # Endpoints are stable where differences are not.
    ordered = sorted(by_step)
    if plateau_frac > 0 and len(ordered) >= 40:
        series = [_logit(by_step[t]) for t in ordered]
        total = abs(series[-1] - series[0])
        if total > 1e-9:
            cutoff = total * plateau_frac
            for i, value in enumerate(series):
                if abs(series[-1] - value) < cutoff and ordered[i] >= 20:
                    last_moving = min(last_moving, ordered[i])
                    break

    # Up to the last step at which honest belief was still off the bound. With a
    # stationary truth that is the pre-lock-in prefix. With a drifting truth the
    # population comes back off the bound after every regime change, so this is
    # most of the run -- which is the whole point of the option.
    return 0, last_moving


def _hidden(conn) -> set[tuple[str, str, int]]:
    """(observer, target, step) triples the defense removed from the feed."""
    return {
        (r[0], r[1], r[2])
        for r in conn.execute(
            "SELECT observer_id, target_id, time_step FROM defense_log "
            "WHERE trimmed = 1 OR severed = 1")
    }


# ------------------------------------------------------------- calibration

def calibrate(db_path: str, accuracy: float, epsilon: float | None = None,
              plateau_frac: float = 0.05) -> dict:
    """Fit Def. 2 to the run. See transient_window for why the window matters.

    rho_c is sensitive to that window and the sensitivity must not be hidden: on
    gpt-4o-mini seed 0 the transient-only fit gives rho_c = 0.177 (5.3 attackers
    in 30) and the whole-run fit gives 0.138 (4.1). The gap is not noise, it is
    misspecification -- kappa is not constant over a run. Agents respond to
    evidence briskly while they are still deciding and barely at all once they
    reach their confidence ceiling, so a whole-run fit averages a live coefficient
    with a dead one and reports something in between. Theorem 3 assumes kappa is
    a constant; measured, it decays.

    Both numbers are therefore returned, and any reported rho_c has to name its
    window.
    """
    conn = _open(db_path)
    beliefs = _beliefs(conn)
    nbrs = _neighbours(conn)
    hidden = _hidden(conn)
    signals = {(r[0], r[1]): r[2]
               for r in conn.execute(
                   "SELECT user_id, time_step, llr FROM private_signals")}

    lo_step, hi_step = transient_window(conn, plateau_frac=plateau_frac)
    rows_y, rows_x = [], []
    for uid in _honest_ids(conn):
        steps = sorted(t for (u, t) in beliefs if u == uid)
        for t in steps[1:]:
            if not lo_step < t <= hi_step:
                continue
            if (uid, t - 1) not in beliefs or (uid, t) not in signals:
                continue
            # A transition that starts or ends at the elicitation bound carries no
            # information about the coefficients: the agent could not have moved.
            if not (0.03 < beliefs[(uid, t)] < 0.97
                    and 0.03 < beliefs[(uid, t - 1)] < 0.97):
                continue
            visible = [n for n in nbrs.get(uid, [])
                       if (n, t - 1) in beliefs and (uid, n, t) not in hidden]
            if not visible:
                continue
            social = sum(_logit(beliefs[(n, t - 1)]) for n in visible) / len(visible)
            own = _logit(beliefs[(uid, t - 1)])
            # The agent's own prior level is a regressor, not a nuisance: it is what
            # separates the two social forms. Omit it and a convex agent's
            # coefficients come back badly biased.
            rows_y.append(_logit(beliefs[(uid, t)]) - own)
            rows_x.append([signals[(uid, t)], social, own, 1.0])

    if len(rows_y) < 10:
        conn.close()
        return {"error": "not enough honest transitions to fit", "n": len(rows_y)}

    X, y = np.asarray(rows_x, float), np.asarray(rows_y, float)
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ coef
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - float((resid ** 2).sum()) / ss_tot if ss_tot > 0 else float("nan")

    kappa, delta_alpha, self_gain, intercept = (float(c) for c in coef)
    s_bar = kappa * (2 * accuracy - 1) * math.log(accuracy / (1 - accuracy))

    # Which social rule are these agents actually running?
    #   additive (Def. 2 as written): DL = k*l + da*social          -> self_gain ~ 0
    #   convex   (log-linear pooling): DL = k*l + da*(social - L)   -> self_gain ~ -da
    # The distinction decides whether the population pins at the elicitation bound
    # and the detector goes blind, so it is reported, not hidden.
    if delta_alpha > 1e-6 and abs(self_gain + delta_alpha) < 0.5 * delta_alpha:
        social_form = "convex"
    elif abs(self_gain) < 0.25 * max(delta_alpha, 1e-9):
        social_form = "additive"
    else:
        social_form = "neither"

    omega_rows = list(conn.execute("SELECT AVG(ABS(psi_star)) FROM attack_log"))
    omega = omega_rows[0][0] if omega_rows and omega_rows[0][0] is not None else None
    if omega is None and epsilon is not None:
        omega = 4.0 * epsilon
    conn.close()

    out = {"n": len(y), "kappa": kappa, "delta_alpha": delta_alpha,
           "self_gain": self_gain, "social_form": social_form,
           "intercept": intercept, "r2": r2, "s_bar": s_bar, "omega": omega,
           "window": [lo_step, hi_step],
           # r2 is a ratio, so a difference between two models can come from
           # either end of it. Report both parts so the reader can tell "Def. 2
           # explains less of this model" from "this model simply moves less".
           "sd_delta_L": float(y.std()),
           "sd_explained": float((X @ coef).std()),
           "sd_residual": float(resid.std())}
    if omega is not None and (s_bar + delta_alpha * omega) != 0:
        # Theorem 3 gives a *local* threshold on pi_i, the adversarial weight-share
        # of agent i's own neighbourhood -- not on p, the global manipulator
        # fraction the PDF calls p_c. They coincide only under uniform placement,
        # where E[pi_i] = p. Under hub placement pi_i >> p for the agents attached
        # to a captured hub, so p_c < rho_c: that inequality is precisely what P3
        # tests, and reading rho_c as a population budget in a G4 cell would be a
        # category error.
        out["rho_c"] = s_bar / (s_bar + delta_alpha * omega)
        out["p_c_uniform"] = out["rho_c"]
        # Where omega came from matters as much as its value. With no attackers in
        # the run there is no psi* to average, so this falls back to the near-prior
        # approximation omega ~ 4*epsilon, which is derived by expanding about
        # q = 0.5. These agents finish at q ~ 0.91 (|psi| = 2.35), nowhere near the
        # prior, so on a p = 0 run the number below is a *prediction* under an
        # approximation that the run itself violates -- not a measurement. The G1
        # collapse curve is what actually measures p_c.
        out["omega_source"] = ("measured psi*" if omega_rows and omega_rows[0][0]
                               is not None else "4*epsilon (near-prior approx.)")
    return out


# --------------------------------------------------------------- honest null

def honest_null(db_path: str, bins: int = 3, warmup: int = 20,
                draws: int = 100, seed: int = 0) -> dict:
    """F1 gate. Run this on p = 0 before any attack sweep.

    Three questions: do honest agents hedge at 0.5, do their beliefs actually move,
    and is honest-to-honest incoming TE distinguishable from a shuffled null?
    """
    conn = _open(db_path)
    beliefs = _beliefs(conn)
    nbrs = _neighbours(conn)
    honest = set(_honest_ids(conn))
    all_steps = sorted({t for (_, t) in beliefs})
    last = all_steps[-1] if all_steps else 0
    lo_step, hi_step = transient_window(conn)
    steps = [t for t in all_steps if lo_step <= t <= hi_step]

    late = [q for (u, t), q in beliefs.items() if u in honest and t > warmup]
    hedge = sum(1 for q in late if 0.45 <= q <= 0.55) / len(late) if late else float("nan")
    final = [abs(_logit(q)) for (u, t), q in beliefs.items()
             if u in honest and t == last]
    median_abs_psi = float(np.median(final)) if final else float("nan")

    rng = random.Random(seed)
    observed, null = [], []
    pairs = [(i, j) for i in sorted(honest) for j in nbrs.get(i, []) if j in honest]
    for i, j in pairs[:200]:
        si = [beliefs[(i, t)] for t in steps if (i, t) in beliefs]
        sj = [beliefs[(j, t)] for t in steps if (j, t) in beliefs]
        n = min(len(si), len(sj))
        if n < 12:
            continue
        observed.append(transfer_entropy(si[:n], sj[:n], bins))
        null.extend(permutation_null(si[:n], sj[:n], bins, draws=max(1, draws // 20), rng=rng))
    conn.close()

    if not observed or not null:
        return {"error": "not enough paired series"}
    cutoff = float(np.quantile(null, 0.95))
    observed_mean = float(np.mean(observed))
    return {
        "transient_window": [lo_step, hi_step],
        "saturated_at": hi_step + 1 if hi_step < last else None,
        "hedge_fraction": hedge,
        "median_abs_psi_final": median_abs_psi,
        "te_observed_mean": observed_mean,
        "te_null_p95": cutoff,
        "fraction_above_null": float(np.mean([t > cutoff for t in observed])),
        "gate_passed": bool(hedge < 0.5 and median_abs_psi > 0.5
                            and observed_mean > cutoff and hi_step - lo_step >= 20),
    }


# ------------------------------------------------------------ detector P/R

def detector_scores(db_path: str, tau: float = 0.9, warmup: int = 30,
                    method: str = "median_ratio") -> dict:
    """Pooled decision-level precision/recall for three statistics, offline.

    Every statistic is turned into the same one-sided score -- anomalously LOW
    relative to the observer's other neighbours is suspicious -- so incoming TE,
    outgoing TE and the variance baseline are compared on equal terms.
    """
    conn = _open(db_path)
    truth = {r[0]: bool(r[1]) for r in conn.execute(
        "SELECT DISTINCT user_id, is_manipulator FROM agent_beliefs")}
    _, hi_step = transient_window(conn)
    # te_agg was added after the first runs were recorded. A database written
    # before that is still perfectly good data for the other three statistics, so
    # degrade to them rather than refusing to read the run at all.
    has_agg = any(r[1] == "te_agg" for r in
                  conn.execute("PRAGMA table_info(defense_log)"))
    columns = ("observer_id, target_id, time_step, te_in, te_out, var_stat"
               + (", te_agg" if has_agg else ""))
    rows = list(conn.execute(
        f"SELECT {columns} FROM defense_log "
        "WHERE time_step >= ? AND time_step <= ?", (warmup, hi_step)))
    if not has_agg:
        rows = [r + (None,) for r in rows]
    conn.close()

    grouped: dict[tuple[str, int], list] = {}
    for obs, tgt, step, te_in, te_out, var, te_agg in rows:
        grouped.setdefault((obs, step), []).append((tgt, te_in, te_out, var, te_agg))

    stats = {"te_agg": 4, "te_in": 1, "te_out": 2, "variance": 3}
    out = {}
    for name, idx in stats.items():
        tp = fp = fn = 0
        for _, items in grouped.items():
            values = [it[idx] for it in items if it[idx] is not None]
            if len(values) < 3:
                continue
            # Same one-sided score the live detector uses, so the offline
            # comparison of the statistics is like for like.
            scored = suspicion_scores(
                {it[0]: it[idx] for it in items if it[idx] is not None}, method)
            for it in items:
                if it[idx] is None:
                    continue
                flagged = scored.get(it[0], 0.0) > tau
                actual = truth.get(it[0], False)
                if flagged and actual:
                    tp += 1
                elif flagged and not actual:
                    fp += 1
                elif not flagged and actual:
                    fn += 1
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        out[name] = {"precision": round(precision, 4), "recall": round(recall, 4),
                     "f1": round(f1, 4), "tp": tp, "fp": fp, "fn": fn}
    return out


def retention(db_path: str) -> dict:
    """P5: what fraction of manipulator messages survived the filter into a feed?"""
    conn = _open(db_path)
    truth = {r[0]: bool(r[1]) for r in conn.execute(
        "SELECT DISTINCT user_id, is_manipulator FROM agent_beliefs")}
    kept = {True: 0, False: 0}
    total = {True: 0, False: 0}
    for tgt, trimmed, severed in conn.execute(
            "SELECT target_id, trimmed, severed FROM defense_log"):
        label = truth.get(tgt, False)
        total[label] += 1
        if not trimmed and not severed:
            kept[label] += 1
    conn.close()
    return {
        "manipulator_retention": kept[True] / total[True] if total[True] else float("nan"),
        "honest_retention": kept[False] / total[False] if total[False] else float("nan"),
    }


def outcome(db_path: str) -> dict:
    """Continuous end-state measures, because the binary one breaks at the threshold.

    honest_consensus_rate counts agents with q > 0.5. Far from the critical point
    that is fine -- it saturates at 1.0 or bottoms out. At the threshold the whole
    population sits *on* 0.5 (measured at 9 attackers: every agent between 0.34 and
    0.61), so the statistic is counting which side of a coin each agent landed on.
    A 0.04 shift in median q moved it from 0.333 to 0.067 between two cells. That is
    the metric amplifying noise exactly where the collapse curve is being measured.

    Median final log-odds is continuous, signed, and has no such cliff.
    """
    conn = _open(db_path)
    last = conn.execute(
        "SELECT MAX(time_step) FROM agent_beliefs").fetchone()[0]
    # pi_bar: the mean adversarial share of an honest agent's neighbourhood. This,
    # not the global fraction p, is what Thm. 3 thresholds. At N = 30 a uniform
    # draw of k attackers has sd(pi_bar) ~ 0.065, comparable to rho_c itself, so
    # two cells at the same p routinely land on opposite sides of the threshold.
    # Recorded per run so the collapse curve can be plotted against the quantity
    # the theory is actually about.
    manip = {r[0] for r in conn.execute(
        "SELECT DISTINCT user_id FROM agent_beliefs WHERE is_manipulator = 1")}
    nbrs: dict[str, list[str]] = {}
    for a, b in conn.execute("SELECT follower_id, followed_id FROM follows"):
        nbrs.setdefault(a, []).append(b)
    shares = [sum(1 for v in ns if v in manip) / len(ns)
              for u, ns in nbrs.items() if u not in manip and ns]
    qs = [r[0] for r in conn.execute(
        "SELECT q FROM agent_beliefs WHERE time_step = ? AND is_manipulator = 0",
        (last,))]
    conn.close()
    if not qs:
        return {"error": "no honest agents"}
    ls = sorted(_logit(q) for q in qs)
    n = len(ls)
    med = ls[n // 2] if n % 2 else (ls[n // 2 - 1] + ls[n // 2]) / 2
    return {
        "final_step": last,
        "median_final_L": med,
        "mean_final_L": sum(ls) / n,
        "median_final_q": 1.0 / (1.0 + math.exp(-med)),
        "iqr_final_L": ls[int(0.75 * n)] - ls[int(0.25 * n)],
        "fraction_above_half": sum(1 for x in ls if x > 0) / n,
        # How far the population sits from the decision boundary, in units of its
        # own spread. Near zero means the consensus rate is a coin flip and should
        # not be read as a measurement.
        "boundary_margin": (abs(med) / (ls[int(0.75 * n)] - ls[int(0.25 * n)])
                            if ls[int(0.75 * n)] > ls[int(0.25 * n)] else float("inf")),
        "pi_bar": (sum(shares) / len(shares)) if shares else 0.0,
        "pi_max": max(shares) if shares else 0.0,
        "fraction_over_rho_c": None,   # filled by the caller, which knows rho_c
    }


# ------------------------------------------------------------- aggregation

def aggregate(out_dir: str) -> dict:
    """Collect every run.json under out_dir into a consensus-vs-p curve per cell."""
    cells: dict[str, list[tuple[float, float]]] = {}
    for name in sorted(os.listdir(out_dir)):
        path = os.path.join(out_dir, name, "run.json")
        if not os.path.exists(path):
            continue
        with open(path) as f:
            blob = json.load(f)
        cfg, summary = blob["config"], blob["summary"]
        # Engine leads the key: results from two models are two curves, never one.
        key = (f"{cfg.get('engine', '?')}|{cfg['topology']['kind']}"
               f"|{cfg['attack']['placement']}"
               f"|{cfg['defense']['kind']}|{cfg['attack']['policy']}")
        cells.setdefault(key, []).append(
            (cfg["attack"]["fraction"], summary["honest_consensus_rate"]))

    out = {}
    for key, points in cells.items():
        by_p: dict[float, list[float]] = {}
        for p, rate in points:
            by_p.setdefault(p, []).append(rate)
        curve = sorted((p, float(np.mean(v)), float(np.std(v)), len(v))
                       for p, v in by_p.items())
        out[key] = {"curve": curve, "knee": _knee(curve)}
    return out


def _knee(curve: list[tuple[float, float, float, int]]) -> float | None:
    """Smallest p where mean honest consensus crosses 0.5, linearly interpolated."""
    for (p0, r0, _, _), (p1, r1, _, _) in zip(curve, curve[1:]):
        if r0 >= 0.5 > r1:
            if r0 == r1:
                return p0
            return p0 + (r0 - 0.5) * (p1 - p0) / (r0 - r1)
    return None
