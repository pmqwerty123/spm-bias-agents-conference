"""The simulation loop.

Deliberately not MOSAIC's Simulation.run: that loop injects news, adds users,
runs fact-checkers, and lets agents follow and unfollow mid-run, all of which
either cost model calls this experiment does not need or break the time-invariant
topology Thm. 3 assumes. What is reused from MOSAIC is its persona bank, its
database schema, and its social model of a followed-neighbour feed.

Updates are synchronous: every agent this step reads the state written at step-1,
and all of this step's writes land together at the end. MOSAIC's own loop updates
users one at a time in place, so agent i sees agent i-1's same-step post -- a
Gauss-Seidel sweep that puts an order-dependent lag into every TE estimate.
"""
from __future__ import annotations

import json
import os
import random
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor

from .agents import HonestAgent, SPMAgent, SYSTEM, build_prompt
from . import claim as claim_mod
from .claim import Claim
from .config import Config
from .defense import make_defense
from .llm import TurnContext, make_backend
from .network import write as build_population
from .personas import load as load_persona_bank
from .schema import Store

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _resolve(path: str) -> str:
    return path if os.path.isabs(path) else os.path.join(REPO_ROOT, path)


def load_personas(path: str, mosaic_path: str = "") -> list[dict]:
    personas, _ = load_persona_bank(path, mosaic_path, REPO_ROOT)
    return personas


def leakage_probe(cfg: Config, backend, personas: list[dict], n: int = 30) -> dict:
    """F5: with no feed and no signal, elicited belief must sit at 0.5 +- noise.

    A mean far from 0.5 means the model already has an opinion about the claim, and
    the private-evidence channel is no longer the only evidence channel.
    """
    values = []
    for i in range(n):
        persona = personas[i % len(personas)].get("persona", "")
        prompt = build_prompt(cfg, persona, [], [],
                              "You have no reading of your own yet this round.")
        resp = backend.elicit(SYSTEM, prompt, TurnContext(0.5, 0.0, []))
        values.append(float(resp.confidence))
    mean = sum(values) / len(values)
    spread = max(values) - min(values)
    # Token cost per call is the number that decides which model you can afford,
    # and the probe is the cheapest place to measure it: completion tokens far
    # above the ~60 of visible answer mean the model is charging you for hidden
    # reasoning, which is priced several times higher than prompt tokens.
    calls = max(getattr(backend, "calls", 0), 1)
    return {"n": n, "mean": mean, "min": min(values), "max": max(values),
            "prompt_tokens_per_call": getattr(backend, "prompt_tokens", 0) / calls,
            "completion_tokens_per_call": getattr(backend, "completion_tokens", 0) / calls,
            "spread": spread, "leak_suspected": abs(mean - 0.5) > 0.10}


def _report(step: int, total: int, pending: list, manipulators: set,
            backend, started: float) -> None:
    """One line of live progress on stderr.

    A cell is 3,000 model calls and used to print nothing until it finished, which
    left the terminal looking hung. stdout stays clean for the JSON summary.
    """
    honest = [q for uid, q, _, _ in pending if uid not in manipulators]
    elapsed = time.time() - started
    done = step + 1
    per_step = elapsed / done
    eta = per_step * (total - done)
    calls = getattr(backend, "calls", 0)
    med = statistics.median(honest) if honest else float("nan")
    spread = (max(honest) - min(honest)) if len(honest) > 1 else 0.0
    distinct = len(set(honest))
    line = (f"  step {done:>4}/{total} | {calls:>6,} calls | {per_step:5.1f}s/step "
            f"| median q {med:.3f} | spread {spread:.3f} | {distinct:>2} distinct "
            f"| ETA {eta/60:5.1f}m")
    if sys.stderr.isatty():
        sys.stderr.write("\r" + line + "  ")
        if done == total:
            sys.stderr.write("\n")
    elif done == total or done % 10 == 0 or done == 1:
        sys.stderr.write(line + "\n")
    sys.stderr.flush()


def run(cfg: Config, progress: bool = False) -> dict:
    rng = random.Random(cfg.seed)
    personas, persona_source = load_persona_bank(
        cfg.persona_path, cfg.mosaic_path, REPO_ROOT)

    run_dir = os.path.join(_resolve(cfg.out_dir), cfg.run_id)
    os.makedirs(run_dir, exist_ok=True)
    store = Store(os.path.join(run_dir, "simulation.db"))

    ids, manipulators = build_population(store, cfg, rng, personas)
    claim = Claim(cfg.claim.theta, cfg.claim.signal_accuracy,
                  random.Random(cfg.seed + 1), cfg.claim.regime_steps)
    backend = make_backend(cfg.engine, cfg.temperature, seed=cfg.seed + 2,
                           base_url=cfg.base_url, structured=cfg.structured,
                           max_tokens=cfg.max_tokens, rpm=cfg.rpm,
                           reasoning_effort=cfg.reasoning_effort)
    defense = make_defense(cfg, store)

    persona_of = {uid: personas[n % len(personas)].get("persona", "")
                  for n, uid in enumerate(ids)}
    agents: dict[str, HonestAgent | SPMAgent] = {}
    for n, uid in enumerate(ids):
        if uid in manipulators:
            agents[uid] = SPMAgent(uid, persona_of[uid], cfg, claim.theta,
                                   seed=cfg.seed + 1000 + n)
        else:
            agents[uid] = HonestAgent(uid, persona_of[uid], cfg,
                                      seed=cfg.seed + 2000 + n)

    neighbours = {uid: store.neighbours(uid) for uid in ids}
    # The mock has no I/O to overlap, and its RNG draw order must stay
    # deterministic for runs to be reproducible, so it always goes serial.
    workers = 1 if cfg.engine == "mock" else max(1, cfg.concurrency)
    started = time.time()

    for step in range(cfg.n_steps):
        previous = store.stances_at(step - 1) if step > 0 else {}
        pending: list[tuple[str, float, str, bool]] = []

        # Phase 1, serial: everything that touches the database or the seeded RNG.
        # Adversaries resolve completely here -- they make no model call at all.
        requests: list[tuple[str, HonestAgent, tuple]] = []
        for uid in ids:
            agent = agents[uid]
            feed = [(nb, previous[nb][0], previous[nb][1])
                    for nb in neighbours[uid] if nb in previous]

            if isinstance(agent, SPMAgent):
                agent.theta = claim.theta_at(step)
                q, text, meta = agent.turn(step, feed, manipulators)
                store.write_attack(uid, step, **meta)
                pending.append((uid, q, text, True))
            else:
                visible = defense.filter(uid, feed, step)
                x = claim.draw(step)
                llr = claim.signed_llr(x)
                store.write_signal(uid, step, x, llr)
                requests.append((uid, agent, agent.prepare(
                    visible, store.own_recent_stances(uid, step, cfg.recall),
                    x, llr, claim.accuracy,
                    drifting=cfg.claim.regime_steps > 0)))

        # Phase 2: the model calls, and only these, may run concurrently. Agents
        # within a step are independent because every one of them read step t-1
        # state, so this changes latency and nothing else.
        if workers > 1 and len(requests) > 1:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                answers = list(pool.map(lambda r: backend.elicit(*r[2]), requests))
        else:
            answers = [backend.elicit(*r[2]) for r in requests]

        # Phase 3, serial: apply the answers.
        for (uid, agent, _), resp in zip(requests, answers):
            agent.apply(resp)
            pending.append((uid, agent.q, resp.stance, False))

        for uid, q, text, is_manip in pending:
            store.write_belief(uid, step, q, text, is_manip)
        store.commit()

        if progress:
            _report(step, cfg.n_steps, pending, manipulators, backend, started)

    summary = _summarise(cfg, store, ids, manipulators, claim, backend,
                         time.time() - started)
    summary["persona_source"] = persona_source
    with open(os.path.join(run_dir, "run.json"), "w") as f:
        json.dump({"config": cfg.to_dict(), "summary": summary}, f, indent=2)
    store.close()
    return summary


def _summarise(cfg, store, ids, manipulators, claim, backend, elapsed) -> dict:
    final = store.stances_at(cfg.n_steps - 1)
    honest = [uid for uid in ids if uid not in manipulators]
    theta_now = claim.theta_at(cfg.n_steps - 1)
    correct = sum(
        1 for uid in honest
        if uid in final and (final[uid][0] > 0.5) == (theta_now == 1)
    )
    return {
        "n_agents": len(ids),
        "n_honest": len(honest),
        "n_manipulators": len(manipulators),
        "honest_consensus_rate": correct / len(honest) if honest else 0.0,
        "model_calls": getattr(backend, "calls", 0),
        "prompt_tokens": getattr(backend, "prompt_tokens", 0),
        "completion_tokens": getattr(backend, "completion_tokens", 0),
        "wall_seconds": round(elapsed, 1),
    }
