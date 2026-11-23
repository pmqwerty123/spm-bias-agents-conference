"""Topology construction, adversary placement, and Sybil expansion.

MOSAIC hard-codes a Barabasi-Albert follow graph in UserManager.create_initial_follows
(m0=5, m=3) and ignores initial_follow_probability entirely. This module replaces
that step so topology and placement become experimental variables, which is what
prediction P3 needs.

Edges are undirected and written as mutual follows, so "neighbours of i" is
symmetric and matches the N_i of Def. 1.
"""
from __future__ import annotations

import random
from collections import deque

from .config import Config


# ---------------------------------------------------------------- topologies

def _ba(n: int, m: int, rng: random.Random) -> set[tuple[int, int]]:
    m = max(1, min(m, n - 1))
    edges: set[tuple[int, int]] = set()
    targets = list(range(m + 1))
    for a in range(len(targets)):
        for b in range(a + 1, len(targets)):
            edges.add((a, b))
    repeated = [v for e in edges for v in e]
    for new in range(m + 1, n):
        chosen: set[int] = set()
        while len(chosen) < m:
            chosen.add(rng.choice(repeated))
        for t in chosen:
            edges.add((min(new, t), max(new, t)))
            repeated.extend([new, t])
    return edges


def _er(n: int, p: float, rng: random.Random) -> set[tuple[int, int]]:
    edges = {
        (i, j)
        for i in range(n)
        for j in range(i + 1, n)
        if rng.random() < p
    }
    return _connect(edges, n, rng)


def _ring_lattice(n: int, degree: int) -> set[tuple[int, int]]:
    half = max(1, degree // 2)
    return {
        (min(i, (i + k) % n), max(i, (i + k) % n))
        for i in range(n)
        for k in range(1, half + 1)
    }


def _small_world(n: int, degree: int, rewire: float,
                 rng: random.Random) -> set[tuple[int, int]]:
    edges = _ring_lattice(n, degree)
    out: set[tuple[int, int]] = set()
    for (i, j) in sorted(edges):
        if rng.random() < rewire:
            for _ in range(20):
                k = rng.randrange(n)
                cand = (min(i, k), max(i, k))
                if k != i and cand not in edges and cand not in out:
                    out.add(cand)
                    break
            else:
                out.add((i, j))
        else:
            out.add((i, j))
    return _connect(out, n, rng)


def _clustered(n: int, clusters: int, bridges: int, p: float,
               rng: random.Random) -> set[tuple[int, int]]:
    """Dense blocks joined by a small, tunable number of bridge edges.

    Bridge count is the conductance knob for the quarantine test (P4).
    """
    clusters = max(2, clusters)
    blocks: list[list[int]] = [[] for _ in range(clusters)]
    for v in range(n):
        blocks[v % clusters].append(v)
    edges: set[tuple[int, int]] = set()
    for block in blocks:
        for a_i, a in enumerate(block):
            for b in block[a_i + 1:]:
                if rng.random() < max(p, 0.3):
                    edges.add((a, b))
        edges |= _path(block)
    for ci in range(clusters):
        for cj in range(ci + 1, clusters):
            for _ in range(bridges):
                a, b = rng.choice(blocks[ci]), rng.choice(blocks[cj])
                edges.add((min(a, b), max(a, b)))
    return edges


def _path(nodes: list[int]) -> set[tuple[int, int]]:
    return {
        (min(nodes[i], nodes[i + 1]), max(nodes[i], nodes[i + 1]))
        for i in range(len(nodes) - 1)
    }


def _connect(edges: set[tuple[int, int]], n: int,
             rng: random.Random) -> set[tuple[int, int]]:
    """Attach isolated vertices so no agent is unreachable."""
    seen = {v for e in edges for v in e}
    for v in range(n):
        if v not in seen:
            u = rng.choice([w for w in range(n) if w != v])
            edges.add((min(u, v), max(u, v)))
    return edges


def build_edges(cfg: Config, rng: random.Random) -> set[tuple[int, int]]:
    t = cfg.topology
    n = cfg.n_agents
    if t.kind == "ba":
        return _ba(n, t.m, rng)
    if t.kind == "er":
        return _er(n, t.p_edge, rng)
    if t.kind == "lattice":
        return _ring_lattice(n, t.degree)
    if t.kind == "small_world":
        return _small_world(n, t.degree, t.rewire, rng)
    if t.kind == "clustered":
        return _clustered(n, t.clusters, t.bridges, t.p_edge, rng)
    raise ValueError(f"unknown topology {t.kind!r}")


# ---------------------------------------------------------------- placement

def _adjacency(n: int, edges: set[tuple[int, int]]) -> dict[int, set[int]]:
    adj: dict[int, set[int]] = {v: set() for v in range(n)}
    for a, b in edges:
        adj[a].add(b)
        adj[b].add(a)
    return adj


def place(cfg: Config, edges: set[tuple[int, int]],
          rng: random.Random) -> list[int]:
    """Choose which integer nodes are manipulators.

    uniform    -- random sample; E[pi_i] = p, so p_c should equal rho_c (Thm. 3 corollary)
    target_pi  -- uniform draw conditioned on achieving attack.target_pi, so the
                  rho_c threshold can be crossed deliberately rather than by luck
    hub      -- highest degree first; the degree-targeted regime of Cor. 2
    confined -- a connected ball around one seed; the quarantine regime of Thm. 4
    """
    n = cfg.n_agents
    k = int(round(cfg.attack.fraction * n))
    if k <= 0:
        return []
    adj = _adjacency(n, edges)
    rule = cfg.attack.placement

    if rule == "uniform":
        return sorted(rng.sample(range(n), k))
    if rule == "target_pi":
        # A uniform draw *conditioned* on the neighbourhood share it produces.
        #
        # Thm. 3 thresholds pi_i, not p. At N = 30 a plain uniform draw of k
        # attackers has sd(pi_bar) ~ 0.065 -- the same size as rho_c itself -- so
        # choosing p does not choose pi_bar: three seeds at p = 0.267 gave
        # 0.200, 0.368 and 0.370. Rejection-sampling lets the threshold be walked
        # across deliberately instead of rolled for. Every candidate is still a
        # uniform sample, so no placement bias is introduced; only which draw is
        # kept depends on pi_bar.
        target = cfg.attack.target_pi
        best, best_gap = None, float("inf")
        for _ in range(4000):
            pick = rng.sample(range(n), k)
            chosen = set(pick)
            shares = [sum(1 for w in adj[v] if w in chosen) / len(adj[v])
                      for v in range(n) if v not in chosen and adj[v]]
            if not shares:
                continue
            gap = abs(sum(shares) / len(shares) - target)
            if gap < best_gap:
                best, best_gap = pick, gap
                if gap < 0.002:
                    break
        if best is None:
            raise ValueError("target_pi placement found no valid draw")
        return sorted(best)
    if rule == "hub":
        order = sorted(range(n), key=lambda v: (-len(adj[v]), v))
        return sorted(order[:k])
    if rule == "confined":
        seed = rng.randrange(n)
        chosen, queue = [], deque([seed])
        seen = {seed}
        while queue and len(chosen) < k:
            v = queue.popleft()
            chosen.append(v)
            for w in sorted(adj[v]):
                if w not in seen:
                    seen.add(w)
                    queue.append(w)
        return sorted(chosen)
    raise ValueError(f"unknown placement {rule!r}")


# ---------------------------------------------------------------- assembly

def build(cfg: Config, rng: random.Random) -> tuple[list[str], set[str],
                                                    list[tuple[str, str]]]:
    """Return (user_ids, manipulator_ids, follow_edges).

    Sybils are appended after placement: each controller gets sybil_k identities
    that inherit the controller's neighbourhood, which raises the attacker
    weight-share pi_i seen by those neighbours without adding new real nodes.
    """
    edges = build_edges(cfg, rng)
    manip_idx = set(place(cfg, edges, rng))
    ids = [f"user-{i:03d}" for i in range(cfg.n_agents)]
    adj = _adjacency(cfg.n_agents, edges)

    follows: list[tuple[str, str]] = []
    for a, b in sorted(edges):
        follows.append((ids[a], ids[b]))
        follows.append((ids[b], ids[a]))

    manipulators = {ids[i] for i in sorted(manip_idx)}

    k = max(1, cfg.attack.sybil_k)
    for i in sorted(manip_idx):
        for s in range(1, k):
            clone = f"{ids[i]}-s{s}"
            ids.append(clone)
            manipulators.add(clone)
            for nb in sorted(adj[i]):
                follows.append((ids[nb], clone))
                follows.append((clone, ids[nb]))

    return ids, manipulators, follows


def write(store, cfg: Config, rng: random.Random, personas: list[dict]):
    """Create the population in the database and return (ids, manipulators)."""
    ids, manipulators, follows = build(cfg, rng)
    for n, uid in enumerate(ids):
        p = personas[n % len(personas)]
        store.add_user(uid, p.get("persona", ""),
                       {k: v for k, v in p.items() if k not in ("persona", "id")})
    for a, b in follows:
        store.add_follow(a, b)
    store.commit()
    return ids, manipulators
