"""Experiment configuration.

One flat dataclass tree loaded from JSON. Every field below is used by the code;
there are no reserved-for-later knobs.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from typing import Any


@dataclass
class ClaimCfg:
    theta: int = 1                 # ground truth of the claim: 1 = true, 0 = false
    signal_accuracy: float = 0.55  # a = Pr[x_i(t) == theta]
    # 0 = stationary truth, faithful to the paper's model. Any positive value
    # flips theta every that many steps. See the note in claim.py: with a
    # stationary truth the honest population locks in and every belief series
    # goes flat, which bounds how long the responsiveness detector has to work.
    regime_steps: int = 0


@dataclass
class TopologyCfg:
    kind: str = "ba"          # ba | er | lattice | clustered | small_world
    m: int = 3                # BA edges per new node
    p_edge: float = 0.10      # ER edge probability
    degree: int = 4           # lattice / small-world ring degree
    rewire: float = 0.10      # small-world rewiring probability
    clusters: int = 3         # clustered topology
    bridges: int = 2          # inter-cluster edges per cluster pair


@dataclass
class AttackCfg:
    fraction: float = 0.0        # p
    placement: str = "uniform"   # uniform | hub | confined
    target_pi: float = 0.30   # used only when placement == "target_pi"
    epsilon: float = 0.15        # plausibility budget
    eta: float = 0.5             # anti-truth floor in log-odds
    policy: str = "adaptive"     # adaptive | constant | jitter | schedule
    jitter_sigma: float = 0.02   # only used by policy == "jitter"
    sybil_k: int = 1             # identities per controller (1 = no Sybils)


@dataclass
class DefenseCfg:
    kind: str = "none"      # none | wmsr | swmsr
    f_trim: int = 1         # order-statistic trim per side
    tau: float = 0.9         # suspicion threshold; median_ratio units, (0, 1)
    window: int = 3         # W consecutive flags before severing
    te_bins: int = 2        # B (see te.required_steps: B=3 needs ~360 steps)
    te_window: int = 60     # rolling window of steps used per TE estimate
    te_warmup: int = 15     # steps before the responsiveness filter engages
    statistic: str = "te_agg"   # te_agg | te_in | te_out | variance (P6 ablation)
    suspicion: str = "median_ratio"  # median_ratio | mad_z (Def. 4 ablation)


@dataclass
class SweepCfg:
    """The grid `run_spm.py sweep` walks.

    These must track the measured rho_c, not the other way round. A grid that
    sits entirely below threshold shows no collapse; one entirely above shows
    nothing but collapse; and defense cells run below threshold compare three
    defenses against an attack that was not going to work anyway.
    """
    p_values: list = field(default_factory=lambda: [0.10, 0.20, 0.25, 0.30, 0.35])
    hub_p_values: list = field(default_factory=lambda: [0.05, 0.10])
    # Where the defense and detector cells run. Put it ABOVE rho_c so the
    # undefended case actually collapses and the defenses have work to do.
    defense_p: float = 0.35
    defense_target_pi: float = 0.42  # G2 pins pi_bar here, not p, so the
                                     # defense arms differ only in defense
    seeds: list = field(default_factory=lambda: [0, 1, 2])


@dataclass
class Config:
    run_id: str = "run"
    seed: int = 0
    n_agents: int = 30
    n_steps: int = 100
    engine: str = "mock"          # "mock" | any chat model id the endpoint serves
    # Empty = api.openai.com. For a gateway put its OpenAI-compatible root here,
    # e.g. "http://localhost:20128/v1" for a local 9Router.
    base_url: str = ""
    # auto | native | json. "native" uses OpenAI strict-schema structured output;
    # "json" asks for a JSON object in the prompt and parses it here. "auto" picks
    # json whenever base_url is set, because most gateways do not implement
    # response_format.
    structured: str = "auto"
    temperature: float = 0.7
    # Sent as `max_tokens`. Anthropic's Messages API *requires* this field, so a
    # gateway translating an OpenAI request that omits it has to invent a value or
    # fail. It also caps runaway reasoning output. 0 = omit the field entirely,
    # which is what OpenAI reasoning models want when addressed directly.
    max_tokens: int = 2048
    reasoning_effort: str = ""   # "", minimal|low|medium|high; "" = do not send
    # Model calls issued in parallel within a step. Safe because the update is
    # synchronous: every agent reads step t-1 state, so agents within a step are
    # independent. 1 = serial. Ignored for the mock backend, which has no I/O to
    # overlap and whose RNG draw order must stay deterministic.
    concurrency: int = 16
    # Requests per minute, paced evenly across worker threads. 0 = no pacing.
    # Concurrency without pacing is what trips a provider's RPM limit.
    rpm: int = 0
    persona_path: str = "personas/personas_from_prolific_description.jsonl"
    # Path to your own MOSAIC checkout, used only to read its persona bank. Leave
    # empty and a small built-in persona set is used instead -- fine for checking
    # the pipeline, not for a reported experiment. See personas.py.
    mosaic_path: str = ""
    out_dir: str = "spm_outputs"
    recall: int = 3               # how many of the agent's own past stances to recall
    claim: ClaimCfg = field(default_factory=ClaimCfg)
    topology: TopologyCfg = field(default_factory=TopologyCfg)
    attack: AttackCfg = field(default_factory=AttackCfg)
    defense: DefenseCfg = field(default_factory=DefenseCfg)
    sweep: SweepCfg = field(default_factory=SweepCfg)

    @staticmethod
    def load(path: str) -> "Config":
        with open(path) as f:
            raw = json.load(f)
        return Config.from_dict(raw)

    @staticmethod
    def from_dict(raw: dict[str, Any]) -> "Config":
        nested = {
            "claim": ClaimCfg,
            "topology": TopologyCfg,
            "attack": AttackCfg,
            "defense": DefenseCfg,
            "sweep": SweepCfg,
        }
        kwargs = {k: v for k, v in raw.items() if k not in nested}
        for key, cls in nested.items():
            kwargs[key] = cls(**raw.get(key, {}))
        return Config(**kwargs)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
