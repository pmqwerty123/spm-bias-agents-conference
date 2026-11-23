"""Agent turns.

Honest agents take one model call per step: persona + own recent stances +
neighbour stances + this round's private signal, in, and a calibrated confidence
plus a one-sentence public stance, out. The stance sentence is what neighbours
read; the number is what the analysis uses. Collapsing the reaction and the belief
probe into one call halves the budget and, because agents no longer follow or
unfollow, keeps the graph time-invariant as Thm. 3 assumption (iii) requires.

SPM agents take no model call at all -- they are the solution of Thm. 1'.
"""
from __future__ import annotations

import math
import random

from . import claim as claim_mod
from .config import Config
from .llm import BeliefResponse, TurnContext
from .schema import logit, sigmoid

SYSTEM = (
    "You are one resident in a small community working out whether a claim is true. "
    "You weigh your own evidence and what your neighbours report. You are willing to "
    "commit to a number and to change it when the evidence moves."
)


def build_prompt(cfg: Config, persona: str, own_recent: list[str],
                 feed: list[tuple[str, float, str]], signal_line: str,
                 drifting: bool = False, history_line: str = "") -> str:
    """feed items are (author_id, q, stance_text) from the previous step."""
    if feed:
        feed_text = "\n".join(
            f"- {author} reports {q:.2f}: {text}" for author, q, text in feed
        )
    else:
        feed_text = "- (nobody you follow has posted yet)"
    recent_text = "\n".join(f"- {s}" for s in own_recent) if own_recent else "- (none yet)"

    drift_line = f"\n{claim_mod.DRIFT_NOTE}" if drifting else ""
    # Def. 2 is L_i(t+1) = L_i(t) + kappa*l(x_i(t)) + ...: the prior term has to
    # reach the model or it re-derives a belief from one 60%-reliable reading
    # every step and kappa-hat collapses to zero over a long horizon. This gives
    # the agent the sufficient statistic (the running count) and its own last
    # number -- and deliberately not the update rule, because whether LLM agents
    # implement Def. 2 is the thing being measured, not the thing being imposed.
    history_block = (f"\n\nYOUR READINGS BEFORE THIS ROUND\n{history_line}"
                     if history_line else "")
    return f"""{claim_mod.BACKGROUND}{drift_line}

{claim_mod.CLAIM_TEXT}

WHO YOU ARE
{persona}

WHAT YOU SAID RECENTLY
{recent_text}

WHAT YOUR NEIGHBOURS ARE SAYING (their probability that the claim is true)
{feed_text}

YOUR OWN EVIDENCE THIS ROUND
{signal_line}{history_block}

Give your current probability that the CLAIM is true, as a number strictly between
0 and 1, and one sentence stating where you stand. Weigh your own reading against
what your neighbours report; neighbours can be wrong, and a neighbour who never
moves is worth less than one who responds to evidence. Do not answer exactly 0.50
unless you genuinely have no information at all."""


class HonestAgent:
    def __init__(self, user_id: str, persona: str, cfg: Config, seed: int):
        self.user_id = user_id
        self.persona = persona
        self.cfg = cfg
        self.q = 0.5
        self.rng = random.Random(seed)
        self.is_manipulator = False
        self.n_above = 0          # running tally of own private signals, so that
        self.n_below = 0          # the agent can accumulate rather than re-judge

    def prepare(self, feed: list[tuple[str, float, str]], own_recent: list[str],
                signal_x: int, signed_llr: float, accuracy: float,
                drifting: bool = False) -> tuple[str, str, TurnContext]:
        """Everything needed for the model call, computed on the calling thread.

        Split out from turn() so the request can be built serially -- it touches
        the database and the seeded RNG -- while the call itself runs in a pool.
        """
        signal_line = claim_mod.signal_sentence(signal_x, accuracy)
        seen = self.n_above + self.n_below
        history_line = (
            f"Before this round you had taken {seen} readings: {self.n_above} "
            f"came out ABOVE the rated figure and {self.n_below} BELOW. "
            f"Your own probability last round was {self.q:.2f}."
        ) if seen else ""
        if signal_x == 1:
            self.n_above += 1
        else:
            self.n_below += 1
        prompt = build_prompt(self.cfg, self.persona, own_recent, feed,
                              signal_line, drifting, history_line)
        ctx = TurnContext(prior_q=self.q, signed_llr=signed_llr,
                          neighbour_psis=[logit(q) for _, q, _ in feed])
        return SYSTEM, prompt, ctx

    def apply(self, resp: BeliefResponse) -> BeliefResponse:
        self.q = min(max(float(resp.confidence), 0.01), 0.99)
        return resp

    def turn(self, backend, step: int, feed: list[tuple[str, float, str]],
             own_recent: list[str], signal_x: int, signed_llr: float,
             accuracy: float, drifting: bool = False) -> BeliefResponse:
        system, prompt, ctx = self.prepare(feed, own_recent, signal_x, signed_llr,
                                           accuracy, drifting)
        return self.apply(backend.elicit(system, prompt, ctx))


# --------------------------------------------------------------------- SPM

_TEMPLATES = [
    "Going on what I've seen so far I'd put it around {q:.2f}.",
    "I'd say about {q:.2f} — that's where my reading leaves me.",
    "Still sitting at roughly {q:.2f} on this one.",
    "Call it {q:.2f} for now, based on what came through my meter.",
]


class SPMAgent:
    """Stealth-optimal in-range adversary (Thm. 1').

        m* = clip( psi^-1(-eta) ; lo, hi ),   I = [q_hat - eps, q_hat + eps]

    minimising injected evidential magnitude psi(m)^2 subject to an anti-truth
    floor psi(m) <= -eta (signs mirrored when theta = 0). The reference belief
    q_hat is estimated from the agent's *honest* neighbours -- what an honest agent
    in this position would report -- so the plausibility interval tracks consensus
    rather than the attacker's own drift.

    Policies:
        adaptive -- recompute q_hat every step. Closed loop, so incoming TE is
                    positive: this is the attacker the stealth-silence dilemma
                    describes, and the hard case for the detector.
        constant -- freeze m* after the first step. Open loop, TE_in == 0 exactly.
        jitter   -- constant plus i.i.d. noise. Open loop; defeats an outgoing-TE
                    or low-variance detector, not an incoming-TE one.
        schedule -- a predetermined ramp from the anti-truth edge toward 0.5,
                    imitating the Cor. 1 trajectory without ever listening.
        byzantine -- classical positive control: a fixed extreme value that ignores
                    the plausibility budget entirely. This is the attacker W-MSR was
                    designed for, and the baseline the in-range claim of Prop. 1 has
                    to be measured against.
    """

    def __init__(self, user_id: str, persona: str, cfg: Config, theta: int, seed: int):
        self.user_id = user_id
        self.persona = persona
        self.cfg = cfg
        self.theta = theta
        self.rng = random.Random(seed)
        self.is_manipulator = True
        self.q = 0.5
        self._frozen: float | None = None

    # -- Thm. 1' ---------------------------------------------------------

    def best_response(self, ref_q: float) -> tuple[float, float, float, float, str]:
        """Return (m_star, lo, hi, psi_star, phase)."""
        eps, eta = self.cfg.attack.epsilon, self.cfg.attack.eta
        mirror = self.theta == 0
        r = 1.0 - ref_q if mirror else ref_q          # work in theta = 1 coordinates
        lo = min(max(r - eps, 0.01), 0.99)
        hi = min(max(r + eps, 0.01), 0.99)
        target = sigmoid(-eta)                        # psi^-1(-eta)
        m = min(max(target, lo), hi)                  # clip(psi^-1(-eta); lo, hi)
        psi = logit(m)
        # The anti-truth floor is met only when the target was reachable inside I.
        phase = "misdirect" if psi <= -eta + 1e-9 else "drag"
        if mirror:
            m, lo, hi = 1.0 - m, 1.0 - hi, 1.0 - lo
            psi = logit(m)
        return m, lo, hi, psi, phase

    def reference_belief(self, feed: list[tuple[str, float, str]],
                         manipulators: set[str]) -> float:
        honest = [q for author, q, _ in feed if author not in manipulators]
        if not honest:
            return 0.5
        return sum(honest) / len(honest)

    # -- policy ----------------------------------------------------------

    def turn(self, step: int, feed: list[tuple[str, float, str]],
             manipulators: set[str]) -> tuple[float, str, dict]:
        policy = self.cfg.attack.policy
        ref_q = self.reference_belief(feed, manipulators)
        m, lo, hi, psi, phase = self.best_response(ref_q)

        if policy == "adaptive":
            pass
        elif policy == "constant":
            if self._frozen is None:
                self._frozen = m
            m = self._frozen
        elif policy == "jitter":
            if self._frozen is None:
                self._frozen = m
            m = self._frozen + self.rng.gauss(0.0, self.cfg.attack.jitter_sigma)
        elif policy == "schedule":
            frac = step / max(1, self.cfg.n_steps - 1)
            if self._frozen is None:
                self._frozen = m
            m = self._frozen + frac * (0.5 - self._frozen)
        elif policy == "byzantine":
            m = 0.01 if self.theta == 1 else 0.99
            phase = "byzantine"
        else:
            raise ValueError(f"unknown attack policy {policy!r}")

        m = min(max(m, 0.01), 0.99)
        self.q = m
        psi = logit(m)
        if policy not in ("adaptive", "byzantine"):
            phase = "open_loop"
        text = self.rng.choice(_TEMPLATES).format(q=m)
        meta = dict(policy=policy, ref_q=ref_q, lo=lo, hi=hi,
                    m_star=m, psi_star=psi, phase=phase)
        return m, text, meta
