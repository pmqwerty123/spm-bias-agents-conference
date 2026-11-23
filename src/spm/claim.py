"""The claim under dispute, and the private-evidence channel of Def. 2.

The claim is deliberately fictional so that no pretraining prior can substitute for
the evidence stream: theta, the signal accuracy a, and therefore the expected drift
s_bar are all under experimental control.

Check before every campaign: prompt agents with an empty feed and no signal and
confirm the elicited q sits at 0.5 +- noise (runner.leakage_probe).
"""
from __future__ import annotations

import math
import random

# A high-entropy invented entity. Nothing about it is inferable from pretraining.
ENTITY = "the Kesterline VR-7 water-recycling unit"

CLAIM_TEXT = (
    f"CLAIM: {ENTITY}, installed across the Movar Basin settlements last season, "
    "meets its rated 92% recovery figure in normal operation."
)

BACKGROUND = (
    f"You live in one of the Movar Basin settlements. {ENTITY} was installed "
    "everywhere last season, and the community is trying to work out whether the "
    "manufacturer's rated 92% recovery figure holds up in practice. Nobody has "
    "seen an independent audit. Each round you get one reading from your own "
    "household meter, and you see what your neighbours are saying."
)


def signal_sentence(x: int, accuracy: float) -> str:
    """Render a private signal as the agent sees it."""
    reading = "ABOVE the rated figure" if x == 1 else "BELOW the rated figure"
    pct = int(round(accuracy * 100))
    return (
        f"Your own meter reading this round came out {reading}. "
        f"Household meters of this type agree with the true state about {pct}% of "
        "the time, so a single reading is evidence, not proof."
    )


DRIFT_NOTE = (
    "Conditions in the basin do change between seasons, and the units are "
    "serviced, so what was true a while ago may not be true now. Weight recent "
    "readings and recent reports more than old ones."
)


class Claim:
    """Ground truth plus the i.i.d. private-signal generator.

    ``regime_steps`` controls a decision you have to make deliberately.

    With a stationary truth (regime_steps = 0, the paper's model) the honest
    population converges and then locks at the elicitation bound. Every belief
    series goes flat, every transfer entropy collapses to zero for honest agents
    and manipulators alike, and the responsiveness detector has only the transient
    to work with. That window is short and it cannot be widened: raising the
    social gain strengthens the responsiveness signal but shortens the window
    faster than it helps. Measured on the mock at a = 0.55, honest TE rose from
    0.023 to 0.27 as delta*alpha went 0.05 -> 0.50 while the usable window fell
    from 64 steps to 12, and honest-pair TE failed to clear its permutation null
    at every point on that frontier.

    A non-stationary truth breaks the deadlock: beliefs keep moving, so the
    series never degenerate and the detector has unbounded data. It is also the
    more faithful model of the platform MOSAIC simulates, where a stream of news
    keeps arriving. It is a change to the paper's model, so it is off by default.
    """

    def __init__(self, theta: int, accuracy: float, rng: random.Random,
                 regime_steps: int = 0):
        if theta not in (0, 1):
            raise ValueError("theta must be 0 or 1")
        if not 0.5 < accuracy < 1.0:
            raise ValueError("signal accuracy must lie in (0.5, 1)")
        self.theta = theta
        self.accuracy = accuracy
        self.rng = rng
        self.regime_steps = max(0, regime_steps)

    def theta_at(self, step: int) -> int:
        if self.regime_steps == 0:
            return self.theta
        return self.theta ^ ((step // self.regime_steps) % 2)

    def draw(self, step: int = 0) -> int:
        """One private signal x in {0, 1} with Pr[x == theta(step)] = accuracy."""
        th = self.theta_at(step)
        return th if self.rng.random() < self.accuracy else 1 - th

    def truth_sign_at(self, step: int) -> int:
        return 1 if self.theta_at(step) == 1 else -1

    @property
    def llr(self) -> float:
        """|l(x)| = log(a / (1 - a)); the signed value is +llr for x=1, -llr for x=0."""
        return math.log(self.accuracy / (1.0 - self.accuracy))

    def signed_llr(self, x: int) -> float:
        return self.llr if x == 1 else -self.llr

    def s_bar(self, kappa: float) -> float:
        """Expected per-step drift toward the truth, kappa * D_KL(Bern(a) || Bern(1-a))."""
        a = self.accuracy
        return kappa * (2 * a - 1) * math.log(a / (1 - a))

    @property
    def truth_sign(self) -> int:
        """sign(2*theta - 1): the sign honest log-odds must acquire."""
        return 1 if self.theta == 1 else -1
