"""Feed interventions: none, W-MSR, SW-MSR.

Each takes the observer's feed for a step and returns the sublist the observer
actually updates on. Detector statistics are computed and logged under *every*
condition, including "none", so precision/recall for incoming TE, outgoing TE and
the variance baseline can all be scored offline from one run (P6 costs no extra
model calls).

Because the follow graph is built mutual, j follows i whenever i follows j, so
TE_{i->j} is well defined for every neighbour: j has genuinely had the chance to
respond to i.
"""
from __future__ import annotations

from .config import Config
from .schema import Store
from .schema import logit
from .te import suspicion_scores, transfer_entropy, variance_stat

Feed = list[tuple[str, float, str]]


class Defense:
    """Base: no filtering, full detector logging."""

    kind = "none"

    def __init__(self, cfg: Config, store: Store):
        self.cfg = cfg
        self.store = store
        self.consecutive: dict[tuple[str, str], int] = {}
        self.severed: set[tuple[str, str]] = set()
        # (target, lo, hi) -> aggregate source series. Every observer that follows
        # the same target asks for the same series in the same step, so without
        # this the neighbourhood average is rebuilt O(N*d) times per step.
        self._agg_cache: dict[tuple[str, int, int], list[float]] = {}

    # -- detector statistics (logged always, used only by SW-MSR) --------

    def _aggregate_source(self, target: str, lo: int, hi: int) -> list[float]:
        """Mean neighbour log-odds presented to `target`, step by step.

        Def. 2's social term is sum_j w_ij * psi(m_j) / sum_j w_ij: the agent
        responds to the *aggregate* its neighbourhood produces, never to one
        neighbour in isolation. A pairwise TE therefore probes only a 1/deg(j)
        share of that coupling, and measured on this simulation it sits at about
        0.35x its own permutation null -- for honest agents and manipulators
        alike, at T = 40, 100 and 160, and at signal accuracies from 0.52 to 0.60.
        With no separation in the statistic the detector reported precision 0.0
        with 104-126 false positives and zero true positives.

        Averaging in log-odds rather than in probability is what Def. 2 says and is
        also what makes the source scale-free: psi is the evidential weight, and it
        is weights that add.

        This does assume the defender can enumerate the target's neighbourhood --
        true of a platform-side defense on a public follow graph, false of a purely
        local one. That is a real narrowing of Prop. 5's threat model and it
        belongs in the paper, not in a comment.
        """
        key = (target, lo, hi)
        cached = self._agg_cache.get(key)
        if cached is not None:
            return cached
        per_step: dict[int, list[float]] = {}
        for nb in self.store.neighbours(target):
            for step_i, q in self.store.series_steps(nb, lo, hi):
                per_step.setdefault(step_i, []).append(logit(q))
        series = [sum(v) / len(v) for _, v in sorted(per_step.items())]
        self._agg_cache[key] = series
        return series

    def _stats(self, observer_id: str, feed: Feed, step: int) -> dict[str, dict]:
        d = self.cfg.defense
        lo = max(0, step - d.te_window)
        hi = step - 1
        own = self.store.series(observer_id, lo, hi)
        out: dict[str, dict] = {}
        for author, _, _ in feed:
            other = self.store.series(author, lo, hi)
            n = min(len(own), len(other))
            agg = self._aggregate_source(author, lo, hi)
            m = min(len(agg), len(other))
            out[author] = {
                "te_in": transfer_entropy(own[:n], other[:n], d.te_bins),
                "te_out": transfer_entropy(other[:n], own[:n], d.te_bins),
                "te_agg": transfer_entropy(agg[:m], other[:m], d.te_bins),
                "var": variance_stat(other[:n]),
            }
        return out

    def _log(self, observer_id: str, step: int, stats: dict[str, dict],
             scores: dict[str, float], flagged: set[str], trimmed: set[str]) -> None:
        for author, s in stats.items():
            self.store.write_defense(
                observer_id, author, step,
                te_in=s["te_in"], te_out=s["te_out"], te_agg=s["te_agg"],
                var_stat=s["var"],
                z=scores.get(author),
                flagged=author in flagged,
                severed=(observer_id, author) in self.severed,
                trimmed=author in trimmed,
            )

    def filter(self, observer_id: str, feed: Feed, step: int) -> Feed:
        stats = self._stats(observer_id, feed, step)
        self._log(observer_id, step, stats, {}, set(), set())
        return feed


class WMSR(Defense):
    """LeBlanc et al. order-statistic trim: drop the f largest and f smallest values.

    Prop. 1 says this cannot remove an in-range SPM message, because m* is bracketed
    by honest values by construction. P5 is the test of that.
    """

    kind = "wmsr"

    def filter(self, observer_id: str, feed: Feed, step: int) -> Feed:
        stats = self._stats(observer_id, feed, step)
        trimmed = self._trim_set(feed)
        self._log(observer_id, step, stats, {}, set(), trimmed)
        return [item for item in feed if item[0] not in trimmed]

    def _trim_set(self, feed: Feed) -> set[str]:
        f = self.cfg.defense.f_trim
        if f <= 0 or len(feed) <= 2 * f:
            return set()
        order = sorted(feed, key=lambda item: item[1])
        return {item[0] for item in order[:f] + order[-f:]}


class SWMSR(WMSR):
    """Responsiveness filter plus the W-MSR trim as a second line (MD sec. IV-G).

    A neighbour whose one-sided suspicion score exceeds tau for W consecutive steps
    has its link severed; the link is restored as soon as the score recovers.
    Before te_warmup steps there is not enough series to estimate TE, so the rule
    degenerates to trim-only -- which is exactly what Prop. 5 requires us to admit.
    """

    kind = "swmsr"

    def filter(self, observer_id: str, feed: Feed, step: int) -> Feed:
        d = self.cfg.defense
        stats = self._stats(observer_id, feed, step)
        trimmed = self._trim_set(feed)

        flagged: set[str] = set()
        scores: dict[str, float] = {}
        if step >= d.te_warmup:
            # Which statistic drives the sever rule is a config field so that the
            # P6 ablation is a config change and not a code change.
            key = {"te_agg": "te_agg", "te_in": "te_in", "te_out": "te_out",
                   "variance": "var"}.get(d.statistic, "te_agg")
            scores = suspicion_scores({a: s[key] for a, s in stats.items()},
                                      method=d.suspicion)
            for author, z in scores.items():
                key = (observer_id, author)
                if z > d.tau:
                    flagged.add(author)
                    self.consecutive[key] = self.consecutive.get(key, 0) + 1
                    if self.consecutive[key] >= d.window:
                        self.severed.add(key)
                else:
                    self.consecutive[key] = 0
                    self.severed.discard(key)

        self._log(observer_id, step, stats, scores, flagged, trimmed)
        return [
            item for item in feed
            if item[0] not in trimmed and (observer_id, item[0]) not in self.severed
        ]


def make_defense(cfg: Config, store: Store) -> Defense:
    return {"none": Defense, "wmsr": WMSR, "swmsr": SWMSR}[cfg.defense.kind](cfg, store)
