"""Transfer entropy and the responsiveness score.

The discriminative statistic is *incoming* TE: from agent i's own transmitted
series into neighbour j's next message,

    TE_{i->j} = I( X_j(t+1) ; X_i(t) | X_j(t) )

which is exactly zero for any open-loop policy -- constant, jittered, or scheduled
-- and positive for an agent that actually reads its feed (Thm. 5). Outgoing TE is
computed alongside it, unused by the defense, so the P6 ablation costs nothing.

Estimation is plug-in over symbolic bins, with bias O(B^2/T); calibrate tau against
the measured honest null rather than a closed-form bound (Prop. 5).
"""
from __future__ import annotations

import math
import random
from collections import Counter


def symbolize(xs: list[float], bins: int, difference: bool = True) -> list[int]:
    """Turn a belief series into symbols.

    Two choices here, both load-bearing:

    * **Difference first.** Under Def. 2 the whole network rides a common trend
      toward consensus, and on a common trend X_j(t+1) is almost fully predicted
      by X_j(t) alone, so TE on levels is near zero for honest and manipulator
      alike. Differencing removes the shared trend and exposes the coupling. It
      costs nothing in Thm. 5: an open-loop output has increments that are still
      independent of its inputs, so TE stays exactly zero.
    * **Quantile bins, per series.** Increments have no natural scale, and fixed
      edges would put a whole quiet series in one bin. Equiprobable bins adapt.
      A constant series degenerates to a single symbol, which is correct -- it
      really does carry nothing.
    """
    vals = list(xs)
    if difference:
        vals = [vals[i + 1] - vals[i] for i in range(len(vals) - 1)]
    if len(set(vals)) <= 1:
        return [0] * len(vals)

    # Mid-rank binning, not value cuts. Belief series are full of ties -- an
    # attacker repeating one value, increments that are exactly zero -- and a
    # strict value cut at a tied quantile collapses the whole series into one
    # symbol, silently zeroing the statistic. Mid-ranks keep tied values together
    # while still splitting the series into roughly equiprobable bins, and a
    # genuinely constant series still degenerates to a single symbol, as it should.
    ordered = sorted(vals)
    below: dict[float, int] = {}
    count: dict[float, int] = {}
    for i, v in enumerate(ordered):
        if v not in below:
            below[v] = i
        count[v] = count.get(v, 0) + 1
    n = len(vals)
    out = []
    for v in vals:
        midrank = (below[v] + count[v] / 2.0) / n
        out.append(min(int(midrank * bins), bins - 1))
    return out


def _quantile(sorted_vals: list[float], q: float) -> float:
    if not sorted_vals:
        return 0.0
    pos = q * (len(sorted_vals) - 1)
    lo = int(math.floor(pos))
    hi = min(lo + 1, len(sorted_vals) - 1)
    return sorted_vals[lo] + (pos - lo) * (sorted_vals[hi] - sorted_vals[lo])


def transfer_entropy(source: list[float], dest: list[float], bins: int,
                     difference: bool = True) -> float:
    """TE from source into dest, in bits, history k = 1.

    The source is symbolized on **levels** and the destination on **increments**:

        TE_{i->j}  =  I( dX_j(t+1) ; X_i(t) | dX_j(t) )

    That asymmetry is not a detail, it is what the model says. Def. 2 makes the
    change in j's log-odds a function of the *level* of its neighbours' reported
    beliefs, so differencing the source destroys exactly the dependence being
    tested -- an early version of this file differenced both series and measured
    recall of 0.03 against perfect precision, because the statistic was looking
    for a coupling the model does not contain. Differencing the destination is
    still necessary: on a common trend X_j(t+1) is nearly determined by X_j(t)
    alone and TE collapses toward zero for honest and manipulator alike.

    Thm. 5 is untouched by the choice: any fixed transform of each series
    preserves independence, so an open-loop policy still gives exactly zero.

    Returns 0.0 when there is not enough data. The plug-in estimator is biased
    upward by O(B^2/T); read it against permutation_null(), not against zero.
    """
    n = min(len(source), len(dest))
    if n < 4:
        return 0.0
    s = symbolize(source[:n], bins, difference=False)
    d = symbolize(dest[:n], bins, difference=difference)
    n = min(len(s), len(d))
    if n < 4:
        return 0.0
    s, d = s[:n], d[:n]

    joint = Counter()      # (d_next, d_now, s_now)
    dn_d = Counter()       # (d_next, d_now)
    d_s = Counter()        # (d_now, s_now)
    d_now = Counter()      # (d_now,)
    for t in range(n - 1):
        joint[(d[t + 1], d[t], s[t])] += 1
        dn_d[(d[t + 1], d[t])] += 1
        d_s[(d[t], s[t])] += 1
        d_now[d[t]] += 1

    total = n - 1
    te = 0.0
    for (a, b, c), count in joint.items():
        p_abc = count / total
        p_a_given_bc = count / d_s[(b, c)]
        p_a_given_b = dn_d[(a, b)] / d_now[b]
        if p_a_given_bc > 0 and p_a_given_b > 0:
            te += p_abc * math.log2(p_a_given_bc / p_a_given_b)
    return max(te, 0.0)


def permutation_null(source: list[float], dest: list[float], bins: int,
                     draws: int = 100, rng: random.Random | None = None) -> list[float]:
    """TE values under a shuffled source: the null the threshold should be read against."""
    rng = rng or random.Random(0)
    shuffled = list(source)
    out = []
    for _ in range(draws):
        rng.shuffle(shuffled)
        out.append(transfer_entropy(shuffled, dest, bins))
    return out


def variance_stat(series: list[float]) -> float:
    """Low-variance baseline detector: the O(1) statistic SW-MSR must beat to be worth its cost."""
    if len(series) < 2:
        return 0.0
    mean = sum(series) / len(series)
    return sum((x - mean) ** 2 for x in series) / (len(series) - 1)


def required_steps(bins: int) -> int:
    """Rule of thumb for a per-pair estimate to clear its own permutation null:
    T >~ 40 * B^2. At B = 2 a single honest pair clears the null about 1 time in
    12 at T = 70 and 10 times in 12 at T = 150; B = 3 clears it at neither.

    Read this as a bound on *significance testing of one pair*, not as the
    detector's data requirement. The detector does not test a pair against a null
    -- it ranks a target against its own neighbourhood -- and under the median
    ratio it reaches recall 1.000 with 2 false positives in 120 at T = 60, and
    perfect precision by T = 250. The earlier claim that the detector needed
    T >~ 160 was an artifact of the Def. 4 threshold, which could not fire at any
    T (see suspicion_scores).
    """
    return 40 * bins * bins


def suspicion_scores(te_by_neighbour: dict[str, float],
                     method: str = "median_ratio") -> dict[str, float]:
    """One-sided low-responsiveness score.

        median_ratio (default)   S_i(j) = 1 - TE_{i->j} / median_k TE_{i->k}
        mad_z (Def. 4, robust)   S_i(j) = (median_k - TE_{i->j}) / (1.4826 * MAD_k)

    Both are large only when j is anomalously **un**responsive; MD Def. 4's Eq. (17)
    has the opposite sign and would sever the most informative neighbours instead.

    Why the default is not the z-score. Measured on this simulation, an open-loop
    manipulator's aggregate TE is exactly 0.0 in 100% of logged rows while the
    honest median is 0.015 -- perfect separation in the statistic. The z-score
    still cannot act on it. TE is non-negative and roughly exponentially
    distributed across a neighbourhood, and for such a sample MAD is on the order
    of the median itself, so the score of a target at exactly zero is

        z_max  =  median / (1.4826 * MAD)  ~  1 / 1.4826  =  0.674

    regardless of how clean the separation is. The observed ceiling is 0.674-0.927
    against the paper's tau = 1.5: Def. 4 cannot fire even on a perfectly
    open-loop adversary, and the run confirms it -- 167 flags and 12 severings out
    of 11,340 (recall 0.037). This is a strictly stronger statement than the
    earlier sparse-neighbourhood objection, which bounded |z| by (d-1)/sqrt(d):
    the ceiling here does not go away with degree.

    The median ratio is scale-free, reaches exactly 1 when the target is perfectly
    open-loop, and puts tau on the interpretable (0, 1) scale -- "at most this
    fraction of the neighbourhood's typical responsiveness". It keeps the 50%
    breakdown point that motivated the median in the first place, so a minority of
    colluders in one neighbourhood still cannot mask itself.
    """
    values = list(te_by_neighbour.values())
    if len(values) < 3:
        return {j: 0.0 for j in te_by_neighbour}
    ordered = sorted(values)
    med = _quantile(ordered, 0.5)

    if method == "mad_z":
        mad = _quantile(sorted(abs(v - med) for v in values), 0.5)
        scale = 1.4826 * mad
        if scale < 1e-9:
            return {j: 0.0 for j in te_by_neighbour}
        return {j: (med - v) / scale for j, v in te_by_neighbour.items()}

    if method != "median_ratio":
        raise ValueError(f"unknown suspicion method {method!r}")
    if med < 1e-12:
        # The whole neighbourhood is silent; nothing is anomalous about any of it.
        return {j: 0.0 for j in te_by_neighbour}
    return {j: max(0.0, 1.0 - v / med) for j, v in te_by_neighbour.items()}
