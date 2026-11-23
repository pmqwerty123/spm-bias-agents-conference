# Release provenance and limitations

Prepared on 2026-09-12 from the local research workspace. No immutable historical
source revision is available for the archived runs. This is a snapshot of the
current implementation, not proof that every historical run used identical code.

## Packaging changes

- Copied only Python experiment modules, the CLI, tests, and mock configuration.
- Replaced historical README material with scoped release documentation.
- Created a blank-key, mock-default `.env.example` rather than copying secrets.
- Corrected the gateway test regex to match across newlines. Runtime algorithms
  were not changed; the correction only fixes error-message validation.
- Included the run-level CSV, not persona text, raw databases, PDFs, internal
  reports, or upstream MOSAIC source.
- Omitted historical helper launchers whose hard-coded run names and inherited
  provider settings could mislabel new runs. Use explicit CLI configurations.

## Scientific limitations retained in the implementation

- `wmsr` is symmetric rank trimming, not canonical W-MSR.
- The temporal estimator uses a nominal-neighborhood aggregate, rather than the
  exact filtered feed delivered to its target.
- With D(t) = q(t+1)-q(t), the statistic uses I(D(t+1); A(t) | D(t)); this includes
  an extra increment lag relative to the immediate response to A(t).
- Historical comments claiming zero TE for every open-loop process, guaranteed
  honest responsiveness, or universal thresholds must not be read as established
  guarantees for this estimator and feedback system.
- Attackers know honest-neighbor identities. Plausibility is numerical and local
  to the attacker's reference neighborhood, not proven textual or recipient stealth.
- Jitter and scheduled policies are not reprojected into the evolving plausibility
  interval. They are temporal stress tests, not always-feasible SPM policies.
- Honest prompts explicitly encourage responsiveness; a neutral-prompt ablation
  remains necessary for stronger interpretation.
- The mock executes a closed-form approximation and is not an LLM reproduction.
- The historical model identifier `gpt-5.6-luna` is a recorded engine string; the
  public deployed model identity still requires author verification.

Correcting the runtime detector requires separately versioned new experiments.
Do not relabel historical measurements as results from a corrected algorithm.

## Data interpretation

The 47-record CSV covers the selected primary-model cohort, not every run in the
working archive. The primary 18 runs comprise 13 uniform and 5 exposure-targeted
placements. Seven have negative final median honest log-odds. Results are
descriptive, not independent network-level causal or generalization estimates.

Some older manuscript examples differ from the released CSV for the eight-attacker
runs. Use the CSV values for this artifact and reconcile manuscript numbers before
submission rather than silently replacing or averaging conflicting values.