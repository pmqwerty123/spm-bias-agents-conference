# Released run-level measurements

`run_summary.csv` contains 47 records from the selected historical primary-model
cohort. It was copied from the paper evidence export and contains numerical
run-level measurements and categorical settings, not agent text or persona rows.

Important columns:

- `run_id`, `seed`, `T`: historical identifier, simulation seed, and horizon.
- `k`, `p`: attacker count and realized global fraction.
- `placement`, `policy`, `defense`, `f`: historical configuration labels.
- `median_final_L`: median final honest log-odds; negative means weakly false-leaning
  median for the true-state-one experiments, not unanimous or confident error.
- `pi_bar`: average nominal local attacker-neighbor fraction over honest nodes.
- `trim_*`, `sever_*`: rates over logged observer-target-round opportunities;
  selectivity is attacker-removal rate minus honest-removal rate.
- `boundary_margin`: absolute median log-odds divided by its across-agent IQR;
  this is descriptive spread normalization, not a statistical confidence interval.
- `mean_psi`, `floor_fraction`: summaries of attacker reports and logged phases.

Blank values are missing/not-applicable, not zeros. `fraction_over_rho_c` is not
used in the release summary. `median_final_q` may reflect transformation of the
median log-odds; for even-sized groups, transformations of medians need not equal
the arithmetic median of transformed observations.

The summary script selects the 18 primary runs using attacker count > 0,
no defense, T=100, and non-hub placement. Its AUC is a within-sample pairwise
ranking statistic, with half credit for ties. Defense summaries use T=60 and
exposure-targeted placement, with sample standard deviations across run records.

Raw records are not provided here because redistribution of embedded personas and
model-generated content requires separate review. Therefore this artifact enables
summary-level reproduction, not independent raw-trajectory verification.