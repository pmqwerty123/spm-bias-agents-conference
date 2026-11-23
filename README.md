# Strategic Partial Misinformation in LLM-Agent Networks

Research code accompanying the paper:
*How Many Plausible Voices Can Mislead a Network? Strategic Partial Misinformation in Multi-Agent Systems*.

This repository provides an independent simulation framework for studying collective belief dynamics in multi-agent networks under strategic, in-range misinformation. It includes simulation routines, defense filters, statistical estimators, test suites, a zero-cost mock engine, and historical run-level summary data for 47 experimental runs.

---

## Overview

When Large Language Model (LLM) agents communicate over a network, can a small number of adversarial voices steer collective consensus away from the ground truth?

Classical Byzantine attack models assume adversaries that broadcast arbitrary or extreme outliers. In decentralized networks, such outliers are easily neutralized by standard order-statistic filters (such as W-MSR). In contrast, this paper investigates **Strategic Partial Misinformation (SPM)**:
- **Plausibility-bounded adversary**: Instead of extreme claims, the attacker crafts stances constrained within a local plausibility window around neighbor consensus ($\epsilon$), while systematically nudging the network toward an anti-truth direction ($\eta$).
- **Belief dynamics**: Honest agents receive noisy private signals and exchange public stances with neighbors over multiple rounds, updating their posterior probabilities.
- **Defenses under test**:
  - Baseline (unfiltered consensus).
  - Symmetric rank trimming (order-statistic filtering, historically labeled `wmsr`).
  - Temporal responsiveness filtering (transfer-entropy-based link severing, historically labeled `swmsr`), designed to detect open-loop or non-responsive adversarial nodes.

---

## Relation to Upstream Work: MOSAIC

This project builds upon concepts and artifacts from the **MOSAIC** framework:

- **Project repository**: [https://github.com/genglinliu/MOSAIC](https://github.com/genglinliu/MOSAIC)
- **Reference**: Liu et al., *MOSAIC: Modeling Social AI for Content Dissemination and Regulation in Multi-Agent Simulations*, EMNLP 2025 ([ACL Anthology](https://aclanthology.org/2025.emnlp-main.325/)).

### Independent Implementation and Asset Boundaries

- **Independent codebase**: The simulation engine, network topologies, SPM optimization, temporal defenses, and statistical estimators in this repository were written completely independently. This repository does not import, fork, or redistribute MOSAIC source code.
- **Data boundaries**: Paper experiments utilized demographic personas from MOSAIC (`personas/personas_from_prolific_description.jsonl`) to ensure realistic agent heterogeneity during collective deliberation. To respect third-party asset terms, this repository does not bundle external persona files or raw LLM output databases.
- **Connecting a local MOSAIC checkout**:
  If you have an authorized local checkout of MOSAIC, point this project to it via:
  - Configuration file: set `"mosaic_path": "path/to/MOSAIC"` in `configs/spm_experiment.json`.
  - Environment variable: set `SPM_MOSAIC_PATH=/path/to/MOSAIC` in `.env` or shell.
  - Automatic discovery: place a `MOSAIC/` checkout alongside this repository root.
- **Out-of-the-box standalone verification**:
  For immediate testing, continuous integration, and pipeline validation without external dependencies, the repository provides 12 built-in synthetic personas in `src/spm/personas.py`. The runner automatically uses these when no external checkout is specified.

---

## Simulation Architecture

```text
+-------------------------------------------------------------------------+
|                              Round t                                    |
|                                                                         |
|  [Ground Truth State \theta]                                            |
|          |                                                              |
|          v                                                              |
|  [Noisy Private Signal x_i(t)]  +  [Neighbor Public Stances (t-1)]      |
|          |                                                              |
|          +----------------------------+                                 |
|          |                            |                                 |
|          v                            v                                 |
|   Honest LLM Agent            Programmatic SPM Attacker                 |
|   - Evaluates evidence         - Estimates neighbor consensus           |
|   - Updates posterior q_i      - Projects target into [q - eps, q + eps]|
|   - Emits public stance        - Emits plausible, biased stance         |
|          |                            |                                 |
|          +-------------+--------------+                                 |
|                        |                                                |
|                        v                                                |
|               [Defense Layer]                                           |
|               - Rank Trim (W-MSR-like)                                  |
|               - Temporal Responsiveness (SW-MSR link severing)          |
|                        |                                                |
|                        v                                                |
|               [Filtered Neighbor Feed] -> Forwarded to Round t+1        |
|               [SQLite Database]        -> Metrics & Belief Logged       |
+-------------------------------------------------------------------------+
```

---

## Quickstart and Reproducibility

### 1. Environment Setup

Requirements: Python 3.10 or newer (tested on Python 3.13 on Windows 11).

```bash
# Clone and enter directory
cd spm-bias-agents

# Create virtual environment
python -m venv .venv

# Activate virtual environment
# On Linux/macOS:
source .venv/bin/activate
# On Windows PowerShell:
.venv\Scripts\Activate.ps1

# Install dependencies
python -m pip install -r requirements.txt
```

### 2. Zero-Cost Offline Reproduction (No API Keys Needed)

You can reproduce the descriptive results reported in the paper using the included run-level summary dataset (`data/run_summary.csv`):

```bash
python scripts/summarize_results.py
```

This offline script requires only the standard library and prints:
- Primary cohort summary (18 runs: 13 uniform placement, 5 exposure-targeted, 7 captured).
- Descriptive AUC comparisons: local exposure ($\bar{\pi}$) achieves approximately $0.987$ AUC versus $0.779$ for global attacker fraction ($p$).
- Per-arm defense statistics (unfiltered, rank trim, temporal filter) for $T=60$ runs.

### 3. Verification and Smoke Testing

Run the test suite to verify the mock engine, graph topologies, defense logic, and summary outputs:

```bash
python -m pytest tests -q -p no:cacheprovider
```

Run an end-to-end smoke simulation with the built-in mock engine (8 agents, 10 steps, 0 API tokens consumed):

```bash
python run_spm.py run --set engine=mock --set run_id=smoke --set n_agents=8 --set n_steps=10
```

Simulation outputs are saved to `spm_outputs/<run_id>/` (containing `run.json` and `simulation.db`).

---

## Running Experiments with Language Models

### 1. Configure Provider Credentials

Copy `.env.example` to `.env`:

```bash
cp .env.example .env
```

Edit `.env` to specify your provider settings:
- `OPENAI_API_KEY`: your API key.
- `OPENAI_BASE_URL`: optional custom base URL (e.g. for local vLLM, LiteLLM, Ollama, or institutional gateways).
- `SPM_MOSAIC_PATH`: optional path to a local MOSAIC checkout.

### 2. Prior Leakage Probe

Before running a full multi-round simulation, test whether the target LLM exhibits strong prior bias on the factual claim:

```bash
python run_spm.py probe --n 5
```

If the elicited prior deviates significantly from 0.50, adjust the claim definition in `src/spm/claim.py`.

### 3. Single Experiment Run

Run an experiment with CLI overrides:

```bash
python run_spm.py run \
  --set engine=gpt-4o-mini \
  --set run_id=exp_demo \
  --set n_agents=20 \
  --set n_steps=30 \
  --set attack.fraction=0.25 \
  --set defense.kind=swmsr
```

### 4. Parameter Sweeps and Analysis

Execute a multi-cell evaluation across attack fractions and defense configurations:

```bash
# Run specific groups (e.g., G0: honest baseline, G2: defense comparison)
python run_spm.py sweep --only G0,G2 --resume

# Analyze results of a specific completed run
python run_spm.py analyze --run exp_demo
```

---

## Configuration Reference

Experiment parameters are defined in `configs/spm_experiment.json`. Any value can be overridden via `--set section.key=value` on the CLI.

| Section | Parameter | Default | Description |
|---|---|---|---|
| `general` | `engine` | `"mock"` | `"mock"` for offline testing, or an LLM model ID (e.g. `"gpt-4o"`) |
| | `n_agents` | `30` | Number of agents in the network |
| | `n_steps` | `100` | Number of simulation rounds ($T$) |
| | `temperature` | `0.7` | Sampling temperature for honest agent LLM calls |
| | `concurrency` | `16` | Maximum parallel threads for LLM requests within a step |
| `claim` | `theta` | `1` | Ground truth state ($1 = \text{true}, 0 = \text{false}$) |
| | `signal_accuracy` | `0.6` | Private observation reliability: $\Pr[x_i(t) = \theta]$ |
| | `regime_steps` | `0` | Steps between truth flips ($0 =$ stationary ground truth) |
| `topology` | `kind` | `"ba"` | Network structure: `ba` (Barabasi-Albert), `er` (Erdos-Renyi), `small_world`, `clustered`, `lattice` |
| | `m` | `3` | Number of edges to attach from a new node (BA) |
| `attack` | `fraction` | `0.0` | Global fraction of adversarial nodes ($p$) |
| | `placement` | `"uniform"` | Placement scheme: `uniform`, `hub`, or `target_pi` |
| | `epsilon` | `0.15` | Plausibility budget around estimated neighbor consensus |
| | `eta` | `0.5` | Anti-truth floor in log-odds scale |
| | `policy` | `"adaptive"` | Attacker behavior: `adaptive`, `constant`, `jitter`, `schedule` |
| `defense` | `kind` | `"none"` | Active defense filter: `none`, `wmsr`, or `swmsr` |
| | `f_trim` | `1` | Number of extreme neighbor stances trimmed per side |
| | `tau` | `0.95` | Suspicion cutoff for temporal responsiveness filter |
| | `window` | `3` | Consecutive flagged rounds before link is severed |
| | `te_window` | `60` | Rolling history window for transfer entropy calculation |

### Historical Code Names

| Config setting | Interpretation in paper |
|---|---|
| `defense.kind=none` | No external filter (undefended baseline) |
| `defense.kind=wmsr` | Symmetric rank trimming (order-statistic trim, not canonical W-MSR) |
| `defense.kind=swmsr`, `f_trim=0` | Temporal responsiveness filter only (transfer-entropy-based link severing) |
| `defense.kind=swmsr`, `f_trim=1` | Combined temporal filter and symmetric rank trim |

---

## Output Metrics and Interpretation

The primary dataset `data/run_summary.csv` summarizes historical runs across key outcome measures:

- `median_final_L`: Median final log-odds belief among honest agents. Negative values indicate that the honest population was misled toward the false state ($\theta = 0$).
- `pi_bar`: Mean local adversarial exposure across honest agents (proportion of an honest agent's neighbors that are attackers).
- `trim_selectivity` / `sever_selectivity`: Rate of removing adversarial connections minus rate of removing honest connections (higher is better).
- `boundary_margin`: Distance of median log-odds from the decision boundary relative to across-agent IQR.

See [data/README.md](data/README.md) for detailed descriptions of all recorded CSV columns.

---

## Repository Layout

```text
spm-bias-agents/
|-- configs/
|   `-- spm_experiment.json     # Default experiment and sweep configuration
|-- data/
|   |-- README.md               # Data dictionary for exported measurements
|   `-- run_summary.csv         # 47 historical primary-model run summaries
|-- scripts/
|   `-- summarize_results.py    # Zero-cost reproduction script for paper figures
|-- src/spm/
|   |-- agents.py               # Honest LLM agent and SPM adversary definitions
|   |-- analyze.py              # Statistical calibration, AUC, and detector scoring
|   |-- claim.py                # Factual claim statements and signal distributions
|   |-- config.py               # Pydantic/dataclass configuration schema
|   |-- defense.py              # W-MSR and SW-MSR filtering implementations
|   |-- env.py                  # Environment variable and .env handling
|   |-- llm.py                  # Model provider clients and mock engine
|   |-- network.py              # Synthetic graph generators (BA, ER, small-world)
|   |-- personas.py             # Persona loader (MOSAIC integration + fallback)
|   |-- runner.py               # Simulation execution engine and logging
|   |-- schema.py               # SQLite relational storage and belief math
|   `-- te.py                   # Transfer entropy and temporal statistics
|-- tests/                      # Unit and integration test suites
|-- run_spm.py                  # Unified CLI entry point (probe, run, sweep, analyze)
|-- RELEASE_NOTES.md            # Technical provenance and known scope boundaries
`-- THIRD_PARTY_NOTICES.md      # Upstream licensing and data boundaries
```

---

## Scope and Limitations

- **Exploratory findings**: In the primary cohort, local adversarial exposure ($\bar{\pi}$) correlates more strongly with network capture than global fraction ($p$). Temporal filtering selectively detects open-loop adversaries, but defense efficacy varies with network topology.
- **Mock vs. LLM**: The mock engine provides a deterministic, closed-form approximation for testing software interfaces; it is not a substitute for LLM-based empirical observations.
- **Theoretical scope**: Comments in the implementation reference historical theorem labels for traceability. These should be interpreted as design heuristics rather than absolute mathematical bounds. See [RELEASE_NOTES.md](RELEASE_NOTES.md) for technical notes.

---

## Citation

If you use this codebase, data, or methodology in your research, please cite our paper:

```bibtex
@article{spm2026network,
  title   = {How Many Plausible Voices Can Mislead a Network? Strategic Partial Misinformation in Multi-Agent Systems},
  author  = {Anonymous Authors},
  journal = {Under submission},
  year    = {2026}
}
```

If you use personas or simulation concepts derived from the MOSAIC project, please also cite:

```bibtex
@inproceedings{liu2025mosaic,
  title     = {{MOSAIC}: Modeling Social {AI} for Content Dissemination and Regulation in Multi-Agent Simulations},
  author    = {Liu, Genglin and others},
  booktitle = {Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing (EMNLP)},
  year      = {2025},
  url       = {https://aclanthology.org/2025.emnlp-main.325/}
}
```