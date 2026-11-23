#!/usr/bin/env python3
"""Entry point for the SPM / SW-MSR experiments.

    python run_spm.py probe   --config configs/spm_experiment.json
    python run_spm.py run     --config configs/spm_experiment.json --set attack.fraction=0.2
                              (values may be bare: --set defense.kind=swmsr)
    python run_spm.py sweep   --config configs/spm_experiment.json
    python run_spm.py analyze --config configs/spm_experiment.json --run <run_id>

Run `probe` and the p=0 cell of `sweep` first, and read the gate in
`analyze --run <p0 run>` before spending anything on the rest (failure mode F1).
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from spm import analyze, env, runner  # noqa: E402
from spm.config import Config  # noqa: E402
from spm.llm import make_backend  # noqa: E402

env.load(search_from=os.path.dirname(os.path.abspath(__file__)))


def _apply_overrides(raw: dict, pairs: list[str]) -> dict:
    """Apply `--set path.key=value` overrides.

    The value is read as JSON when it parses as JSON, and as a plain string
    otherwise, so `--set run_id=dev` works as well as `--set 'run_id="dev"'`.
    Requiring the JSON form would mean requiring shell quoting that Windows cmd
    does not provide, which is a trap rather than a feature.
    """
    for pair in pairs:
        key, _, value = pair.partition("=")
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            parsed = value
        node = raw
        parts = key.split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = parsed
    return raw


def _load(args) -> Config:
    with open(args.config) as f:
        raw = json.load(f)
    raw = env.apply(raw)                       # .env / environment
    if getattr(args, "set", None):             # --set always has the last word
        raw = _apply_overrides(raw, args.set)
    return Config.from_dict(raw)


# ------------------------------------------------------------------ the grid

def model_tag(engine: str) -> str:
    """A filesystem-safe short name for a model id.

    Model ids carry characters that cannot go in a path -- `cc/claude-opus-4-6`,
    `openai:gpt-4o` -- and run directories are named from them.
    """
    tag = "".join(c if c.isalnum() or c in "-." else "-" for c in engine)
    while "--" in tag:
        tag = tag.replace("--", "-")
    return tag.strip("-") or "model"


def sweep_cells(base: dict) -> list[dict]:
    """The minimum viable experiment: 30 runs.

    G0 honest null (gate) | G1 collapse point | G2 defenses (P5)
    G3 detector robustness (P6) | G4 placement contrast (P3)
    """
    sw = base.get("sweep", {})
    seeds = sw.get("seeds", [0, 1, 2])
    p_values = sw.get("p_values", [0.10, 0.20, 0.25, 0.30, 0.35])
    hub_p_values = sw.get("hub_p_values", [0.05, 0.10])
    defense_p = sw.get("defense_p", 0.35)
    cells: list[dict] = []

    def add(p, defense, policy, placement, seed, group, target_pi=None):
        cfg = copy.deepcopy(base)
        cfg["seed"] = seed
        # Not a Config field -- popped before Config.from_dict, see cmd_sweep.
        cfg["group"] = group
        # setdefault, not [...]: a config that omits a section should take the
        # dataclass defaults rather than raise KeyError.
        cfg.setdefault("attack", {}).update(
            fraction=p, policy=policy, placement=placement)
        if placement == "target_pi":
            cfg["attack"]["target_pi"] = (
                target_pi if target_pi is not None
                else sw.get("defense_target_pi", 0.42))
        cfg.setdefault("defense", {})["kind"] = defense
        cfg.setdefault("topology", {}).setdefault("kind", "ba")
        # The model is part of the identity: without it a second sweep against a
        # different model silently overwrites the first.
        # A target_pi cell is identified by the share it pins, not by p -- two
        # cells at the same p but different pinned shares are different experiments.
        tag = (f"pi{cfg['attack']['target_pi']:g}" if placement == "target_pi"
               else f"p{p:g}")
        cfg["run_id"] = (f"{model_tag(cfg.get('engine', 'model'))}__"
                         f"{cfg['topology']['kind']}-{placement}-{tag}"
                         f"-{defense}-{policy}-s{seed}")
        cells.append(cfg)

    for s_ in seeds:
        add(0.0, "none", "adaptive", "uniform", s_, "G0")
    for p in p_values:
        for s_ in seeds:
            add(p, "none", "adaptive", "uniform", s_, "G1")
    # G2 pins pi_bar rather than p. The arms must differ ONLY in the defense: at
    # N = 30 a uniform draw has sd(pi_bar) ~ 0.065, so arms drawn at the same p land
    # at different adversarial shares and a W-MSR vs SW-MSR gap could be placement
    # rather than defense. target_pi rejection-samples uniform placements until
    # pi_bar matches, so every arm faces the same share. The "none" arm is the
    # baseline the other two are measured against; without it neither number means
    # anything.
    for defense in ("none", "wmsr", "swmsr"):
        for s_ in seeds:
            add(defense_p, defense, "adaptive", "target_pi", s_, "G2")
    # G3 pins the same pi_bar as G2 for the same reason, and for one more: G2's
    # swmsr/adaptive arm is the third leg of this comparison. Adaptive, jitter and
    # scheduled attacks are only comparable if all three face the same adversarial
    # share, which uniform placement would not deliver (sd(pi_bar) ~ 0.065).
    for policy in ("jitter", "schedule"):
        for s_ in seeds:
            add(defense_p, "swmsr", policy, "target_pi", s_, "G3")
    for p in hub_p_values:
        for s_ in seeds:
            add(p, "none", "adaptive", "hub", s_, "G4")
    return cells


# ------------------------------------------------------------------ commands

def cmd_probe(args) -> None:
    cfg = _load(args)
    backend = make_backend(cfg.engine, cfg.temperature, seed=cfg.seed,
                           base_url=cfg.base_url, structured=cfg.structured,
                           max_tokens=cfg.max_tokens, rpm=cfg.rpm,
                           reasoning_effort=cfg.reasoning_effort)
    personas = runner.load_personas(cfg.persona_path, cfg.mosaic_path)
    result = runner.leakage_probe(cfg, backend, personas, n=args.n)
    print(json.dumps(result, indent=2))
    if result["leak_suspected"]:
        print("\nWARNING: elicited prior is far from 0.5. The model appears to have "
              "an opinion about the claim already. Change the claim in claim.py "
              "before running anything else.", file=sys.stderr)


def cmd_run(args) -> None:
    cfg = _load(args)
    print(json.dumps(runner.run(cfg, progress=True), indent=2))


def cmd_sweep(args) -> None:
    with open(args.config) as f:
        base = env.apply(json.load(f))
    if getattr(args, "set", None):
        base = _apply_overrides(base, args.set)
    cells = sweep_cells(base)
    if getattr(args, "only", None):
        want = {g.strip().upper() for g in args.only.split(",")}
        cells = [c for c in cells if c["group"] in want]
        if not cells:
            raise SystemExit(f"--only {args.only} matched no cells "
                             "(groups: G0,G1,G2,G3,G4)")
    failed: list[str] = []
    print(f"{len(cells)} runs planned | engine={base.get('engine')!r} "
          f"base_url={base.get('base_url') or 'api.openai.com'} "
          f"| from env: {env.describe()}", file=sys.stderr)
    for i, raw in enumerate(cells, 1):
        raw = {k: v for k, v in raw.items() if k != "group"}
        cfg = Config.from_dict(raw)
        target = os.path.join(runner._resolve(cfg.out_dir), cfg.run_id, "run.json")
        if args.resume and os.path.exists(target):
            print(f"[{i}/{len(cells)}] skip {cfg.run_id}", file=sys.stderr)
            continue
        print(f"[{i}/{len(cells)}] {cfg.run_id}", file=sys.stderr)
        try:
            summary = runner.run(cfg, progress=True)
        except Exception as exc:  # noqa: BLE001
            # One cell failing must not destroy the other 38. The cell writes no
            # run.json, so `sweep --resume` retries exactly this one later.
            failed.append(cfg.run_id)
            print(f"    FAILED: {type(exc).__name__}: {str(exc)[:600]}", file=sys.stderr)
            print("    continuing; re-run with --resume to retry this cell",
                  file=sys.stderr)
            continue
        print(f"    consensus={summary['honest_consensus_rate']:.3f} "
              f"calls={summary['model_calls']} "
              f"tokens={summary['prompt_tokens']}+{summary['completion_tokens']}",
              file=sys.stderr)
    if failed:
        print(f"\n{len(failed)}/{len(cells)} cells failed and were skipped:",
              file=sys.stderr)
        for name in failed:
            print(f"  {name}", file=sys.stderr)
        print("re-run `sweep --resume` to retry only these", file=sys.stderr)


def cmd_analyze(args) -> None:
    cfg = _load(args)
    out_dir = runner._resolve(cfg.out_dir)
    if args.run:
        db = os.path.join(out_dir, args.run, "simulation.db")
        # Analyse a run with the settings that run actually used, not whatever the
        # config file says now. s_bar depends on signal_accuracy, so reading it
        # from the current config silently produces a wrong rho_c for any run made
        # before the file changed -- exactly the kind of error that gets into a paper.
        run_json = os.path.join(out_dir, args.run, "run.json")
        used = cfg
        if os.path.exists(run_json):
            with open(run_json) as f:
                used = Config.from_dict(json.load(f)["config"])
            if used.claim.signal_accuracy != cfg.claim.signal_accuracy:
                print(f"note: analysing with the run's own signal_accuracy="
                      f"{used.claim.signal_accuracy} (config file now says "
                      f"{cfg.claim.signal_accuracy})", file=sys.stderr)
        else:
            print(f"warning: no run.json for {args.run!r}; falling back to the "
                  "current config, which may not match the run", file=sys.stderr)
        report = {
            "run": args.run,
            "engine": used.engine,
            "signal_accuracy": used.claim.signal_accuracy,
            # Both windows, always. rho_c moves ~28% between them on real data,
            # and a single number would hide a modelling choice rather than make
            # one. See analyze.calibrate.
            "calibration": analyze.calibrate(db, used.claim.signal_accuracy,
                                             used.attack.epsilon),
            "calibration_full_run": analyze.calibrate(db, used.claim.signal_accuracy,
                                                      used.attack.epsilon,
                                                      plateau_frac=0.0),
            "honest_null": analyze.honest_null(db, bins=used.defense.te_bins),
            "detectors": analyze.detector_scores(db, tau=used.defense.tau,
                                                 warmup=used.defense.te_warmup),
            "outcome": analyze.outcome(db),
            "retention": analyze.retention(db),
        }
    else:
        report = analyze.aggregate(out_dir)
    print(json.dumps(report, indent=2, default=str))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)

    for name, fn in (("probe", cmd_probe), ("run", cmd_run),
                     ("sweep", cmd_sweep), ("analyze", cmd_analyze)):
        p = sub.add_parser(name)
        p.add_argument("--config", default="configs/spm_experiment.json")
        p.set_defaults(func=fn)
        if name in ("probe", "run", "analyze", "sweep"):
            p.add_argument("--set", action="append", default=[],
                           metavar="path.key=json", help="override a config field")
        if name == "probe":
            p.add_argument("--n", type=int, default=30)
        if name == "sweep":
            p.add_argument("--resume", action="store_true",
                           help="skip cells that already have a run.json")
            p.add_argument("--only", default=None, metavar="G0[,G1,...]",
                           help="run only these groups: G0 gate, G1 collapse, "
                                "G2 defenses, G3 detector, G4 placement")
        if name == "analyze":
            p.add_argument("--run", default=None,
                           help="run_id to analyse; omit to aggregate all runs")

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
