"""Reproduce descriptive summaries from the released CSV without model calls."""
from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path


DEFAULT_DATA = Path(__file__).resolve().parents[1] / "data" / "run_summary.csv"


def auc(rows: list[dict], key: str) -> float:
    captured = [r for r in rows if float(r["median_final_L"]) < 0]
    held = [r for r in rows if float(r["median_final_L"]) >= 0]
    if not captured or not held:
        raise ValueError("AUC requires both captured and held runs")
    return statistics.mean(
        float(a[key]) > float(h[key])
        or 0.5 * (float(a[key]) == float(h[key]))
        for a in captured for h in held
    )


def summarize(path: Path = DEFAULT_DATA) -> dict:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len({r["run_id"] for r in rows}) != len(rows):
        raise ValueError("Duplicate run IDs in data")
    primary = [r for r in rows if int(r["k"]) > 0
               and r["defense"] == "none" and int(r["T"]) == 100
               and r["placement"] != "hub"]
    groups: dict[str, list[dict]] = {}
    for row in rows:
        if int(row["T"]) == 60 and row["placement"] == "target_pi":
            key = f"{row['defense']}_f{row['f']}_{row['policy']}"
            groups.setdefault(key, []).append(row)
    summaries = {}
    for key, group in sorted(groups.items()):
        result = {"n": len(group)}
        for field in ("median_final_L", "trim_selectivity", "sever_selectivity",
                      "sever_attacker", "sever_honest"):
            values = [float(r[field]) for r in group]
            result[field] = {
                "mean": statistics.mean(values),
                "sd": statistics.stdev(values) if len(values) > 1 else None,
            }
        summaries[key] = result
    return {
        "included_runs": len(rows),
        "primary_n": len(primary),
        "primary_captures": sum(float(r["median_final_L"]) < 0 for r in primary),
        "descriptive_auc": {key: auc(primary, key) for key in ("p", "pi_bar")},
        "groups": summaries,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    args = parser.parse_args()
    print(json.dumps(summarize(args.data), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()