"""Checks on the public summary artifact; no raw database or API required."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from summarize_results import auc, summarize


def test_released_cohort_and_auc():
    result = summarize()
    assert result["included_runs"] == 47
    assert result["primary_n"] == 18
    assert result["primary_captures"] == 7
    assert result["descriptive_auc"]["p"] == pytest.approx(0.7792207792207793)
    assert result["descriptive_auc"]["pi_bar"] == pytest.approx(0.987012987012987)
    assert result["groups"]["swmsr_f0_adaptive"]["n"] == 3
    assert result["groups"]["swmsr_f0_adaptive"]["sever_attacker"]["mean"] == pytest.approx(0.08252786982326638)


def test_auc_ties_and_missing_class():
    rows = [{"median_final_L": -1, "score": 0.2},
            {"median_final_L": 1, "score": 0.2}]
    assert auc(rows, "score") == 0.5
    with pytest.raises(ValueError, match="both captured and held"):
        auc(rows[:1], "score")