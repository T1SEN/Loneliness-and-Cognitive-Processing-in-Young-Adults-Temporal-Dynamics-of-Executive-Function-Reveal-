#!/usr/bin/env python3
"""
One-predictor models: Each EF outcome regressed on UCLA loneliness only
(no DASS covariates) to check raw directional effects.
"""

from __future__ import annotations

import sys
from pathlib import Path

import statsmodels.formula.api as smf

sys.path.append(str(Path(__file__).resolve().parent))
from loneliness_exec_models import build_analysis_dataframe  # noqa: E402


def main():
    df = build_analysis_dataframe()
    cols = [
        ("stroop_effect", "Stroop 간섭"),
        ("prp_bottleneck", "PRP 병목"),
        ("wcst_total_errors", "WCST 총 오류"),
        ("wcst_persev_errors", "WCST 보속 오류"),
        ("wcst_nonpersev_errors", "WCST 비보속 오류"),
    ]
    for col, label in cols:
        data = df[["z_ucla", col]].dropna()
        if len(data) < 30:
            continue
        model = smf.ols(f"{col} ~ z_ucla", data=data).fit()
        print(f"\n=== {label} (n={int(model.nobs)}) ===")
        print(model.summary())


if __name__ == "__main__":
    main()
