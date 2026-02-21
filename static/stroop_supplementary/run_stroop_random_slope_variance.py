"""
Stroop random slope variance (trial-level LMM).

Fits the trial-level interference model with random intercepts and
random slopes for trial position, then reports VarCorr summaries.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

ROOT = Path(__file__).resolve().parents[1]
if not (ROOT / "static").exists():
    ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from static.analysis.utils import get_output_dir
from static.stroop_lmm.run_stroop_trial_lmm import (
    prepare_interference_trials,
    load_base_data,
    add_zscores,
)


def main() -> None:
    base = load_base_data()
    base = add_zscores(
        base,
        ["ucla_score", "dass_depression", "dass_anxiety", "dass_stress", "age"],
    )

    int_trials = prepare_interference_trials()
    if int_trials.empty:
        raise RuntimeError("No interference trials available after filtering.")

    predictors = base[
        [
            "participant_id",
            "z_ucla_score",
            "z_dass_depression",
            "z_dass_anxiety",
            "z_dass_stress",
            "z_age",
            "gender_male",
        ]
    ].dropna()

    int_df = int_trials.merge(predictors, on="participant_id", how="inner")
    if int_df.empty:
        raise RuntimeError("No interference trials available after merging predictors.")

    formula = (
        "log_rt ~ trial_scaled * cond_code * z_ucla_score + "
        "z_dass_depression + z_dass_anxiety + z_dass_stress + z_age + gender_male"
    )

    model = smf.mixedlm(
        formula,
        data=int_df,
        groups=int_df["participant_id"],
        re_formula="1 + trial_scaled",
    )
    result = model.fit(reml=False, method="lbfgs", maxiter=200)

    cov_re = result.cov_re
    bse_re = getattr(result, "bse_re", None)

    var_int = float(cov_re.loc["Group", "Group"]) if "Group" in cov_re.index else np.nan
    var_slope = (
        float(cov_re.loc["trial_scaled", "trial_scaled"])
        if "trial_scaled" in cov_re.index
        else np.nan
    )
    cov_int_slope = (
        float(cov_re.loc["Group", "trial_scaled"])
        if ("Group" in cov_re.index and "trial_scaled" in cov_re.index)
        else np.nan
    )

    corr_int_slope = np.nan
    if np.isfinite(var_int) and np.isfinite(var_slope) and var_int > 0 and var_slope > 0:
        corr_int_slope = cov_int_slope / np.sqrt(var_int * var_slope)

    var_slope_se = np.nan
    var_slope_ci_low_raw = np.nan
    var_slope_ci_low = np.nan
    var_slope_ci_high = np.nan
    var_slope_ci_note = ""
    if bse_re is not None:
        for idx, val in bse_re.items():
            if "trial_scaled Var" in str(idx):
                var_slope_se = float(val)
                break
        if np.isfinite(var_slope_se) and np.isfinite(var_slope):
            var_slope_ci_low_raw = var_slope - 1.96 * var_slope_se
            var_slope_ci_low = max(0.0, var_slope_ci_low_raw)
            var_slope_ci_high = var_slope + 1.96 * var_slope_se
            if var_slope_ci_low_raw < 0:
                var_slope_ci_note = (
                    "Wald CI lower bound fell below 0; reported lower bound is truncated to 0 "
                    "because variance is non-negative."
                )

    output = pd.DataFrame(
        [
            {
                "n_trials": int(len(int_df)),
                "n_participants": int(int_df["participant_id"].nunique()),
                "re_formula": "1 + trial_scaled",
                "method": "lbfgs",
                "converged": bool(getattr(result, "converged", False)),
                "var_intercept": var_int,
                "var_slope_trial_scaled": var_slope,
                "cov_int_slope": cov_int_slope,
                "corr_int_slope": corr_int_slope,
                "var_slope_se": var_slope_se,
                "var_slope_ci_low_raw": var_slope_ci_low_raw,
                "var_slope_ci_low": var_slope_ci_low,
                "var_slope_ci_high": var_slope_ci_high,
                "var_slope_ci_note": var_slope_ci_note,
            }
        ]
    )

    out_dir = get_output_dir("overall", bucket="supplementary")
    out_path = out_dir / "stroop_random_slope_variance.csv"
    output.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"Saved: {out_path}")
    print(output.to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stroop random slope variance (trial-level LMM).")
    _ = parser.parse_args()
    main()
