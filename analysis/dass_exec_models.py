"""
Frequentist OLS: EF outcomes ~ DASS subscales (D/A/S)

Fits OLS models with HC3 robust SEs to test associations between each
executive-function indicator and DASS-21 subscales (Depression/Anxiety/Stress),
controlling for age and gender. Runs both univariate (one subscale at a time)
and multivariate (all three together) versions, and saves tidy coefficient
tables and model fit stats.

Outputs (results/analysis_outputs/):
  - dass_exec_models_coefficients.csv
  - dass_exec_models_fit.csv
  - dass_key_pvalues.csv (quick filter of D/A/S terms)

Usage:
  python analysis/dass_exec_models.py
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from data_loader_utils import load_master_dataset
import statsmodels.formula.api as smf
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from data_loader_utils import normalize_gender_series
from statistical_utils import apply_multiple_comparison_correction


BASE = Path(__file__).resolve().parent.parent
RES = BASE / "results"
OUT = RES / "analysis_outputs"
OUT.mkdir(parents=True, exist_ok=True)


def z(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    if s.dropna().std() in (0, None) or s.dropna().empty:
        return pd.Series(np.nan, index=s.index)
    return (s - s.mean()) / s.std()


def build_df() -> pd.DataFrame:
    master = load_master_dataset(use_cache=True, merge_cognitive_summary=True)

    def _pick(candidates):
        for c in candidates:
            if c in master.columns:
                return master[c]
        return pd.Series(np.nan, index=master.index)

    df = pd.DataFrame({
        "participant_id": master["participant_id"],
        "age": pd.to_numeric(master.get("age"), errors="coerce"),
        "gender": normalize_gender_series(master.get("gender_normalized", master.get("gender"))),
    })

    if "ucla_total" in master.columns:
        df["ucla_total"] = pd.to_numeric(master["ucla_total"], errors="coerce")
    elif "ucla_score" in master.columns:
        df["ucla_total"] = pd.to_numeric(master["ucla_score"], errors="coerce")

    df["dass_dep"] = pd.to_numeric(master.get("dass_depression"), errors="coerce")
    df["dass_anx"] = pd.to_numeric(master.get("dass_anxiety"), errors="coerce")
    df["dass_stress"] = pd.to_numeric(master.get("dass_stress"), errors="coerce")

    # EF metrics from master (with fallbacks)
    df["prp_bottleneck"] = pd.to_numeric(master.get("prp_bottleneck"), errors="coerce")
    if df["prp_bottleneck"].isna().all():
        short_col = pd.to_numeric(_pick(["rt2_soa_50", "rt2_soa_150"]), errors="coerce")
        long_col = pd.to_numeric(_pick(["rt2_soa_1200", "rt2_soa_1200_ms"]), errors="coerce")
        if not short_col.isna().all() and not long_col.isna().all():
            df["prp_bottleneck"] = short_col - long_col

    df["stroop_effect"] = pd.to_numeric(_pick(["stroop_interference", "stroop_effect"]), errors="coerce")
    if df["stroop_effect"].isna().all():
        incong = pd.to_numeric(_pick(["mrt_incong", "rt_mean_incongruent"]), errors="coerce")
        cong = pd.to_numeric(_pick(["mrt_cong", "rt_mean_congruent"]), errors="coerce")
        if not incong.isna().all() and not cong.isna().all():
            df["stroop_effect"] = incong - cong

    df["wcst_total_errors"] = pd.to_numeric(
        _pick(["wcst_total_errors", "totalerrorcount", "total_error_count", "totalErrorCount"]), errors="coerce"
    )
    df["wcst_persev_errors"] = pd.to_numeric(
        _pick(["wcst_persev_errors", "perseverativeerrorcount", "perseverative_error_count", "perseverativeErrorCount"]), errors="coerce"
    )
    df["wcst_nonpersev_errors"] = pd.to_numeric(
        _pick(["wcst_nonpersev_errors", "nonperseverativeerrorcount", "non_perseverative_error_count", "nonPerseverativeErrorCount"]), errors="coerce"
    )
    df["wcst_conceptual_pct"] = pd.to_numeric(
        _pick(["wcst_conceptual_pct", "conceptuallevelresponsespercent", "conceptual_level_responses_percent"]), errors="coerce"
    )
    df["wcst_persev_resp_pct"] = pd.to_numeric(
        _pick(["wcst_persev_resp_pct", "perseverativeresponsespercent", "perseverative_responses_percent"]), errors="coerce"
    )
    df["wcst_failure_to_maintain_set"] = pd.to_numeric(
        _pick(["wcst_failure_to_maintain_set", "failuretomaintainset", "failure_to_maintain_set"]), errors="coerce"
    )

    # z-scores for D/A/S
    df["z_dass_dep"] = z(df["dass_dep"])
    df["z_dass_anx"] = z(df["dass_anx"])
    df["z_dass_stress"] = z(df["dass_stress"])
    return df


def add_meta_control(df: pd.DataFrame) -> pd.DataFrame:
    ef_cols = ["stroop_effect", "prp_bottleneck", "wcst_total_errors"]
    complete = df.dropna(subset=ef_cols).copy()
    if len(complete) < 15:
        df["meta_control"] = np.nan
        return df
    scaler = StandardScaler()
    X = scaler.fit_transform(complete[ef_cols])
    pca = PCA(n_components=1, random_state=42)
    scores = pca.fit_transform(X).ravel()
    tmp = complete[["participant_id"]].assign(meta_control=scores)
    return df.merge(tmp, on="participant_id", how="left")


@dataclass
class FitRow:
    outcome: str
    spec: str
    nobs: int
    r2: float
    aic: float
    bic: float


def run_models(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    outcomes = [
        ("stroop_effect", "Stroop interference (ms)"),
        ("prp_bottleneck", "PRP bottleneck (ms)"),
        ("wcst_total_errors", "WCST total errors"),
        ("meta_control", "Latent meta-control"),
    ]

    rows_coef: List[pd.DataFrame] = []
    rows_fit: List[dict] = []

    # Model specs
    specs = {
        "dep_only": "y ~ z_dass_dep + age + C(gender)",
        "anx_only": "y ~ z_dass_anx + age + C(gender)",
        "str_only": "y ~ z_dass_stress + age + C(gender)",
        "all_three": "y ~ z_dass_dep + z_dass_anx + z_dass_stress + age + C(gender)",
    }

    for col, nice in outcomes:
        need = [col, "z_dass_dep", "z_dass_anx", "z_dass_stress", "age", "gender"]
        data = df[need].dropna()
        if len(data) < 25:
            continue
        data = data.rename(columns={col: "y"})
        for spec_name, formula in specs.items():
            try:
                model = smf.ols(formula=formula, data=data).fit(cov_type="HC3")
            except Exception:
                continue
            tab = model.summary2().tables[1].copy()
            tab = tab.rename(columns={
                "Coef.": "estimate",
                "Std.Err.": "std_error",
                "[0.025": "conf_low",
                "0.975]": "conf_high",
                "P>|t|": "p_value",
                "t": "stat",
            })
            coef_df = tab.reset_index().rename(columns={"index": "term"})
            coef_df.insert(0, "outcome", nice)
            coef_df.insert(1, "spec", spec_name)
            rows_coef.append(coef_df)

            rows_fit.append(
                {
                    "outcome": nice,
                    "spec": spec_name,
                    "nobs": int(model.nobs),
                    "r2": float(model.rsquared),
                    "aic": float(model.aic),
                    "bic": float(model.bic),
                }
            )

    coef_all = pd.concat(rows_coef, ignore_index=True) if rows_coef else pd.DataFrame()
    fit_all = pd.DataFrame(rows_fit)
    return coef_all, fit_all


def main() -> None:
    df = build_df()
    df = add_meta_control(df)
    coef, fit = run_models(df)
    coef_path = OUT / "dass_exec_models_coefficients.csv"
    fit_path = OUT / "dass_exec_models_fit.csv"
    coef.to_csv(coef_path, index=False)
    fit.to_csv(fit_path, index=False)
    print(f"Saved coefficients to {coef_path.name}")
    print(f"Saved fit stats to {fit_path.name}")
    # Quick key rows
    if not coef.empty:
        key = coef[coef["term"].isin(["z_dass_dep", "z_dass_anx", "z_dass_stress"])]

        # ADDED: Apply multiple comparison correction (FDR) to key p-values
        if 'p_value' in key.columns and len(key) > 0:
            p_values = key['p_value'].values
            reject_fdr, p_adjusted_fdr = apply_multiple_comparison_correction(
                p_values,
                method='fdr_bh',
                alpha=0.05
            )
            key = key.copy()
            key['p_adjusted_fdr'] = p_adjusted_fdr
            key['significant_fdr'] = reject_fdr
            print(f"\n[Multiple comparison correction applied: {len(key)} tests, FDR method]")

        key.to_csv(OUT / "dass_key_pvalues.csv", index=False)
        print("\n=== Key D/A/S coefficients (HC3) ===")
        cols = [c for c in ["outcome", "spec", "term", "estimate", "p_value", "p_adjusted_fdr", "conf_low", "conf_high"] if c in key.columns]
        print(key[cols].to_string(index=False))


if __name__ == "__main__":
    main()
