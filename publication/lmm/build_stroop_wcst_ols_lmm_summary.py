from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

OUTPUT_DIR = ROOT / "publication" / "data" / "outputs" / "analysis" / "reviewer_response"
WCST_OLS_DIR = ROOT / "publication" / "data" / "outputs" / "analysis" / "wcst"
OUT_PATH = OUTPUT_DIR / "stroop_wcst_ols_lmm_summary.csv"


def add_record(
    records: list[dict[str, object]],
    task: str,
    model_type: str,
    analysis: str,
    effect: str,
    beta: float | None,
    se: float | None,
    stat_type: str | None,
    stat: float | None,
    p: float | None,
    n: int | None,
    n_trials: int | None,
    n_participants: int | None,
    scale: str | None,
    segment: str | None,
    source: str,
) -> None:
    records.append({
        "task": task,
        "model_type": model_type,
        "analysis": analysis,
        "effect": effect,
        "beta": beta,
        "se": se,
        "stat_type": stat_type,
        "stat": stat,
        "p": p,
        "n": n,
        "n_trials": n_trials,
        "n_participants": n_participants,
        "scale": scale,
        "segment": segment,
        "source": source,
    })


def main() -> None:
    records: list[dict[str, object]] = []

    # Stroop OLS (participant-level interference slope)
    stroop_primary_path = OUTPUT_DIR / "primary_dv_results.csv"
    if stroop_primary_path.exists():
        df = pd.read_csv(stroop_primary_path)
        row = df[df["dv"] == "stroop_interference_slope"]
        if not row.empty:
            r = row.iloc[0]
            add_record(
                records,
                task="stroop",
                model_type="OLS",
                analysis="interference_slope_ols",
                effect="ucla_beta",
                beta=float(r["ucla_beta"]),
                se=float(r["ucla_se"]),
                stat_type=None,
                stat=None,
                p=float(r["ucla_p"]),
                n=int(r["n"]),
                n_trials=None,
                n_participants=int(r["n"]),
                scale="ms",
                segment=None,
                source=str(stroop_primary_path),
            )

    # Stroop trial-level OLS (segment x UCLA)
    stroop_trial_ols_path = OUTPUT_DIR / "stroop_trial_level_ols.csv"
    if stroop_trial_ols_path.exists():
        df = pd.read_csv(stroop_trial_ols_path)
        if not df.empty:
            r = df.iloc[0]
            add_record(
                records,
                task="stroop",
                model_type="OLS",
                analysis="trial_level_ols_segment_x_ucla",
                effect="segment_x_ucla",
                beta=float(r["beta"]),
                se=float(r["se"]),
                stat_type="t",
                stat=float(r["t"]),
                p=float(r["p"]),
                n=None,
                n_trials=int(r["n_trials"]),
                n_participants=int(r["n_participants"]),
                scale="ms",
                segment=None,
                source=str(stroop_trial_ols_path),
            )

    # Stroop LMM (trial_scaled x cond x UCLA)
    stroop_lmm_path = OUTPUT_DIR / "stroop_interference_slope_lmm.csv"
    if stroop_lmm_path.exists():
        df = pd.read_csv(stroop_lmm_path)
        if not df.empty:
            r = df.iloc[0]
            add_record(
                records,
                task="stroop",
                model_type="LMM",
                analysis="interference_slope_lmm",
                effect="trial_scaled_x_cond_x_ucla",
                beta=float(r["interference_time_x_ucla_beta"]),
                se=float(r["interference_time_x_ucla_se"]),
                stat_type="z",
                stat=float(r["interference_time_x_ucla_z"]),
                p=float(r["interference_time_x_ucla_p"]),
                n=None,
                n_trials=int(r["n_trials"]),
                n_participants=int(r["n_participants"]),
                scale="log_rt",
                segment=None,
                source=str(stroop_lmm_path),
            )

    # WCST OLS: segment RT means/slopes
    wcst_seg_path = WCST_OLS_DIR / "wcst_segment_rt_regression_nodiscovery_nopreswitch_ols.csv"
    if wcst_seg_path.exists():
        df = pd.read_csv(wcst_seg_path)
        for _, r in df.iterrows():
            add_record(
                records,
                task="wcst",
                model_type="OLS",
                analysis="segment_rt_ols",
                effect=str(r["outcome_column"]),
                beta=float(r["ucla_beta"]),
                se=float(r["ucla_se"]),
                stat_type="t",
                stat=float(r["ucla_t"]),
                p=float(r["ucla_p"]),
                n=int(r["n"]),
                n_trials=None,
                n_participants=int(r["n"]),
                scale="ms",
                segment=str(r.get("segment", "")),
                source=str(wcst_seg_path),
            )

    # WCST OLS: segment RT deltas
    wcst_delta_path = WCST_OLS_DIR / "wcst_segment_rt_delta_regression_nodiscovery_nopreswitch_ols.csv"
    if wcst_delta_path.exists():
        df = pd.read_csv(wcst_delta_path)
        for _, r in df.iterrows():
            add_record(
                records,
                task="wcst",
                model_type="OLS",
                analysis="segment_rt_delta_ols",
                effect=str(r["outcome_column"]),
                beta=float(r["ucla_beta"]),
                se=float(r["ucla_se"]),
                stat_type="t",
                stat=float(r["ucla_t"]),
                p=float(r["ucla_p"]),
                n=int(r["n"]),
                n_trials=None,
                n_participants=int(r["n"]),
                scale="ms",
                segment=str(r.get("segment", "")),
                source=str(wcst_delta_path),
            )

    # WCST OLS: overall mean and category slope
    wcst_overall_ols_path = OUTPUT_DIR / "wcst_overall_category_rt_ols.csv"
    if wcst_overall_ols_path.exists():
        df = pd.read_csv(wcst_overall_ols_path)
        for _, r in df.iterrows():
            add_record(
                records,
                task="wcst",
                model_type="OLS",
                analysis="overall_category_rt_ols",
                effect=str(r["outcome_column"]),
                beta=float(r["ucla_beta"]),
                se=float(r["ucla_se"]),
                stat_type="t",
                stat=float(r["ucla_t"]),
                p=float(r["ucla_p"]),
                n=int(r["n"]),
                n_trials=None,
                n_participants=int(r["n"]),
                scale="ms",
                segment=None,
                source=str(wcst_overall_ols_path),
            )

    # WCST OLS: post-error log RT mean
    wcst_post_error_ols_path = OUTPUT_DIR / "wcst_post_error_log_ols.csv"
    if wcst_post_error_ols_path.exists():
        df = pd.read_csv(wcst_post_error_ols_path)
        if not df.empty:
            r = df.iloc[0]
            add_record(
                records,
                task="wcst",
                model_type="OLS",
                analysis="post_error_log_ols",
                effect="post_error_log_rt_mean",
                beta=float(r["ucla_beta"]),
                se=float(r["ucla_se"]),
                stat_type="t",
                stat=float(r["ucla_t"]),
                p=float(r["ucla_p"]),
                n=None,
                n_trials=int(r["n_trials"]),
                n_participants=int(r["n_participants"]),
                scale="log_rt",
                segment=None,
                source=str(wcst_post_error_ols_path),
            )

    # WCST OLS: post-shift error log RT mean
    wcst_post_shift_ols_path = OUTPUT_DIR / "wcst_post_shift_error_log_ols.csv"
    if wcst_post_shift_ols_path.exists():
        df = pd.read_csv(wcst_post_shift_ols_path)
        if not df.empty:
            r = df.iloc[0]
            add_record(
                records,
                task="wcst",
                model_type="OLS",
                analysis="post_shift_error_log_ols",
                effect="post_shift_error_log_rt_mean",
                beta=float(r["ucla_beta"]),
                se=float(r["ucla_se"]),
                stat_type="t",
                stat=float(r["ucla_t"]),
                p=float(r["ucla_p"]),
                n=None,
                n_trials=int(r["n_trials"]),
                n_participants=int(r["n_participants"]),
                scale="log_rt",
                segment=None,
                source=str(wcst_post_shift_ols_path),
            )


    out_df = pd.DataFrame(records)
    out_df.to_csv(OUT_PATH, index=False, encoding="utf-8-sig")
    print(f"Saved: {OUT_PATH}")
    print(out_df.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
