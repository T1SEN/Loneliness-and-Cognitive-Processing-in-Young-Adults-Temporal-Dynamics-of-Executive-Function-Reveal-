"""
WCST non-HMM extensions for mind-wandering proxies.

Analyses included:
1) Non-HMM UCLA regressions with HC3 + FDR across WCST outcomes.
2) Trial-level lapse/run-length proxies (z-RT thresholds).
3) Shift-locked RT/accuracy trajectories around rule changes.
"""

from __future__ import annotations

import sys
if sys.platform.startswith("win") and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests

from publication.analysis.utils import get_analysis_data, get_output_dir, get_tier1_outcomes
from publication.preprocessing.wcst.loaders import load_wcst_trials

NON_HMM_EXCLUDE = ("hmm", "slow_prob")

LAPSE_LABELS = {
    "wcst_lapse_rate_z25": "WCST Lapse Rate (z>2.5)",
    "wcst_lapse_rate_z30": "WCST Lapse Rate (z>3.0)",
    "wcst_lapse_run_mean_z25": "WCST Lapse Run Mean (z>2.5)",
    "wcst_lapse_run_max_z25": "WCST Lapse Run Max (z>2.5)",
    "wcst_lapse_run_mean_z30": "WCST Lapse Run Mean (z>3.0)",
    "wcst_lapse_run_max_z30": "WCST Lapse Run Max (z>3.0)",
    "wcst_error_run_mean": "WCST Error Run Mean",
    "wcst_error_run_max": "WCST Error Run Max",
    "wcst_error_rate": "WCST Error Rate",
}

SHIFT_LABELS = {
    "wcst_shift_rt_km2": "WCST Shift RT (k=-2)",
    "wcst_shift_rt_km1": "WCST Shift RT (k=-1)",
    "wcst_shift_rt_k0": "WCST Shift RT (k=0)",
    "wcst_shift_rt_k1": "WCST Shift RT (k=1)",
    "wcst_shift_rt_k2": "WCST Shift RT (k=2)",
    "wcst_shift_rt_k3": "WCST Shift RT (k=3)",
    "wcst_shift_rt_k4": "WCST Shift RT (k=4)",
    "wcst_shift_rt_k5": "WCST Shift RT (k=5)",
    "wcst_shift_acc_km2": "WCST Shift Accuracy (k=-2)",
    "wcst_shift_acc_km1": "WCST Shift Accuracy (k=-1)",
    "wcst_shift_acc_k0": "WCST Shift Accuracy (k=0)",
    "wcst_shift_acc_k1": "WCST Shift Accuracy (k=1)",
    "wcst_shift_acc_k2": "WCST Shift Accuracy (k=2)",
    "wcst_shift_acc_k3": "WCST Shift Accuracy (k=3)",
    "wcst_shift_acc_k4": "WCST Shift Accuracy (k=4)",
    "wcst_shift_acc_k5": "WCST Shift Accuracy (k=5)",
    "wcst_shift_rt_pre_mean": "WCST Shift RT Pre-Mean (k=-2..-1)",
    "wcst_shift_rt_post_mean": "WCST Shift RT Post-Mean (k=0..2)",
    "wcst_shift_rt_delta": "WCST Shift RT Delta (post-pre)",
    "wcst_shift_acc_pre_mean": "WCST Shift Accuracy Pre-Mean (k=-2..-1)",
    "wcst_shift_acc_post_mean": "WCST Shift Accuracy Post-Mean (k=0..2)",
    "wcst_shift_acc_delta": "WCST Shift Accuracy Delta (post-pre)",
}


def _run_lengths(mask: Iterable[bool]) -> list[int]:
    lengths: list[int] = []
    count = 0
    for val in mask:
        if val:
            count += 1
        elif count:
            lengths.append(count)
            count = 0
    if count:
        lengths.append(count)
    return lengths


def _summarize_runs(lengths: list[int]) -> tuple[float, float]:
    if not lengths:
        return 0.0, 0.0
    return float(np.mean(lengths)), float(np.max(lengths))


def compute_lapse_metrics(
    trials: pd.DataFrame,
    min_trials: int = 60,
) -> pd.DataFrame:
    records: list[dict[str, float]] = []

    for pid, pdata in trials.groupby("participant_id"):
        pdata = pdata.sort_values("trial_index") if "trial_index" in pdata.columns else pdata
        if len(pdata) < min_trials:
            continue

        rt = pdata["rt_ms"].to_numpy(dtype=float)
        if len(rt) < min_trials:
            continue
        mean_rt = float(np.mean(rt))
        std_rt = float(np.std(rt, ddof=1))
        if not np.isfinite(std_rt) or std_rt <= 0:
            continue

        z_rt = (rt - mean_rt) / std_rt
        lapse_z25 = z_rt > 2.5
        lapse_z30 = z_rt > 3.0

        lapse_runs_25 = _run_lengths(lapse_z25)
        lapse_runs_30 = _run_lengths(lapse_z30)
        lapse_mean_25, lapse_max_25 = _summarize_runs(lapse_runs_25)
        lapse_mean_30, lapse_max_30 = _summarize_runs(lapse_runs_30)

        if "correct" in pdata.columns:
            correct = pdata["correct"].astype(bool).to_numpy()
            error_mask = ~correct
            error_runs = _run_lengths(error_mask)
            error_run_mean, error_run_max = _summarize_runs(error_runs)
            error_rate = float(np.mean(error_mask))
        else:
            error_run_mean = np.nan
            error_run_max = np.nan
            error_rate = np.nan

        records.append({
            "participant_id": pid,
            "wcst_lapse_rate_z25": float(np.mean(lapse_z25)),
            "wcst_lapse_rate_z30": float(np.mean(lapse_z30)),
            "wcst_lapse_run_mean_z25": lapse_mean_25,
            "wcst_lapse_run_max_z25": lapse_max_25,
            "wcst_lapse_run_mean_z30": lapse_mean_30,
            "wcst_lapse_run_max_z30": lapse_max_30,
            "wcst_error_run_mean": error_run_mean,
            "wcst_error_run_max": error_run_max,
            "wcst_error_rate": error_rate,
            "wcst_n_trials": int(len(pdata)),
        })

    return pd.DataFrame(records)


def compute_shift_locked_metrics(
    trials: pd.DataFrame,
    offsets: Iterable[int] = (-2, -1, 0, 1, 2, 3, 4, 5),
    min_shifts: int = 3,
) -> pd.DataFrame:
    records: list[dict[str, float]] = []
    offsets = list(offsets)

    for pid, pdata in trials.groupby("participant_id"):
        pdata = pdata.sort_values("trial_index") if "trial_index" in pdata.columns else pdata
        if "cond" not in pdata.columns:
            continue
        cond = pdata["cond"].astype(str).str.lower().values
        if len(cond) < 10:
            continue

        change_idx = [i for i in range(1, len(cond)) if cond[i] != cond[i - 1]]
        if len(change_idx) < min_shifts:
            continue

        rt = pdata["rt_ms"].to_numpy(dtype=float)
        correct = pdata["correct"].astype(bool).to_numpy() if "correct" in pdata.columns else None

        rt_metrics: dict[str, float] = {}
        acc_metrics: dict[str, float] = {}
        for k in offsets:
            values_rt: list[float] = []
            values_acc: list[float] = []
            for idx in change_idx:
                j = idx + k
                if j < 0 or j >= len(rt):
                    continue
                if np.isfinite(rt[j]):
                    values_rt.append(float(rt[j]))
                if correct is not None:
                    values_acc.append(float(correct[j]))

            suffix = f"km{abs(k)}" if k < 0 else f"k{k}"
            rt_metrics[f"wcst_shift_rt_{suffix}"] = float(np.mean(values_rt)) if values_rt else np.nan
            if correct is not None:
                acc_metrics[f"wcst_shift_acc_{suffix}"] = float(np.mean(values_acc)) if values_acc else np.nan

        pre_keys = ["wcst_shift_rt_km2", "wcst_shift_rt_km1"]
        post_keys = ["wcst_shift_rt_k0", "wcst_shift_rt_k1", "wcst_shift_rt_k2"]
        rt_pre_vals = [rt_metrics.get(k) for k in pre_keys if np.isfinite(rt_metrics.get(k, np.nan))]
        rt_post_vals = [rt_metrics.get(k) for k in post_keys if np.isfinite(rt_metrics.get(k, np.nan))]
        rt_pre_mean = float(np.mean(rt_pre_vals)) if rt_pre_vals else np.nan
        rt_post_mean = float(np.mean(rt_post_vals)) if rt_post_vals else np.nan
        rt_delta = rt_post_mean - rt_pre_mean if np.isfinite(rt_pre_mean) and np.isfinite(rt_post_mean) else np.nan

        acc_pre_mean = np.nan
        acc_post_mean = np.nan
        acc_delta = np.nan
        if correct is not None:
            pre_keys_acc = ["wcst_shift_acc_km2", "wcst_shift_acc_km1"]
            post_keys_acc = ["wcst_shift_acc_k0", "wcst_shift_acc_k1", "wcst_shift_acc_k2"]
            acc_pre_vals = [acc_metrics.get(k) for k in pre_keys_acc if np.isfinite(acc_metrics.get(k, np.nan))]
            acc_post_vals = [acc_metrics.get(k) for k in post_keys_acc if np.isfinite(acc_metrics.get(k, np.nan))]
            acc_pre_mean = float(np.mean(acc_pre_vals)) if acc_pre_vals else np.nan
            acc_post_mean = float(np.mean(acc_post_vals)) if acc_post_vals else np.nan
            acc_delta = acc_post_mean - acc_pre_mean if np.isfinite(acc_pre_mean) and np.isfinite(acc_post_mean) else np.nan

        record = {"participant_id": pid, "wcst_n_shifts": int(len(change_idx))}
        record.update(rt_metrics)
        record.update(acc_metrics)
        record.update({
            "wcst_shift_rt_pre_mean": rt_pre_mean,
            "wcst_shift_rt_post_mean": rt_post_mean,
            "wcst_shift_rt_delta": rt_delta,
            "wcst_shift_acc_pre_mean": acc_pre_mean,
            "wcst_shift_acc_post_mean": acc_post_mean,
            "wcst_shift_acc_delta": acc_delta,
        })
        records.append(record)

    return pd.DataFrame(records)


def run_ucla_regression(
    df: pd.DataFrame,
    outcome: str,
    cov_type: str = "HC3",
    min_n: int = 30,
) -> dict | None:
    required = [
        outcome,
        "z_ucla_score",
        "z_dass_depression",
        "z_dass_anxiety",
        "z_dass_stress",
        "z_age",
        "gender_male",
    ]
    cols = [c for c in required if c in df.columns]
    sub = df[cols].dropna()
    if len(sub) < min_n:
        return None

    formula = (
        f"{outcome} ~ z_ucla_score + z_dass_depression + "
        "z_dass_anxiety + z_dass_stress + z_age + C(gender_male)"
    )
    try:
        model = smf.ols(formula, data=sub).fit(cov_type=cov_type)
    except Exception:
        return None

    return {
        "outcome_column": outcome,
        "n": int(len(sub)),
        "ucla_beta": float(model.params.get("z_ucla_score", np.nan)),
        "ucla_se": float(model.bse.get("z_ucla_score", np.nan)),
        "ucla_t": float(model.tvalues.get("z_ucla_score", np.nan)),
        "ucla_p": float(model.pvalues.get("z_ucla_score", np.nan)),
        "r2": float(model.rsquared),
        "adj_r2": float(model.rsquared_adj),
        "cov_type": cov_type,
    }


def add_fdr(df: pd.DataFrame, p_col: str = "ucla_p") -> pd.DataFrame:
    df = df.copy()
    pvals = df[p_col].to_numpy()
    mask = np.isfinite(pvals)
    df["ucla_fdr_q"] = np.nan
    df["ucla_fdr_sig"] = False
    if mask.any():
        _, qvals, _, _ = multipletests(pvals[mask], method="fdr_bh")
        df.loc[mask, "ucla_fdr_q"] = qvals
        df.loc[mask, "ucla_fdr_sig"] = qvals < 0.05
    return df


def run_non_hmm_regressions(output_dir: Path) -> pd.DataFrame:
    df = get_analysis_data("wcst")
    outcomes = get_tier1_outcomes("wcst")
    label_map = {col: label for col, label in outcomes}

    non_hmm_cols = [
        col for col in label_map
        if col in df.columns and not any(key in col.lower() for key in NON_HMM_EXCLUDE)
    ]

    results = []
    for col in non_hmm_cols:
        res = run_ucla_regression(df, col, cov_type="HC3")
        if res:
            res["outcome"] = label_map.get(col, col)
            results.append(res)

    results_df = pd.DataFrame(results)
    if not results_df.empty:
        results_df = add_fdr(results_df)
        results_df = results_df.sort_values("ucla_p")
        results_df.to_csv(output_dir / "wcst_non_hmm_hc3.csv", index=False, encoding="utf-8-sig")

    return results_df


def run_trial_level_regressions(
    output_dir: Path,
    trials: pd.DataFrame,
) -> pd.DataFrame:
    df = get_analysis_data("wcst")
    lapse_df = compute_lapse_metrics(trials)
    lapse_df.to_csv(output_dir / "wcst_trial_level_mw_proxies.csv", index=False, encoding="utf-8-sig")

    merged = df.merge(lapse_df, on="participant_id", how="inner")
    metric_cols = [c for c in lapse_df.columns if c not in {"participant_id", "wcst_n_trials"}]

    results = []
    for col in metric_cols:
        res = run_ucla_regression(merged, col, cov_type="HC3")
        if res:
            res["outcome"] = LAPSE_LABELS.get(col, col)
            results.append(res)

    results_df = pd.DataFrame(results)
    if not results_df.empty:
        results_df = add_fdr(results_df)
        results_df = results_df.sort_values("ucla_p")
        results_df.to_csv(output_dir / "wcst_trial_level_mw_regression.csv", index=False, encoding="utf-8-sig")

    return results_df


def run_shift_locked_regressions(
    output_dir: Path,
    trials: pd.DataFrame,
) -> pd.DataFrame:
    df = get_analysis_data("wcst")
    shift_df = compute_shift_locked_metrics(trials)
    shift_df.to_csv(output_dir / "wcst_shift_locked_features.csv", index=False, encoding="utf-8-sig")

    merged = df.merge(shift_df, on="participant_id", how="inner")
    metric_cols = [c for c in shift_df.columns if c not in {"participant_id", "wcst_n_shifts"}]

    results = []
    for col in metric_cols:
        res = run_ucla_regression(merged, col, cov_type="HC3")
        if res:
            res["outcome"] = SHIFT_LABELS.get(col, col)
            results.append(res)

    results_df = pd.DataFrame(results)
    if not results_df.empty:
        results_df = add_fdr(results_df)
        results_df = results_df.sort_values("ucla_p")
        results_df.to_csv(output_dir / "wcst_shift_locked_regression.csv", index=False, encoding="utf-8-sig")

    return results_df


def write_summary(
    output_dir: Path,
    non_hmm: pd.DataFrame,
    lapse: pd.DataFrame,
    shift: pd.DataFrame,
) -> None:
    lines: list[str] = []
    lines.append("WCST Mind-Wandering Proxies (Non-HMM)")
    lines.append("=" * 70)

    def _summarize(df: pd.DataFrame, title: str) -> None:
        lines.append("")
        lines.append(title)
        lines.append("-" * 70)
        if df.empty:
            lines.append("No results.")
            return
        sig = df[df["ucla_fdr_sig"] == True]
        lines.append(f"Total outcomes: {len(df)}")
        lines.append(f"FDR<0.05: {len(sig)}")
        top = df.head(10)
        lines.append("Top (by p):")
        for _, row in top.iterrows():
            p = row["ucla_p"]
            q = row.get("ucla_fdr_q", np.nan)
            lines.append(
                f"- {row['outcome']} | beta={row['ucla_beta']:.4g} | p={p:.4g} | q={q:.4g}"
            )

    _summarize(non_hmm, "Non-HMM outcomes (HC3 + FDR)")
    _summarize(lapse, "Trial-level lapse/run proxies (HC3 + FDR)")
    _summarize(shift, "Shift-locked trajectories (HC3 + FDR)")

    summary_path = output_dir / "wcst_mw_extension_summary.txt"
    summary_path.write_text("\n".join(lines), encoding="utf-8")


def run() -> None:
    output_dir = get_output_dir("wcst")

    trials, _ = load_wcst_trials(apply_trial_filters=True, filter_rt=True)

    non_hmm = run_non_hmm_regressions(output_dir)
    lapse = run_trial_level_regressions(output_dir, trials)
    shift = run_shift_locked_regressions(output_dir, trials)

    write_summary(output_dir, non_hmm, lapse, shift)

    print(f"[OK] Non-HMM regression: {output_dir / 'wcst_non_hmm_hc3.csv'}")
    print(f"[OK] Trial-level features: {output_dir / 'wcst_trial_level_mw_proxies.csv'}")
    print(f"[OK] Trial-level regression: {output_dir / 'wcst_trial_level_mw_regression.csv'}")
    print(f"[OK] Shift-locked features: {output_dir / 'wcst_shift_locked_features.csv'}")
    print(f"[OK] Shift-locked regression: {output_dir / 'wcst_shift_locked_regression.csv'}")
    print(f"[OK] Summary: {output_dir / 'wcst_mw_extension_summary.txt'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="WCST non-HMM extensions for mind-wandering proxies.")
    parser.parse_args()
    run()
