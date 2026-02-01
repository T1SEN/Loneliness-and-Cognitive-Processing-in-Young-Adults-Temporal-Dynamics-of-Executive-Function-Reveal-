"""Overall feature loader and builder (task-specific QC)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..constants import (
    STROOP_RT_MIN,
    STROOP_RT_MAX,
    WCST_RT_MIN,
    WCST_RT_MAX,
    get_results_dir,
)
from ..core import ensure_participant_id
from .qc import prepare_stroop_trials, prepare_wcst_trials


def derive_overall_features(data_dir: Path | None = None) -> pd.DataFrame:
    if data_dir is None:
        data_dir = get_results_dir("overall")

    features_path = data_dir / "5_overall_features.csv"
    if not features_path.exists():
        return pd.DataFrame()

    features = pd.read_csv(features_path, encoding="utf-8-sig")
    features = ensure_participant_id(features)
    return features


def _label_wcst_phases(
    wcst: pd.DataFrame,
    rule_col: str,
    trial_col: str,
    confirm_len: int = 3,
    n_categories: int = 6,
) -> pd.DataFrame:
    df = wcst.sort_values(["participant_id", trial_col]).copy()
    df["category_num"] = np.nan
    df["phase"] = pd.NA

    for pid, grp in df.groupby("participant_id"):
        grp_sorted = grp.sort_values(trial_col).copy()
        idxs = grp_sorted.index.to_list()
        rules = (
            grp_sorted[rule_col].astype(str).str.lower().replace({"color": "colour"}).to_numpy()
        )
        correct = grp_sorted["correct"].astype(bool).to_numpy()

        change_indices = [i for i in range(1, len(rules)) if rules[i] != rules[i - 1]]
        segment_starts = [0] + change_indices
        segment_ends = change_indices + [len(rules)]

        for cat_idx, (start, end) in enumerate(zip(segment_starts, segment_ends), start=1):
            if cat_idx > n_categories:
                break
            if start >= end:
                continue

            first_correct = None
            for j in range(start, end):
                if correct[j]:
                    first_correct = j
                    break

            reacq_idx = None
            if confirm_len >= 1:
                for j in range(start, end - (confirm_len - 1)):
                    if np.all(correct[j : j + confirm_len]):
                        reacq_idx = j + confirm_len - 1
                        break

            for i in range(start, end):
                row_idx = idxs[i]
                df.at[row_idx, "category_num"] = float(cat_idx)
                if first_correct is None:
                    df.at[row_idx, "phase"] = "exploration"
                elif i < first_correct:
                    df.at[row_idx, "phase"] = "exploration"
                elif reacq_idx is None or i <= reacq_idx:
                    df.at[row_idx, "phase"] = "confirmation"
                else:
                    df.at[row_idx, "phase"] = "exploitation"

    df["phase"] = pd.Categorical(df["phase"], categories=["exploration", "confirmation", "exploitation"])
    return df


def _assign_stroop_segments(stroop: pd.DataFrame) -> pd.DataFrame:
    df = stroop.sort_values(["participant_id", "trial_order"]).copy()

    def _assign_segment(group: pd.DataFrame) -> pd.DataFrame:
        group = group.copy()
        group["segment"] = np.nan
        valid = group["trial_order"].notna()
        if not valid.any():
            return group
        sub = group[valid].sort_values("trial_order").copy()
        n_trials = len(sub)
        if n_trials == 0:
            return group
        positions = np.arange(1, n_trials + 1)
        edges = np.linspace(0, n_trials, 5)
        sub["segment"] = pd.cut(
            positions,
            bins=edges,
            labels=[1, 2, 3, 4],
            include_lowest=True,
        ).astype(float)
        group.loc[sub.index, "segment"] = sub["segment"]
        return group

    return df.groupby("participant_id", group_keys=False).apply(_assign_segment)


def _compute_stroop_interference(stroop: pd.DataFrame) -> pd.Series:
    valid = (
        (stroop["cond"].isin({"congruent", "incongruent"}))
        & (stroop["correct"])
        & (~stroop["timeout"])
        & (stroop["rt_ms"].between(STROOP_RT_MIN, STROOP_RT_MAX))
    )
    subset = stroop[valid].copy()
    means = subset.groupby(["participant_id", "cond"])["rt_ms"].mean().unstack()
    if "incongruent" not in means.columns or "congruent" not in means.columns:
        return pd.Series(dtype=float)
    return means["incongruent"] - means["congruent"]


def _compute_stroop_interference_slope(stroop: pd.DataFrame) -> pd.Series:
    if stroop.empty:
        return pd.Series(dtype=float)

    stroop = _assign_stroop_segments(stroop)
    valid = (
        (stroop["cond"].isin({"congruent", "incongruent"}))
        & (stroop["correct"])
        & (~stroop["timeout"])
        & (stroop["rt_ms"].between(STROOP_RT_MIN, STROOP_RT_MAX))
    )
    subset = stroop[valid].copy()
    seg_means = (
        subset.groupby(["participant_id", "segment", "cond"])["rt_ms"].mean().unstack()
    )
    if "incongruent" not in seg_means.columns or "congruent" not in seg_means.columns:
        return pd.Series(dtype=float)

    seg_means["interference"] = seg_means["incongruent"] - seg_means["congruent"]
    seg_means = seg_means.reset_index()[["participant_id", "segment", "interference"]]

    def _slope(group: pd.DataFrame) -> float:
        group = group.dropna(subset=["interference"])
        if len(group) < 2:
            return np.nan
        x = group["segment"].astype(float).to_numpy()
        y = group["interference"].to_numpy()
        return float(np.polyfit(x, y, 1)[0])

    return seg_means.groupby("participant_id").apply(_slope)


def _compute_wcst_perseverative_error_rate(wcst: pd.DataFrame) -> pd.Series:
    if wcst.empty:
        return pd.Series(dtype=float)
    counts = wcst.groupby("participant_id").size()
    pe_counts = wcst.groupby("participant_id")["isPE"].sum()
    rate = (pe_counts / counts) * 100
    return rate.replace([np.inf, -np.inf], np.nan)


def _compute_wcst_phase_means(wcst: pd.DataFrame) -> pd.DataFrame:
    if wcst.empty:
        return pd.DataFrame(columns=["participant_id"])

    wcst = wcst.copy()
    wcst = wcst.dropna(subset=["trial_order"])
    wcst = _label_wcst_phases(wcst, rule_col="rule", trial_col="trial_order", confirm_len=3)

    rt_valid = (
        wcst["rt_ms"].between(WCST_RT_MIN, WCST_RT_MAX)
        & (~wcst["timeout"])
        & (wcst["phase"].notna())
    )
    rt_df = wcst[rt_valid].copy()
    phase_means = (
        rt_df.groupby(["participant_id", "phase"])["rt_ms"].mean().unstack()
    )
    phase_means = phase_means.rename(
        columns={
            "exploration": "wcst_exploration_rt",
            "confirmation": "wcst_confirmation_rt",
            "exploitation": "wcst_exploitation_rt",
        }
    )

    pre_rt = rt_df[rt_df["phase"].isin(["exploration", "confirmation"])]
    pre_rt_mean = pre_rt.groupby("participant_id")["rt_ms"].mean()
    phase_means["wcst_pre_exploitation_rt"] = pre_rt_mean

    phase_means["wcst_confirmation_minus_exploitation_rt"] = (
        phase_means["wcst_confirmation_rt"] - phase_means["wcst_exploitation_rt"]
    )
    phase_means["wcst_pre_exploitation_minus_exploitation_rt"] = (
        phase_means["wcst_pre_exploitation_rt"] - phase_means["wcst_exploitation_rt"]
    )

    return phase_means.reset_index()


def build_overall_features(
    data_dir: Path | None = None,
    qc_ids: set[str] | None = None,
    save: bool = True,
    verbose: bool = True,
) -> pd.DataFrame:
    if data_dir is None:
        data_dir = get_results_dir("overall")

    stroop = prepare_stroop_trials(data_dir)
    wcst = prepare_wcst_trials(data_dir)

    if qc_ids is None:
        ids_path = data_dir / "filtered_participant_ids.csv"
        if ids_path.exists():
            ids_df = pd.read_csv(ids_path, encoding="utf-8-sig")
            ids_df = ensure_participant_id(ids_df)
            qc_ids = set(ids_df["participant_id"].dropna().astype(str))

    if qc_ids:
        stroop = stroop[stroop["participant_id"].isin(qc_ids)].copy()
        wcst = wcst[wcst["participant_id"].isin(qc_ids)].copy()

    if verbose:
        print(f"  [FEATURES] Stroop trials: {len(stroop)}")
        print(f"  [FEATURES] WCST trials: {len(wcst)}")

    if qc_ids:
        participant_ids = sorted(qc_ids)
    else:
        participant_ids = sorted(set(stroop["participant_id"]) | set(wcst["participant_id"]))
    features = pd.DataFrame({"participant_id": participant_ids})

    # Stroop metrics
    stroop_interference = _compute_stroop_interference(stroop)
    stroop_slope = _compute_stroop_interference_slope(stroop)

    features = features.merge(
        stroop_interference.rename("stroop_interference"),
        on="participant_id",
        how="left",
    )
    features = features.merge(
        stroop_slope.rename("stroop_interference_slope"),
        on="participant_id",
        how="left",
    )

    # WCST metrics
    wcst_pe_rate = _compute_wcst_perseverative_error_rate(wcst)
    features = features.merge(
        wcst_pe_rate.rename("wcst_perseverative_error_rate"),
        on="participant_id",
        how="left",
    )

    phase_means = _compute_wcst_phase_means(wcst)
    if not phase_means.empty:
        features = features.merge(phase_means, on="participant_id", how="left")

    if save:
        output_path = data_dir / "5_overall_features.csv"
        features.to_csv(output_path, index=False, encoding="utf-8-sig")
        if verbose:
            print(f"  [FEATURES] Saved: {output_path}")

    return features
