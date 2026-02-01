"""
WCST feature derivation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..constants import WCST_RT_MIN, WCST_RT_MAX
from .phase import label_wcst_phases


def compute_wcst_perseverative_error_rate(wcst: pd.DataFrame) -> pd.Series:
    if wcst.empty:
        return pd.Series(dtype=float)
    counts = wcst.groupby("participant_id").size()
    pe_counts = wcst.groupby("participant_id")["isPE"].sum()
    rate = (pe_counts / counts) * 100
    return rate.replace([np.inf, -np.inf], np.nan)


def compute_wcst_phase_means(wcst: pd.DataFrame) -> pd.DataFrame:
    if wcst.empty:
        return pd.DataFrame(columns=["participant_id"])

    wcst = wcst.copy()
    wcst = wcst.dropna(subset=["trial_order"])
    wcst = label_wcst_phases(wcst, rule_col="rule", trial_col="trial_order", confirm_len=3)

    rt_valid = (
        wcst["rt_ms"].between(WCST_RT_MIN, WCST_RT_MAX)
        & (~wcst["timeout"])
        & (wcst["phase"].notna())
    )
    rt_df = wcst[rt_valid].copy()

    # --- correct-only phase means (primary) ---
    rt_correct = rt_df[rt_df["correct"].astype(bool)]
    phase_means = (
        rt_correct.groupby(["participant_id", "phase"])["rt_ms"].mean().unstack()
    )
    phase_means = phase_means.rename(
        columns={
            "exploration": "wcst_exploration_rt",
            "confirmation": "wcst_confirmation_rt",
            "exploitation": "wcst_exploitation_rt",
        }
    )

    pre_rt = rt_correct[rt_correct["phase"].isin(["exploration", "confirmation"])]
    pre_rt_mean = pre_rt.groupby("participant_id")["rt_ms"].mean()
    phase_means["wcst_pre_exploitation_rt"] = pre_rt_mean

    phase_means["wcst_confirmation_minus_exploitation_rt"] = (
        phase_means["wcst_confirmation_rt"] - phase_means["wcst_exploitation_rt"]
    )
    phase_means["wcst_pre_exploitation_minus_exploitation_rt"] = (
        phase_means["wcst_pre_exploitation_rt"] - phase_means["wcst_exploitation_rt"]
    )

    # --- all-trials phase means (including errors) ---
    phase_all = rt_df.groupby(["participant_id", "phase"])["rt_ms"].mean().unstack()
    phase_all = phase_all.rename(
        columns={
            "exploration": "wcst_exploration_rt_all",
            "confirmation": "wcst_confirmation_rt_all",
            "exploitation": "wcst_exploitation_rt_all",
        }
    )

    pre_rt_all = rt_df[rt_df["phase"].isin(["exploration", "confirmation"])]
    pre_rt_all_mean = pre_rt_all.groupby("participant_id")["rt_ms"].mean()
    phase_all["wcst_pre_exploitation_rt_all"] = pre_rt_all_mean

    phase_all["wcst_confirmation_minus_exploitation_rt_all"] = (
        phase_all["wcst_confirmation_rt_all"] - phase_all["wcst_exploitation_rt_all"]
    )
    phase_all["wcst_pre_exploitation_minus_exploitation_rt_all"] = (
        phase_all["wcst_pre_exploitation_rt_all"] - phase_all["wcst_exploitation_rt_all"]
    )

    result = phase_means.join(phase_all, how="outer")
    return result.reset_index()


def build_wcst_features(wcst: pd.DataFrame) -> pd.DataFrame:
    if wcst.empty:
        return pd.DataFrame(columns=["participant_id"])

    participant_ids = sorted(wcst["participant_id"].dropna().astype(str).unique())
    features = pd.DataFrame({"participant_id": participant_ids})

    pe_rate = compute_wcst_perseverative_error_rate(wcst)
    features = features.merge(
        pe_rate.rename("wcst_perseverative_error_rate"),
        on="participant_id",
        how="left",
    )

    phase_means = compute_wcst_phase_means(wcst)
    if not phase_means.empty:
        features = features.merge(phase_means, on="participant_id", how="left")

    return features
