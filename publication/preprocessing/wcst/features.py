"""WCST feature derivation."""

from __future__ import annotations

from pathlib import Path
from typing import List, Dict

import numpy as np
import pandas as pd

from ..core import coefficient_of_variation
from .loaders import load_wcst_trials
from .mechanism import load_or_compute_wcst_mechanism_features


def derive_wcst_features(
    data_dir: None | str | Path = None,
    filter_rt: bool = True,
) -> pd.DataFrame:
    wcst, _ = load_wcst_trials(data_dir=data_dir, filter_rt=filter_rt)

    rt_col = None
    for cand in ("reactionTimeMs", "rt_ms", "reaction_time_ms", "rt"):
        if cand in wcst.columns:
            rt_col = cand
            break
    if not rt_col:
        return pd.DataFrame(columns=["participant_id", "wcst_pes", "wcst_post_switch_error_rate", "wcst_cv_rt", "wcst_trials"])

    trial_col = None
    for cand in ("trialIndex", "trial_index", "trial"):
        if cand in wcst.columns:
            trial_col = cand
            break

    if trial_col:
        wcst = wcst.sort_values(["participant_id", trial_col])

    rule_col = None
    for cand in ("ruleAtThatTime", "rule_at_that_time", "rule_at_time", "rule"):
        if cand in wcst.columns:
            rule_col = cand
            break

    records: List[Dict] = []
    for pid, grp in wcst.groupby("participant_id"):
        grp = grp.reset_index(drop=True)

        pes = np.nan
        if "correct" in grp.columns:
            rt_vals = grp[rt_col].values
            correct = grp["correct"].values
            post_pe_rts = []
            post_correct_rts = []
            for i in range(len(grp) - 1):
                if correct[i] == False:
                    post_pe_rts.append(rt_vals[i + 1])
                elif correct[i] == True:
                    post_correct_rts.append(rt_vals[i + 1])
            if post_pe_rts and post_correct_rts:
                pes = np.mean(post_pe_rts) - np.mean(post_correct_rts)

        post_switch_error_rate = np.nan
        if rule_col and "correct" in grp.columns:
            rules = grp[rule_col].values
            is_correct = grp["correct"].values
            rule_changes = []
            for i in range(1, len(rules)):
                if rules[i] != rules[i - 1]:
                    rule_changes.append(i)

            post_switch_errors = []
            for change_idx in rule_changes:
                window_end = min(change_idx + 5, len(grp))
                post_switch_trials = is_correct[change_idx:window_end]
                if len(post_switch_trials) >= 3:
                    post_switch_errors.append(1 - post_switch_trials.mean())

            if post_switch_errors:
                post_switch_error_rate = np.nanmean(post_switch_errors)

        records.append({
            "participant_id": pid,
            "wcst_pes": pes,
            "wcst_post_switch_error_rate": post_switch_error_rate,
            "wcst_cv_rt": coefficient_of_variation(grp[rt_col].dropna()),
            "wcst_trials": len(grp),
        })

    features_df = pd.DataFrame(records)

    mechanism_df = load_or_compute_wcst_mechanism_features(data_dir=data_dir)
    if mechanism_df.empty:
        return features_df
    if features_df.empty:
        return mechanism_df

    overlap = [c for c in mechanism_df.columns if c != "participant_id" and c in features_df.columns]
    if overlap:
        features_df = features_df.drop(columns=overlap)

    return features_df.merge(mechanism_df, on="participant_id", how="left")
