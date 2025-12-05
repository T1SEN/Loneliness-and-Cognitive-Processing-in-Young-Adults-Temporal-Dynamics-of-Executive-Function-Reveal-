"""
Trial-Level Feature Derivation Utilities
=========================================

Consolidated module for deriving participant-level aggregates from trial data.

Consolidates:
- derive_trial_features.py
- derive_additional_trial_features.py

Usage:
    from analysis.utils.trial_features import derive_all_features, derive_prp_features

    # Get all features
    features = derive_all_features(use_cache=True)

    # Or individual task features
    prp_features = derive_prp_features()
    stroop_features = derive_stroop_features()
    wcst_features = derive_wcst_features()
"""

from __future__ import annotations

import sys
if sys.platform.startswith("win") and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding='utf-8')

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import warnings

import numpy as np
import pandas as pd

from analysis.utils.trial_data_loader import (
    load_prp_trials,
    load_stroop_trials,
    load_wcst_trials,
)
from analysis.utils.data_loader_utils import ANALYSIS_OUTPUT_DIR

warnings.filterwarnings("ignore")

# Cache location
CACHE_DIR = ANALYSIS_OUTPUT_DIR / "trial_features_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
CACHE_PATH = CACHE_DIR / "all_trial_features.parquet"


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def coefficient_of_variation(series: pd.Series, min_n: int = 3) -> float:
    """Compute coefficient of variation (CV = std/mean)."""
    if len(series) < min_n or series.mean() == 0:
        return np.nan
    return series.std(ddof=1) / series.mean()


def compute_post_error_slowing(
    df: pd.DataFrame,
    rt_col: str,
    correct_col: str,
    participant_col: str = "participant_id",
) -> pd.DataFrame:
    """
    Compute post-error slowing (PES) for each participant.

    PES = mean(RT after error) - mean(RT after correct)
    """
    records = []
    for pid, grp in df.groupby(participant_col):
        grp = grp.copy().sort_index()
        grp["prev_correct"] = grp[correct_col].shift(1)

        post_error = grp[grp["prev_correct"] == False]
        post_correct = grp[grp["prev_correct"] == True]

        post_error_mean = post_error[rt_col].mean() if len(post_error) > 0 else np.nan
        post_correct_mean = post_correct[rt_col].mean() if len(post_correct) > 0 else np.nan

        pes = (
            post_error_mean - post_correct_mean
            if pd.notna(post_error_mean) and pd.notna(post_correct_mean)
            else np.nan
        )

        records.append({
            participant_col: pid,
            "pes": pes,
            "post_error_rt": post_error_mean,
            "post_correct_rt": post_correct_mean,
        })

    return pd.DataFrame(records)


# =============================================================================
# PRP FEATURES
# =============================================================================

def derive_prp_features(
    use_cache: bool = True,
    rt_min: float = 200,
    rt_max: float = 5000,
) -> pd.DataFrame:
    """
    Derive PRP trial-level features.

    Features:
    - prp_t2_cv_all: Overall T2 RT coefficient of variation
    - prp_t2_cv_short: T2 CV at short SOA (≤150ms)
    - prp_t2_cv_long: T2 CV at long SOA (≥600ms)
    - prp_cascade_rate: Rate of T1 error → T2 error cascades
    - prp_cascade_inflation: Cascade rate / baseline T2 error rate
    - prp_pes: Post-error slowing (T2 RT)
    - prp_t2_trials: Number of valid T2 trials
    """
    prp, _ = load_prp_trials(
        use_cache=use_cache,
        rt_min=rt_min,
        rt_max=rt_max,
        require_t1_correct=False,
        require_t2_correct_for_rt=False,
        enforce_short_long_only=False,
        drop_timeouts=True,
    )

    # Ensure numeric columns
    prp["t2_rt_ms"] = pd.to_numeric(prp.get("t2_rt", prp.get("t2_rt_ms")), errors="coerce")
    prp["soa_ms"] = pd.to_numeric(prp.get("soa", prp.get("soa_nominal_ms")), errors="coerce")
    prp = prp[(prp["t2_rt_ms"].notna()) & (prp["t2_rt_ms"] > 0)]

    # Normalize boolean columns
    for col in ["t1_correct", "t2_correct"]:
        if col in prp.columns and prp[col].dtype == "object":
            prp[col] = prp[col].map({
                True: True, "True": True, 1: True,
                False: False, "False": False, 0: False
            })

    records: List[Dict] = []
    for pid, group in prp.groupby("participant_id"):
        # CV metrics
        overall_cv = coefficient_of_variation(group["t2_rt_ms"])
        short_soa = group[group["soa_ms"] <= 150]
        long_soa = group[group["soa_ms"] >= 600]

        # Error cascade metrics
        if "t1_correct" in group.columns and "t2_correct" in group.columns:
            valid = group[(group["t1_correct"].notna()) & (group["t2_correct"].notna())]
            cascade = (~valid["t1_correct"]) & (~valid["t2_correct"])
            cascade_rate = cascade.mean() if len(valid) > 0 else np.nan
            t2_error_rate = 1 - valid["t2_correct"].mean() if len(valid) > 0 else np.nan
            cascade_inflation = cascade_rate / t2_error_rate if t2_error_rate and t2_error_rate > 0 else np.nan
        else:
            cascade_rate = np.nan
            cascade_inflation = np.nan

        # Post-error slowing
        grp_sorted = group.sort_values("trial_index" if "trial_index" in group.columns else "trial")
        if "t2_correct" in grp_sorted.columns:
            grp_sorted["prev_t2_error"] = (~grp_sorted["t2_correct"]).shift(1).fillna(False)
            post_error = grp_sorted[grp_sorted["prev_t2_error"] == True]["t2_rt_ms"]
            post_correct = grp_sorted[grp_sorted["prev_t2_error"] == False]["t2_rt_ms"]
            pes = (
                post_error.mean() - post_correct.mean()
                if len(post_error) > 0 and len(post_correct) > 0
                else np.nan
            )
        else:
            pes = np.nan

        records.append({
            "participant_id": pid,
            "prp_t2_cv_all": overall_cv,
            "prp_t2_cv_short": coefficient_of_variation(short_soa["t2_rt_ms"]),
            "prp_t2_cv_long": coefficient_of_variation(long_soa["t2_rt_ms"]),
            "prp_cascade_rate": cascade_rate,
            "prp_cascade_inflation": cascade_inflation,
            "prp_pes": pes,
            "prp_t2_trials": len(group),
        })

    return pd.DataFrame(records)


# =============================================================================
# STROOP FEATURES
# =============================================================================

def derive_stroop_features(
    use_cache: bool = True,
    rt_min: float = 200,
    rt_max: float = 3000,
) -> pd.DataFrame:
    """
    Derive Stroop trial-level features.

    Features:
    - stroop_post_error_slowing: PES (post-error RT - post-correct RT)
    - stroop_post_error_rt: Mean RT after error trials
    - stroop_post_correct_rt: Mean RT after correct trials
    - stroop_incong_slope: RT slope across incongruent trials (fatigue/learning)
    - stroop_cv_all: Overall RT coefficient of variation
    - stroop_cv_incong: CV for incongruent trials
    - stroop_cv_cong: CV for congruent trials
    - stroop_trials: Number of valid trials
    """
    stroop, _ = load_stroop_trials(
        use_cache=use_cache,
        rt_min=rt_min,
        rt_max=rt_max,
        drop_timeouts=True,
        require_correct_for_rt=False,
    )

    # Find RT column
    rt_col = "rt" if "rt" in stroop.columns else "rt_ms"
    stroop["rt_ms"] = pd.to_numeric(stroop[rt_col], errors="coerce")
    stroop = stroop[stroop["rt_ms"].notna()]

    # Find trial index column
    trial_col = None
    for cand in ("trial", "trialIndex", "idx", "trial_index"):
        if cand in stroop.columns:
            trial_col = cand
            break
    if trial_col:
        stroop["trial_order"] = pd.to_numeric(stroop[trial_col], errors="coerce")
        stroop = stroop.sort_values(["participant_id", "trial_order"])

    # Find condition column
    cond_col = None
    for cand in ("type", "condition", "cond", "congruency"):
        if cand in stroop.columns:
            cond_col = cand
            break

    records: List[Dict] = []
    for pid, group in stroop.groupby("participant_id"):
        grp = group.copy()

        # Post-error slowing
        if "correct" in grp.columns:
            grp["prev_correct"] = grp["correct"].shift(1)
            post_error = grp[grp["prev_correct"] == False]
            post_correct = grp[grp["prev_correct"] == True]
            post_error_mean = post_error["rt_ms"].mean() if len(post_error) > 0 else np.nan
            post_correct_mean = post_correct["rt_ms"].mean() if len(post_correct) > 0 else np.nan
            pes = (
                post_error_mean - post_correct_mean
                if pd.notna(post_error_mean) and pd.notna(post_correct_mean)
                else np.nan
            )
        else:
            pes = np.nan
            post_error_mean = np.nan
            post_correct_mean = np.nan

        # RT slope for incongruent trials
        slope = np.nan
        cv_incong = np.nan
        cv_cong = np.nan
        if cond_col and "trial_order" in grp.columns:
            cond_lower = grp[cond_col].astype(str).str.lower()
            incong = grp[cond_lower == "incongruent"]
            cong = grp[cond_lower == "congruent"]

            if len(incong) >= 5 and incong["trial_order"].nunique() > 1:
                x = incong["trial_order"].values
                y = incong["rt_ms"].values
                coef = np.polyfit(x, y, 1)
                slope = coef[0]

            cv_incong = coefficient_of_variation(incong["rt_ms"])
            cv_cong = coefficient_of_variation(cong["rt_ms"])

        records.append({
            "participant_id": pid,
            "stroop_post_error_slowing": pes,
            "stroop_post_error_rt": post_error_mean,
            "stroop_post_correct_rt": post_correct_mean,
            "stroop_incong_slope": slope,
            "stroop_cv_all": coefficient_of_variation(grp["rt_ms"]),
            "stroop_cv_incong": cv_incong,
            "stroop_cv_cong": cv_cong,
            "stroop_trials": len(grp),
        })

    return pd.DataFrame(records)


# =============================================================================
# WCST FEATURES
# =============================================================================

def derive_wcst_features(use_cache: bool = True) -> pd.DataFrame:
    """
    Derive WCST trial-level features.

    Features:
    - wcst_pes: Post-error slowing
    - wcst_post_switch_error_rate: Error rate in trials following rule switch
    - wcst_cv_rt: RT coefficient of variation
    - wcst_trials: Number of valid trials
    """
    wcst, _ = load_wcst_trials(use_cache=use_cache)

    # Find RT column
    rt_col = None
    for cand in ("reactionTimeMs", "rt_ms", "reaction_time_ms", "rt"):
        if cand in wcst.columns:
            rt_col = cand
            break
    if not rt_col:
        # Return empty dataframe if no RT column
        return pd.DataFrame(columns=["participant_id", "wcst_pes", "wcst_post_switch_error_rate", "wcst_cv_rt", "wcst_trials"])

    # Find trial index column
    trial_col = None
    for cand in ("trialIndex", "trial_index", "trial"):
        if cand in wcst.columns:
            trial_col = cand
            break

    if trial_col:
        wcst = wcst.sort_values(["participant_id", trial_col])

    # Find rule column
    rule_col = None
    for cand in ("ruleAtThatTime", "rule_at_that_time", "rule_at_time", "rule"):
        if cand in wcst.columns:
            rule_col = cand
            break

    records: List[Dict] = []
    for pid, grp in wcst.groupby("participant_id"):
        grp = grp.reset_index(drop=True)

        # Post-error slowing
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

        # Post-switch error rate
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

    return pd.DataFrame(records)


# =============================================================================
# COMBINED FEATURE DERIVATION
# =============================================================================

def derive_all_features(
    use_cache: bool = True,
    force_rebuild: bool = False,
) -> pd.DataFrame:
    """
    Derive all trial-level features and merge into single dataframe.

    Parameters
    ----------
    use_cache : bool
        If True, load from cache if available
    force_rebuild : bool
        If True, rebuild features even if cache exists

    Returns
    -------
    pd.DataFrame
        Combined features with columns from all tasks
    """
    if use_cache and not force_rebuild and CACHE_PATH.exists():
        return pd.read_parquet(CACHE_PATH)

    print("Deriving trial-level features...")

    prp_features = derive_prp_features(use_cache=use_cache)
    print(f"  PRP features: {len(prp_features)} participants")

    stroop_features = derive_stroop_features(use_cache=use_cache)
    print(f"  Stroop features: {len(stroop_features)} participants")

    wcst_features = derive_wcst_features(use_cache=use_cache)
    print(f"  WCST features: {len(wcst_features)} participants")

    # Merge all features
    features = prp_features.merge(stroop_features, on="participant_id", how="outer")
    features = features.merge(wcst_features, on="participant_id", how="outer")

    # Cache result
    features.to_parquet(CACHE_PATH, index=False)
    print(f"  Cached to: {CACHE_PATH}")

    return features


# =============================================================================
# CLI INTERFACE
# =============================================================================

def main():
    """Command-line interface for feature derivation."""
    import argparse

    parser = argparse.ArgumentParser(description="Derive trial-level features")
    parser.add_argument("--task", choices=["prp", "stroop", "wcst", "all"], default="all")
    parser.add_argument("--force-rebuild", action="store_true")
    parser.add_argument("--output", type=str, help="Output CSV path")
    args = parser.parse_args()

    if args.task == "prp":
        features = derive_prp_features()
    elif args.task == "stroop":
        features = derive_stroop_features()
    elif args.task == "wcst":
        features = derive_wcst_features()
    else:
        features = derive_all_features(force_rebuild=args.force_rebuild)

    if args.output:
        features.to_csv(args.output, index=False, encoding="utf-8-sig")
        print(f"Saved to: {args.output}")
    else:
        print(features.head(10).to_string())
        print(f"\nTotal: {len(features)} participants, {len(features.columns)} features")


if __name__ == "__main__":
    main()
