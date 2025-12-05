"""
Post-Error Analysis Utilities
=============================

Provides functions for computing post-error slowing (PES) and related metrics.

Consolidates:
- prp_post_error_adjustments.py
- stroop_post_error_adjustments.py
- wcst_post_error_adaptation_quick.py
- post_error_slowing_integrated.py
- post_error_slowing_gender_moderation.py

Usage:
    from analysis.statistics import compute_pes, compute_post_error_accuracy

    # Compute PES for each participant
    pes_df = compute_pes(trials_df, rt_col='rt_ms', correct_col='correct')

    # Compute post-error accuracy change
    pea_df = compute_post_error_accuracy(trials_df, correct_col='correct')
"""

from __future__ import annotations

import sys
if sys.platform.startswith("win") and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding='utf-8')

from typing import Dict, List, Optional, Union
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


# =============================================================================
# CORE PES COMPUTATION
# =============================================================================

def compute_pes(
    trials_df: pd.DataFrame,
    rt_col: str,
    correct_col: str,
    participant_col: str = "participant_id",
    trial_col: Optional[str] = None,
    min_post_error: int = 3,
    min_post_correct: int = 3,
) -> pd.DataFrame:
    """
    Compute Post-Error Slowing (PES) for each participant.

    PES = mean(RT on trials following errors) - mean(RT on trials following correct)

    Parameters
    ----------
    trials_df : pd.DataFrame
        Trial-level data sorted by trial order
    rt_col : str
        Name of reaction time column
    correct_col : str
        Name of correct/incorrect column (True/False or 1/0)
    participant_col : str
        Name of participant ID column
    trial_col : str, optional
        Name of trial index column for sorting
    min_post_error : int
        Minimum post-error trials required
    min_post_correct : int
        Minimum post-correct trials required

    Returns
    -------
    pd.DataFrame
        Columns: participant_id, pes, post_error_rt, post_correct_rt,
                 n_post_error, n_post_correct
    """
    df = trials_df.copy()

    # Sort by trial order if column specified
    if trial_col and trial_col in df.columns:
        df = df.sort_values([participant_col, trial_col])

    records = []
    for pid, grp in df.groupby(participant_col):
        grp = grp.reset_index(drop=True)

        rt_vals = grp[rt_col].values
        correct_vals = grp[correct_col].astype(bool).values

        post_error_rts = []
        post_correct_rts = []

        for i in range(1, len(grp)):
            if pd.notna(rt_vals[i]) and rt_vals[i] > 0:
                if correct_vals[i - 1] == False:
                    post_error_rts.append(rt_vals[i])
                elif correct_vals[i - 1] == True:
                    post_correct_rts.append(rt_vals[i])

        # Compute means
        post_error_mean = np.mean(post_error_rts) if len(post_error_rts) >= min_post_error else np.nan
        post_correct_mean = np.mean(post_correct_rts) if len(post_correct_rts) >= min_post_correct else np.nan

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
            "n_post_error": len(post_error_rts),
            "n_post_correct": len(post_correct_rts),
        })

    return pd.DataFrame(records)


def compute_post_error_accuracy(
    trials_df: pd.DataFrame,
    correct_col: str,
    participant_col: str = "participant_id",
    trial_col: Optional[str] = None,
    min_trials: int = 5,
) -> pd.DataFrame:
    """
    Compute Post-Error Accuracy (PEA) for each participant.

    PEA = accuracy on trials following errors - accuracy on trials following correct

    Parameters
    ----------
    trials_df : pd.DataFrame
        Trial-level data
    correct_col : str
        Name of correct/incorrect column
    participant_col : str
        Name of participant ID column
    trial_col : str, optional
        Name of trial index column for sorting
    min_trials : int
        Minimum trials required in each category

    Returns
    -------
    pd.DataFrame
        Columns: participant_id, pea, post_error_acc, post_correct_acc
    """
    df = trials_df.copy()

    if trial_col and trial_col in df.columns:
        df = df.sort_values([participant_col, trial_col])

    records = []
    for pid, grp in df.groupby(participant_col):
        grp = grp.reset_index(drop=True)

        correct_vals = grp[correct_col].astype(bool).values

        post_error_correct = []
        post_correct_correct = []

        for i in range(1, len(grp)):
            if correct_vals[i - 1] == False:
                post_error_correct.append(correct_vals[i])
            elif correct_vals[i - 1] == True:
                post_correct_correct.append(correct_vals[i])

        post_error_acc = np.mean(post_error_correct) if len(post_error_correct) >= min_trials else np.nan
        post_correct_acc = np.mean(post_correct_correct) if len(post_correct_correct) >= min_trials else np.nan

        pea = (
            post_error_acc - post_correct_acc
            if pd.notna(post_error_acc) and pd.notna(post_correct_acc)
            else np.nan
        )

        records.append({
            participant_col: pid,
            "pea": pea,
            "post_error_acc": post_error_acc,
            "post_correct_acc": post_correct_acc,
            "n_post_error_trials": len(post_error_correct),
            "n_post_correct_trials": len(post_correct_correct),
        })

    return pd.DataFrame(records)


# =============================================================================
# ROBUST PES MEASURES
# =============================================================================

def compute_pes_robust(
    trials_df: pd.DataFrame,
    rt_col: str,
    correct_col: str,
    participant_col: str = "participant_id",
    trial_col: Optional[str] = None,
) -> pd.DataFrame:
    """
    Compute robust PES using multiple methods.

    Methods:
    1. Traditional: post-error RT - post-correct RT
    2. Pre-error baseline: post-error RT - pre-error RT (controls for slow RTs preceding errors)
    3. Matched: post-error RT - RT at matched position in correct sequences

    Returns
    -------
    pd.DataFrame
        Columns: participant_id, pes_traditional, pes_pre_error, pes_median
    """
    df = trials_df.copy()

    if trial_col and trial_col in df.columns:
        df = df.sort_values([participant_col, trial_col])

    records = []
    for pid, grp in df.groupby(participant_col):
        grp = grp.reset_index(drop=True)

        rt_vals = grp[rt_col].values
        correct_vals = grp[correct_col].astype(bool).values

        post_error_rts = []
        post_correct_rts = []
        pre_error_rts = []

        for i in range(1, len(grp) - 1):
            if pd.notna(rt_vals[i]) and rt_vals[i] > 0:
                # Post-error/post-correct
                if correct_vals[i - 1] == False:
                    post_error_rts.append(rt_vals[i])
                    if pd.notna(rt_vals[i - 1]):
                        pre_error_rts.append(rt_vals[i - 1])
                elif correct_vals[i - 1] == True:
                    post_correct_rts.append(rt_vals[i])

        # Traditional PES
        pes_trad = (
            np.mean(post_error_rts) - np.mean(post_correct_rts)
            if post_error_rts and post_correct_rts
            else np.nan
        )

        # Pre-error corrected PES
        pes_pre = (
            np.mean(post_error_rts) - np.mean(pre_error_rts)
            if post_error_rts and pre_error_rts
            else np.nan
        )

        # Median-based PES (more robust to outliers)
        pes_median = (
            np.median(post_error_rts) - np.median(post_correct_rts)
            if post_error_rts and post_correct_rts
            else np.nan
        )

        records.append({
            participant_col: pid,
            "pes_traditional": pes_trad,
            "pes_pre_error": pes_pre,
            "pes_median": pes_median,
            "n_errors": len(post_error_rts),
        })

    return pd.DataFrame(records)


# =============================================================================
# TASK-SPECIFIC PES
# =============================================================================

def compute_prp_pes(
    prp_trials: pd.DataFrame,
    participant_col: str = "participant_id",
) -> pd.DataFrame:
    """
    Compute PES specifically for PRP task (T2 RT after T2 error).

    Parameters
    ----------
    prp_trials : pd.DataFrame
        PRP trial data with t2_rt and t2_correct columns

    Returns
    -------
    pd.DataFrame
        PRP-specific PES metrics
    """
    # Find T2 RT column
    rt_col = None
    for cand in ("t2_rt", "t2_rt_ms", "T2_RT"):
        if cand in prp_trials.columns:
            rt_col = cand
            break

    # Find T2 correct column
    correct_col = None
    for cand in ("t2_correct", "T2_correct", "t2_accuracy"):
        if cand in prp_trials.columns:
            correct_col = cand
            break

    if not rt_col or not correct_col:
        raise ValueError("PRP data missing t2_rt or t2_correct columns")

    pes_df = compute_pes(
        prp_trials,
        rt_col=rt_col,
        correct_col=correct_col,
        participant_col=participant_col,
    )

    # Rename columns with prp_ prefix
    rename_map = {
        "pes": "prp_pes",
        "post_error_rt": "prp_post_error_t2_rt",
        "post_correct_rt": "prp_post_correct_t2_rt",
        "n_post_error": "prp_n_post_t2_error",
        "n_post_correct": "prp_n_post_t2_correct",
    }
    pes_df = pes_df.rename(columns=rename_map)

    return pes_df


def compute_stroop_pes(
    stroop_trials: pd.DataFrame,
    participant_col: str = "participant_id",
) -> pd.DataFrame:
    """
    Compute PES for Stroop task.

    Parameters
    ----------
    stroop_trials : pd.DataFrame
        Stroop trial data with rt and correct columns

    Returns
    -------
    pd.DataFrame
        Stroop-specific PES metrics
    """
    # Find RT column
    rt_col = None
    for cand in ("rt", "rt_ms", "reaction_time"):
        if cand in stroop_trials.columns:
            rt_col = cand
            break

    # Find correct column
    correct_col = None
    for cand in ("correct", "is_correct", "accuracy"):
        if cand in stroop_trials.columns:
            correct_col = cand
            break

    if not rt_col or not correct_col:
        raise ValueError("Stroop data missing rt or correct columns")

    pes_df = compute_pes(
        stroop_trials,
        rt_col=rt_col,
        correct_col=correct_col,
        participant_col=participant_col,
    )

    # Rename columns with stroop_ prefix
    rename_map = {
        "pes": "stroop_pes",
        "post_error_rt": "stroop_post_error_rt",
        "post_correct_rt": "stroop_post_correct_rt",
        "n_post_error": "stroop_n_post_error",
        "n_post_correct": "stroop_n_post_correct",
    }
    pes_df = pes_df.rename(columns=rename_map)

    return pes_df


def compute_wcst_pes(
    wcst_trials: pd.DataFrame,
    participant_col: str = "participant_id",
) -> pd.DataFrame:
    """
    Compute PES for WCST task.

    Parameters
    ----------
    wcst_trials : pd.DataFrame
        WCST trial data with RT and correct columns

    Returns
    -------
    pd.DataFrame
        WCST-specific PES metrics
    """
    # Find RT column
    rt_col = None
    for cand in ("reactionTimeMs", "rt_ms", "rt", "reaction_time"):
        if cand in wcst_trials.columns:
            rt_col = cand
            break

    # Find correct column
    correct_col = None
    for cand in ("correct", "is_correct", "isCorrect"):
        if cand in wcst_trials.columns:
            correct_col = cand
            break

    if not rt_col or not correct_col:
        raise ValueError("WCST data missing rt or correct columns")

    pes_df = compute_pes(
        wcst_trials,
        rt_col=rt_col,
        correct_col=correct_col,
        participant_col=participant_col,
    )

    # Rename columns with wcst_ prefix
    rename_map = {
        "pes": "wcst_pes",
        "post_error_rt": "wcst_post_error_rt",
        "post_correct_rt": "wcst_post_correct_rt",
        "n_post_error": "wcst_n_post_error",
        "n_post_correct": "wcst_n_post_correct",
    }
    pes_df = pes_df.rename(columns=rename_map)

    return pes_df


# =============================================================================
# CROSS-TASK PES
# =============================================================================

def compute_all_task_pes(
    prp_trials: Optional[pd.DataFrame] = None,
    stroop_trials: Optional[pd.DataFrame] = None,
    wcst_trials: Optional[pd.DataFrame] = None,
    participant_col: str = "participant_id",
) -> pd.DataFrame:
    """
    Compute PES for all available tasks and merge.

    Returns
    -------
    pd.DataFrame
        Combined PES metrics across tasks
    """
    dfs = []

    if prp_trials is not None:
        try:
            prp_pes = compute_prp_pes(prp_trials, participant_col)
            dfs.append(prp_pes)
        except Exception as e:
            print(f"Warning: Could not compute PRP PES: {e}")

    if stroop_trials is not None:
        try:
            stroop_pes = compute_stroop_pes(stroop_trials, participant_col)
            dfs.append(stroop_pes)
        except Exception as e:
            print(f"Warning: Could not compute Stroop PES: {e}")

    if wcst_trials is not None:
        try:
            wcst_pes = compute_wcst_pes(wcst_trials, participant_col)
            dfs.append(wcst_pes)
        except Exception as e:
            print(f"Warning: Could not compute WCST PES: {e}")

    if not dfs:
        return pd.DataFrame(columns=[participant_col])

    result = dfs[0]
    for df in dfs[1:]:
        result = result.merge(df, on=participant_col, how="outer")

    return result


# =============================================================================
# CLI
# =============================================================================

def main():
    """Demo PES computation."""
    print("Post-Error Slowing Utilities Demo")
    print("=" * 50)

    # Generate synthetic trial data
    np.random.seed(42)
    n_participants = 10
    n_trials = 100

    records = []
    for pid in range(n_participants):
        for trial in range(n_trials):
            # Simulate: errors have slower preceding RTs, post-error has slowdown
            is_correct = np.random.random() > 0.15  # 15% error rate
            base_rt = 500 + np.random.normal(0, 50)

            # Add some PES effect
            if trial > 0 and records[-1]["correct"] == False:
                base_rt += 30  # Post-error slowing

            records.append({
                "participant_id": f"P{pid:02d}",
                "trial": trial,
                "rt": max(200, base_rt),
                "correct": is_correct,
            })

    df = pd.DataFrame(records)

    # Compute PES
    pes_df = compute_pes(df, rt_col="rt", correct_col="correct")
    print("\nPES Results:")
    print(pes_df.to_string(index=False))

    print(f"\nMean PES: {pes_df['pes'].mean():.1f} ms")
    print(f"Expected: ~30 ms (simulated effect)")


if __name__ == "__main__":
    main()
