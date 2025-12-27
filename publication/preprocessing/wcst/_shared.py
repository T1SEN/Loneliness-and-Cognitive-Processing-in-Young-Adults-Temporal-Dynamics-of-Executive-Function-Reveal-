"""Shared helpers for WCST feature derivation."""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from ..constants import get_results_dir
from ..core import ensure_participant_id
from .loaders import load_wcst_trials


def _run_lengths(mask: np.ndarray) -> List[int]:
    lengths: List[int] = []
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


def _load_wcst_summary_metrics(data_dir: Path | None) -> pd.DataFrame:
    if data_dir is None:
        data_dir = get_results_dir("wcst")
    summary_path = Path(data_dir) / "3_cognitive_tests_summary.csv"
    if not summary_path.exists():
        return pd.DataFrame()

    summary = pd.read_csv(summary_path, encoding="utf-8")
    summary = ensure_participant_id(summary)
    if "testName" not in summary.columns:
        return pd.DataFrame()
    summary["testName"] = summary["testName"].astype(str).str.strip().str.lower()
    summary = summary[summary["testName"] == "wcst"].copy()
    if summary.empty:
        return pd.DataFrame()

    rename = {
        "completedCategories": "wcst_completed_categories",
        "totalErrorCount": "wcst_total_errors",
        "totalTrialCount": "wcst_total_trials_summary",
        "totalCorrectCount": "wcst_total_correct_summary",
        "perseverativeResponses": "wcst_perseverative_responses",
        "perseverativeResponsesPercent": "wcst_perseverative_response_percent",
        "perseverativeErrorCount": "wcst_perseverative_errors",
        "nonPerseverativeErrorCount": "wcst_nonperseverative_errors",
        "trialsToCompleteFirstCategory": "wcst_trials_to_first_category",
        "failureToMaintainSet": "wcst_failure_to_maintain_set",
        "conceptualLevelResponses": "wcst_clr_count",
        "conceptualLevelResponsesPercent": "wcst_clr_percent",
        "learningToLearn": "wcst_learning_to_learn",
        "learningToLearnHeatonClrDelta": "wcst_learning_to_learn_heaton_clr_delta",
        "learningEfficiencyDeltaTrials": "wcst_learning_efficiency_delta_trials",
        "trialsToFirstConceptualResp": "wcst_trials_to_first_conceptual_resp",
        "trialsToFirstConceptualResp0": "wcst_trials_to_first_conceptual_resp0",
        "hasFirstCLR": "wcst_has_first_clr",
        "categoryClrPercents": "wcst_category_clr_percents",
    }
    cols = [c for c in rename if c in summary.columns]
    summary = summary[["participant_id"] + cols].rename(columns=rename)
    for col in summary.columns:
        if col == "participant_id" or col == "wcst_category_clr_percents":
            continue
        summary[col] = pd.to_numeric(summary[col], errors="coerce").astype(float)

    def _parse_list(val: object) -> List[float]:
        if isinstance(val, list):
            return [float(x) for x in val]
        if not isinstance(val, str):
            return []
        try:
            parsed = ast.literal_eval(val)
        except Exception:
            return []
        if isinstance(parsed, list):
            return [float(x) for x in parsed]
        return []

    clr_lists = summary["wcst_category_clr_percents"].apply(_parse_list)
    summary["wcst_category_clr_percents"] = clr_lists.apply(lambda x: x if x else np.nan)

    delta_clr = []
    slope_clr = []
    for vals in clr_lists:
        if not vals or len(vals) < 3:
            delta_clr.append(np.nan)
            slope_clr.append(np.nan)
            continue
        first = np.mean(vals[:3])
        last = np.mean(vals[-3:])
        delta_clr.append(float(first - last))
        if len(vals) >= 2:
            x = np.arange(1, len(vals) + 1)
            slope = np.polyfit(x, vals, 1)[0]
            slope_clr.append(float(slope))
        else:
            slope_clr.append(np.nan)

    summary["wcst_delta_clr_percent_first3_last3"] = delta_clr
    summary["wcst_learning_slope_clr_percent"] = slope_clr

    return summary


def prepare_wcst_trials(
    data_dir: None | str | Path = None,
    filter_rt: bool = False,
) -> Dict[str, object]:
    if filter_rt:
        wcst, _ = load_wcst_trials(data_dir=data_dir, clean=True, filter_rt=True, apply_trial_filters=False)
    else:
        wcst, _ = load_wcst_trials(data_dir=data_dir, apply_trial_filters=True)

    rt_col = None
    for cand in ("reactionTimeMs", "rt_ms", "reaction_time_ms", "rt"):
        if cand in wcst.columns:
            rt_col = cand
            break

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

    return {
        "wcst": wcst,
        "rt_col": rt_col,
        "trial_col": trial_col,
        "rule_col": rule_col,
    }
