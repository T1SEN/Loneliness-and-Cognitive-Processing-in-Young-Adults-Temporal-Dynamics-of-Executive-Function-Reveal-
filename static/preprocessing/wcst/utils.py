"""
WCST phase utilities for overall analyses.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import pandas as pd

from ..constants import get_results_dir
from ..core import ensure_participant_id


def _coerce_bool_series(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    mapped = series.astype(str).str.strip().str.lower().map(
        {"true": True, "1": True, "yes": True, "false": False, "0": False, "no": False}
    )
    return mapped.fillna(False)


def _read_trials_csv(data_dir: Path) -> pd.DataFrame:
    trials_path = data_dir / "4b_wcst_trials.csv"
    if not trials_path.exists():
        return pd.DataFrame()
    df = pd.read_csv(trials_path, encoding="utf-8-sig")
    if df.empty:
        return df
    df = ensure_participant_id(df)
    return df


def prepare_wcst_trials(data_dir: Path | None = None) -> Dict[str, Any]:
    """
    Load WCST trials from overall-complete data.

    Returns a dict compatible with legacy scripts:
        - wcst: trial-level dataframe
        - rt_col: column name for RT (ms)
        - trial_col: column name for trial order
        - rule_col: column name for rule at that time
    """
    if data_dir is None:
        data_dir = get_results_dir("overall")

    wcst = _read_trials_csv(data_dir)
    if wcst.empty:
        return {"wcst": wcst, "rt_col": None, "trial_col": None, "rule_col": None}

    rt_col = None
    for candidate in ["rt_ms", "reactionTimeMs", "resp_time_ms"]:
        if candidate in wcst.columns:
            rt_col = candidate
            break

    trial_col = None
    for candidate in ["trial_index", "trialIndex", "trial"]:
        if candidate in wcst.columns:
            trial_col = candidate
            break

    rule_col = None
    for candidate in ["ruleAtThatTime", "rule_at_that_time", "rule"]:
        if candidate in wcst.columns:
            rule_col = candidate
            break

    for col in ["correct", "isPE", "isNPE", "timeout", "is_rt_valid"]:
        if col in wcst.columns:
            wcst[col] = _coerce_bool_series(wcst[col]).astype(bool)

    return {
        "wcst": wcst,
        "rt_col": rt_col,
        "trial_col": trial_col,
        "rule_col": rule_col,
    }
