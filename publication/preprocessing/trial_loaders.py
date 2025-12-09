"""
Common trial-level loaders with standardized preprocessing for PRP, Stroop, WCST.
Outputs are cached (optional) to speed repeated analyses.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Tuple, Dict, Any

import pandas as pd

from .constants import (
    DEFAULT_RT_MIN,
    DEFAULT_RT_MAX,
    PRP_RT_MAX,
    STROOP_RT_MAX,
    DEFAULT_SOA_SHORT,
    DEFAULT_SOA_LONG,
    RESULTS_DIR,
    ANALYSIS_OUTPUT_DIR,
)
from .loaders import ensure_participant_id


CACHE_DIR = ANALYSIS_OUTPUT_DIR / "trial_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _maybe_cache(path: Path, df: pd.DataFrame, use_parquet: bool = True) -> None:
    if use_parquet:
        try:
            df.to_parquet(path, index=False)
            return
        except Exception:
            pass
    df.to_csv(path.with_suffix(".csv"), index=False)


def _load_cached(path: Path):
    if path.exists():
        return pd.read_parquet(path)
    csv_fallback = path.with_suffix(".csv")
    if csv_fallback.exists():
        return pd.read_csv(csv_fallback)
    return None


def load_prp_trials(
    use_cache: bool = True,
    force_rebuild: bool = False,
    rt_min: int = DEFAULT_RT_MIN,
    rt_max: int = PRP_RT_MAX,
    require_t1_correct: bool = True,
    enforce_short_long_only: bool = True,
    require_t2_correct_for_rt: bool = True,
    drop_timeouts: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    cache_path = CACHE_DIR / "prp_trials.parquet"
    if use_cache and not force_rebuild:
        cached = _load_cached(cache_path)
        if cached is not None:
            return cached, {"cached": True}

    df = pd.read_csv(RESULTS_DIR / "4a_prp_trials.csv", encoding="utf-8")
    df = ensure_participant_id(df)

    # Prefer _ms columns (have more data) over legacy columns
    rt_col = "t2_rt_ms" if "t2_rt_ms" in df.columns else "t2_rt" if "t2_rt" in df.columns else None
    if rt_col and rt_col != "t2_rt":
        # Drop the old empty t2_rt column if exists to avoid duplicates
        if "t2_rt" in df.columns:
            df = df.drop(columns=["t2_rt"])
        df = df.rename(columns={rt_col: "t2_rt"})
    # Prefer soa_nominal_ms (has more data) over legacy soa
    soa_col = None
    for cand in ["soa_nominal_ms", "soa_ms", "soa"]:
        if cand in df.columns:
            soa_col = cand
            break
    if soa_col and soa_col != "soa":
        # Drop the old empty soa column if exists to avoid duplicates
        if "soa" in df.columns:
            df = df.drop(columns=["soa"])
        df = df.rename(columns={soa_col: "soa"})
    if "t1_correct" not in df.columns:
        raise KeyError("PRP trials missing t1_correct column")
    if "t2_rt" not in df.columns or "soa" not in df.columns:
        raise KeyError("PRP trials missing t2_rt or soa column")

    # Normalize boolean columns to avoid silent NaN-based row drops
    for bool_col in ["t1_correct", "t2_correct", "t2_timeout"]:
        if bool_col in df.columns:
            df[bool_col] = df[bool_col].fillna(False).astype(bool)

    # Filters
    before = len(df)
    df = df[df["t2_rt"].between(rt_min, rt_max)]
    if drop_timeouts and "t2_timeout" in df.columns:
        df = df[df["t2_timeout"] == False]
    if require_t1_correct:
        df = df[df["t1_correct"] == True]
    if require_t2_correct_for_rt and "t2_correct" in df.columns:
        df = df[df["t2_correct"] == True]

    # SOA binning
    def bin_soa(soa_val):
        if soa_val <= DEFAULT_SOA_SHORT:
            return "short"
        if soa_val >= DEFAULT_SOA_LONG:
            return "long"
        return "other"

    df["soa_bin"] = df["soa"].apply(bin_soa)
    if enforce_short_long_only:
        df = df[df["soa_bin"].isin(["short", "long"])]

    summary = {
        "cached": False,
        "rows_before": before,
        "rows_after": len(df),
        "n_participants": df["participant_id"].nunique(),
        "rt_min": rt_min,
        "rt_max": rt_max,
    }
    _maybe_cache(cache_path, df)
    return df, summary


def load_stroop_trials(
    use_cache: bool = True,
    force_rebuild: bool = False,
    rt_min: int = DEFAULT_RT_MIN,
    rt_max: int = STROOP_RT_MAX,
    require_correct_for_rt: bool = True,
    drop_timeouts: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    cache_path = CACHE_DIR / "stroop_trials.parquet"
    if use_cache and not force_rebuild:
        cached = _load_cached(cache_path)
        if cached is not None:
            return cached, {"cached": True}

    df = pd.read_csv(RESULTS_DIR / "4c_stroop_trials.csv", encoding="utf-8")
    df = ensure_participant_id(df)

    # Prefer rt_ms (has more data) over legacy rt column - same pattern as PRP fix
    rt_col = "rt_ms" if "rt_ms" in df.columns else "rt" if "rt" in df.columns else None
    if not rt_col:
        raise KeyError("Stroop trials missing rt/rt_ms column")
    if rt_col != "rt":
        # Drop the old empty rt column if exists to avoid duplicates
        if "rt" in df.columns:
            df = df.drop(columns=["rt"])
        df = df.rename(columns={rt_col: "rt"})

    cond_col = None
    for cand in ["type", "condition", "cond"]:
        if cand in df.columns:
            cond_col = cand
            break
    if cond_col is None:
        raise KeyError("Stroop trials missing condition column")

    # Normalize boolean columns to avoid silent NaN-based row drops
    for bool_col in ["correct", "timeout"]:
        if bool_col in df.columns:
            df[bool_col] = df[bool_col].fillna(False).astype(bool)

    # Filters
    before = len(df)
    if drop_timeouts and "timeout" in df.columns:
        df = df[df["timeout"] == False]
    if require_correct_for_rt and "correct" in df.columns:
        df = df[df["correct"] == True]
    df = df[df["rt"].between(rt_min, rt_max)]

    summary = {
        "cached": False,
        "rows_before": before,
        "rows_after": len(df),
        "n_participants": df["participant_id"].nunique(),
        "rt_min": rt_min,
        "rt_max": rt_max,
    }
    _maybe_cache(cache_path, df)
    return df, summary


def load_wcst_trials(
    use_cache: bool = True,
    force_rebuild: bool = False,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    cache_path = CACHE_DIR / "wcst_trials.parquet"
    if use_cache and not force_rebuild:
        cached = _load_cached(cache_path)
        if cached is not None:
            return cached, {"cached": True}

    df = pd.read_csv(RESULTS_DIR / "4b_wcst_trials.csv", encoding="utf-8")
    df = ensure_participant_id(df)

    # isPE parsing
    if "isPE" not in df.columns:
        def parse_extra(extra_str):
            if not isinstance(extra_str, str):
                return {}
            try:
                return ast.literal_eval(extra_str)
            except Exception:
                return {}
        df["extra_dict"] = df["extra"].apply(parse_extra) if "extra" in df.columns else {}
        df["isPE"] = df.get("extra_dict", {}).apply(lambda x: x.get("isPE", False) if isinstance(x, dict) else False)

    summary = {
        "cached": False,
        "rows_before": len(df),
        "rows_after": len(df),
        "n_participants": df["participant_id"].nunique(),
    }
    _maybe_cache(cache_path, df)
    return df, summary
