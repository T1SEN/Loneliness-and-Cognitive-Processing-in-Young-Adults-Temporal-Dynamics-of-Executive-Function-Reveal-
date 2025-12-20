"""PRP bottleneck mechanism feature derivation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

from ..constants import PRP_RT_MIN, PRP_RT_MAX, get_results_dir
from .loaders import load_prp_trials

MECHANISM_FILENAME = "5_prp_bottleneck_mechanism_features.csv"


def _pick_col(df: pd.DataFrame, candidates: Tuple[str, ...]) -> Optional[str]:
    for col in candidates:
        if col in df.columns:
            return col
    return None


def _normalize_response_order(value: object) -> Optional[str]:
    if not isinstance(value, str):
        return None
    cleaned = value.strip().lower()
    if not cleaned:
        return None
    token = "".join(ch for ch in cleaned if ch.isalnum())
    if token.startswith("t1t2"):
        return "t1_t2"
    if token.startswith("t2t1"):
        return "t2_t1"
    if token in {"t1only", "t2only", "none"}:
        return token
    return None


def _coerce_bool_series(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    text = series.astype(str).str.strip().str.lower()
    mapped = text.map({
        "true": True,
        "false": False,
        "1": True,
        "0": False,
        "yes": True,
        "no": False,
    })
    return mapped.fillna(False).astype(bool)


@dataclass(frozen=True)
class PrpCols:
    pid: str
    soa: str
    rt2: str
    t2_correct: Optional[str]
    t2_timeout: Optional[str]
    response_order: Optional[str]


def _resolve_prp_columns(df: pd.DataFrame) -> PrpCols:
    pid = _pick_col(df, ("participant_id", "participantId", "pid", "subject_id", "subj_id"))
    soa = _pick_col(df, ("soa_nominal_ms", "soa", "soa_ms", "SOA", "soa_nominal"))
    rt2 = _pick_col(df, ("t2_rt_ms", "t2_rt", "rt2", "t2rt", "T2_RT"))

    if pid is None or soa is None or rt2 is None:
        raise ValueError(
            "Required columns not found. Need participant_id, SOA, and T2 RT.\n"
            f"Available columns: {list(df.columns)}"
        )

    t2_correct = _pick_col(df, ("t2_correct", "T2_correct", "t2_is_correct"))
    t2_timeout = _pick_col(df, ("t2_timeout", "T2_timeout", "t2_is_timeout"))
    response_order = _pick_col(df, ("response_order", "resp_order", "order", "order_norm"))

    return PrpCols(
        pid=pid,
        soa=soa,
        rt2=rt2,
        t2_correct=t2_correct,
        t2_timeout=t2_timeout,
        response_order=response_order,
    )


def _cbt_model(soa: np.ndarray, base: float, bottleneck: float) -> np.ndarray:
    return base + np.maximum(0.0, bottleneck - soa)


def _fit_cbt_model(
    soa_values: np.ndarray,
    rt_values: np.ndarray,
    rt_se: Optional[np.ndarray] = None,
    rt_min: float = PRP_RT_MIN,
    rt_max: float = PRP_RT_MAX,
    bottleneck_min: float = 50.0,
    bottleneck_max: float = 1500.0,
    maxfev: int = 20000,
) -> Dict[str, float]:
    if len(soa_values) < 3:
        return {"base": np.nan, "bottleneck": np.nan, "r2": np.nan, "rmse": np.nan}

    order = np.argsort(soa_values)
    soa_sorted = soa_values[order].astype(float)
    rt_sorted = rt_values[order].astype(float)

    sigma = None
    if rt_se is not None:
        se_sorted = rt_se[order].astype(float)
        if np.all(np.isfinite(se_sorted)) and np.all(se_sorted > 0):
            sigma = se_sorted

    base_init = float(np.mean(rt_sorted[-min(2, len(rt_sorted)) :]))
    min_idx = int(np.argmin(soa_sorted))
    min_soa = float(soa_sorted[min_idx])
    min_rt = float(rt_sorted[min_idx])
    bottleneck_init = float(
        np.clip(min_soa + max(0.0, min_rt - base_init), bottleneck_min, bottleneck_max)
    )

    try:
        params, _ = curve_fit(
            _cbt_model,
            soa_sorted,
            rt_sorted,
            p0=[base_init, bottleneck_init],
            bounds=([rt_min, bottleneck_min], [rt_max, bottleneck_max]),
            sigma=sigma,
            absolute_sigma=True if sigma is not None else False,
            maxfev=maxfev,
        )
        base, bottleneck = params
        preds = _cbt_model(soa_sorted, base, bottleneck)
        ss_res = float(np.sum((rt_sorted - preds) ** 2))
        ss_tot = float(np.sum((rt_sorted - np.mean(rt_sorted)) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
        rmse = float(np.sqrt(ss_res / len(rt_sorted)))
        return {"base": float(base), "bottleneck": float(bottleneck), "r2": r2, "rmse": rmse}
    except Exception:
        return {"base": np.nan, "bottleneck": np.nan, "r2": np.nan, "rmse": np.nan}


def compute_prp_bottleneck_features(
    data_dir: Path | None = None,
    min_trials: int = 30,
    min_soa_levels: int = 3,
    short_soa_max: int = 300,
    min_trials_per_soa: int = 3,
    use_measured_soa: bool = False,
    include_incorrect_trials_for_curve: bool = True,
    require_t1_to_t2_order: bool = False,
    apply_trial_filters: bool = True,
    apply_rt_bounds: bool = False,
    maxfev: int = 20000,
) -> pd.DataFrame:
    trials, _ = load_prp_trials(
        data_dir=data_dir,
        apply_trial_filters=apply_trial_filters,
    )
    cols = _resolve_prp_columns(trials)

    df = trials.copy()

    if use_measured_soa:
        measured = _pick_col(df, ("soa_measured_ms", "soa_measured"))
        if measured is not None:
            df[measured] = pd.to_numeric(df[measured], errors="coerce")
            df[cols.soa] = (df[measured] / 10.0).round() * 10.0

    df[cols.rt2] = pd.to_numeric(df[cols.rt2], errors="coerce")
    df[cols.soa] = pd.to_numeric(df[cols.soa], errors="coerce")
    df = df[df[cols.rt2].notna() & df[cols.soa].notna()].copy()

    if apply_rt_bounds:
        df = df[(df[cols.rt2] >= PRP_RT_MIN) & (df[cols.rt2] <= PRP_RT_MAX)].copy()

    if cols.t2_timeout is not None:
        df[cols.t2_timeout] = _coerce_bool_series(df[cols.t2_timeout])
        df = df[df[cols.t2_timeout] == False].copy()  # noqa: E712

    if require_t1_to_t2_order and cols.response_order is not None:
        if cols.response_order == "order_norm":
            order_norm = df["order_norm"]
        else:
            order_norm = df[cols.response_order].apply(_normalize_response_order)
            df["order_norm"] = order_norm
        df = df[order_norm == "t1_t2"].copy()

    if (not include_incorrect_trials_for_curve) and cols.t2_correct is not None:
        df[cols.t2_correct] = _coerce_bool_series(df[cols.t2_correct])
        df_curve = df[df[cols.t2_correct] == True].copy()  # noqa: E712
    else:
        df_curve = df.copy()

    df_short = df_curve[df_curve[cols.soa] <= float(short_soa_max)].copy()

    results = []
    for pid, g_all in df.groupby(cols.pid):
        g_curve = df_curve[df_curve[cols.pid] == pid]
        g_short = df_short[df_short[cols.pid] == pid]

        record = {
            "participant_id": pid,
            "prp_cb_base": np.nan,
            "prp_cb_bottleneck": np.nan,
            "prp_cb_r_squared": np.nan,
            "prp_cb_rmse": np.nan,
            "prp_cb_slope": np.nan,
            "prp_cb_n_trials": int(len(g_curve)),
            "prp_cb_n_soa_levels": int(g_curve[cols.soa].nunique()) if len(g_curve) else 0,
        }

        if len(g_curve) < min_trials or g_curve[cols.soa].nunique() < min_soa_levels:
            results.append(record)
            continue

        agg = (
            g_curve.groupby(cols.soa)[cols.rt2]
            .agg(rt_mean="mean", rt_sd="std", n="count")
            .reset_index()
            .rename(columns={cols.soa: "soa"})
        )
        agg = agg[agg["n"] >= int(min_trials_per_soa)].copy()

        if len(agg) < min_soa_levels:
            results.append(record)
            continue

        agg["rt_se"] = agg["rt_sd"] / np.sqrt(agg["n"].clip(lower=1))

        fit = _fit_cbt_model(
            agg["soa"].to_numpy(),
            agg["rt_mean"].to_numpy(),
            rt_se=agg["rt_se"].to_numpy(),
            maxfev=maxfev,
        )
        record["prp_cb_base"] = fit["base"]
        record["prp_cb_bottleneck"] = fit["bottleneck"]
        record["prp_cb_r_squared"] = fit["r2"]
        record["prp_cb_rmse"] = fit["rmse"]

        short = agg[agg["soa"] <= short_soa_max]
        if len(short) >= 2:
            slope, _ = np.polyfit(short["soa"], short["rt_mean"], 1)
            record["prp_cb_slope"] = float(slope)

        results.append(record)

    return pd.DataFrame(results)


def load_or_compute_prp_bottleneck_mechanism_features(
    data_dir: Path | None = None,
    overwrite: bool = False,
    save: bool = True,
    verbose: bool = False,
    **kwargs,
) -> pd.DataFrame:
    if data_dir is None:
        data_dir = get_results_dir("prp")

    output_path = data_dir / MECHANISM_FILENAME
    if output_path.exists() and not overwrite:
        return pd.read_csv(output_path, encoding="utf-8-sig")

    features = compute_prp_bottleneck_features(data_dir=data_dir, **kwargs)
    if save and not features.empty:
        features.to_csv(output_path, index=False, encoding="utf-8-sig")
        if verbose:
            print(f"[OK] PRP bottleneck features saved: {output_path}")
    return features
